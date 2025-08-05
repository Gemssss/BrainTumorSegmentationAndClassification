import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# === CLASSIFICATION MODEL (ResNet50) ===
class TumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(TumorClassifier, self).__init__()
        base_model = models.resnet50(weights=None)
        in_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        self.model = base_model

    def forward(self, x):
        return self.model(x)


# === ATTENTION BLOCK ===
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# === ATTENTION U-NET WITH RESNET34 ENCODER ===
class AttentionUNet(nn.Module):
    def __init__(self, n_classes=1):
        super(AttentionUNet, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.center = self.conv_block(512, 512)

        self.up4 = self.up_block(512, 256)
        self.att4 = AttentionBlock(256, 256, 128)
        self.dec4 = self.conv_block(512, 256)

        self.up3 = self.up_block(256, 128)
        self.att3 = AttentionBlock(128, 128, 64)
        self.dec3 = self.conv_block(256, 128)

        self.up2 = self.up_block(128, 64)
        self.att2 = AttentionBlock(64, 64, 32)
        self.dec2 = self.conv_block(128, 64)

        self.up1 = self.up_block(64, 64)
        self.att1 = AttentionBlock(64, 64, 32)
        self.dec1 = self.conv_block(128, 64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(self.pool0(enc0))
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        center = self.center(enc4)

        up4 = self.up4(center)
        att4 = self.att4(g=up4, x=enc3)
        dec4 = self.dec4(torch.cat([up4, att4], dim=1))

        up3 = self.up3(dec4)
        att3 = self.att3(g=up3, x=enc2)
        dec3 = self.dec3(torch.cat([up3, att3], dim=1))

        up2 = self.up2(dec3)
        att2 = self.att2(g=up2, x=enc1)
        dec2 = self.dec2(torch.cat([up2, att2], dim=1))

        up1 = self.up1(dec2)
        att1 = self.att1(g=up1, x=enc0)
        dec1 = self.dec1(torch.cat([up1, att1], dim=1))

        out = self.final(dec1)
        return out