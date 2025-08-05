# utils/transforms.py

from torchvision import transforms

# Classification transform (ResNet50 expects RGB 224x224)
classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Segmentation transform (input image resized to 224x224)
segmentation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
