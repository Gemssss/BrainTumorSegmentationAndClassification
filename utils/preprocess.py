from PIL import Image

def ensure_rgb(image: Image.Image) -> Image.Image:
    """
    Ensure image has 3 channels (RGB), as expected by CNNs like ResNet.
    """
    if image.mode != "RGB":
        return image.convert("RGB")
    return image

def center_crop_to_square(image: Image.Image) -> Image.Image:
    """
    Crop the input image to a square by taking the center region.
    Useful to avoid distortion before resizing.
    """
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return image.crop((left, top, right, bottom))
