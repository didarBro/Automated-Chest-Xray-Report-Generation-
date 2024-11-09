from PIL import Image
from torchvision import transforms

def load_image(image_path, image_size=224):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor, image
