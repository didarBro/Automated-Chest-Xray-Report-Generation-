import os
import torch
from django.conf import settings
from django.shortcuts import render
from .model import ViT_GPT2
from .utils import load_image
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchvision import transforms

# Define image transformations for ViT
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>'})

# Define cleanup function for generated reports
def clean_report(generated_report):
    unwanted_tokens = ['<|startoftext|>', '<|endoftext|>', '[PAD]']
    for token in unwanted_tokens:
        generated_report = generated_report.replace(token, '')
    generated_report = ' '.join(generated_report.split()).strip()
    return generated_report

def landing_page(request):
    generated_report = None
    image_url = None

    if request.method == 'POST' and request.FILES['image']:
        # Handle file upload
        image = request.FILES['image']
        
        # Define the path to save the uploaded image in the media folder
        image_path = os.path.join(settings.MEDIA_ROOT, image.name)
        with open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)
        
        # Load the image using the utility function
        image_tensor, image_pil = load_image(image_path)  # Assume load_image returns a tuple (tensor, PIL image)
        
        # Apply the image transformation on the PIL image (not the tensor)
        image_tensor = image_transform(image_pil).unsqueeze(0)  # Apply transform and add batch dimension
        
        # Load the ViT model and GPT-2 model
        vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        vit_output_dim = 768
        model = ViT_GPT2(vit_model, gpt2_model, vit_output_dim)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Load the trained model weights
        model_path = os.path.join(settings.BASE_DIR, 'report_generation', 'model_weights', 'vit_gpt2_model.pth')
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle mismatch in token embeddings
        del state_dict['gpt2.transformer.wte.weight']
        del state_dict['gpt2.lm_head.weight']
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Generate the report and explicitly pass the device
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            generated_ids = model.generate(image_tensor, device=device, max_length=256, num_beams=5, repetition_penalty=2.0, early_stopping=True)

        # Decode the generated report using the tokenizer
        generated_report = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        
        # Clean the generated report
        generated_report = clean_report(generated_report)
        
        # Use the URL to serve the image through Django's media URL
        image_url = os.path.join(settings.MEDIA_URL, image.name)

    return render(request, 'landing_page.html', {'report': generated_report, 'image_url': image_url})
