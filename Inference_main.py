"""
This is a python file to run inference of the models

Name : Aditya Ramanath Poonja
email: adityapoonja@live.com

"""

from models.blip import blip_decoder
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


########################################################################################################################

def load_demo_image(image_size, device):
    """
    function to load a demo image
    """
    img_url = 'InputImages/demo.jpg'
    #raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    # img_url = '/content/drive/My Drive/My_Software_Projects/Input_Frames/time0_frame1.jpg'
    raw_image = Image.open(img_url).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    image = transform(raw_image).unsqueeze(0).to(device)
    return image


########################################################################################################################

if __name__ == '__main__':

    image_size = 384
    print("Loading image....")
    image = load_demo_image(image_size=image_size, device=device)
    print("Image Loaded successfully.")
    model_url = 'PreTrained/model_base_capfilt_large.pth'
    print("Loading model...")
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    print("Model Loaded successfully.")
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        print('caption: ' + caption[0])
