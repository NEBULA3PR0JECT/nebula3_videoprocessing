from nebula3_videoprocessing.videoprocessing.vlm_interface import VlmInterface
import typing
from PIL import Image
import requests
import torch

from transformers import CLIPProcessor, CLIPModel
from models.blip_itm import blip_itm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

class ClipVlmImplementation(VlmInterface):
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def load_image(self):
        url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
        image = Image.open(requests.get(url, stream=True).raw)
        return image

    def compute_similarity(self, image : Image, text : list[str]):

        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        return probs.detach().numpy()[0]

class BlipVlmImplementation(VlmInterface):
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
        self.image_size = 384
        model = blip_itm(pretrained=model_url, image_size=self.image_size, vit='base')
        model.eval()
        self.model = model.to(device=self.device)
    
    def load_image(self):
        img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   
        
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(raw_image).unsqueeze(0).to(self.device)   
        return image

    def compute_similarity(self, image : Image, text : list[str]):

        outputs = []
        for txt in text:
            caption = txt
            
            itm_output = self.model(image, caption, match_head='itm')
            itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
            itm_score = itm_score.cpu().detach().numpy()[0]

            itc_score = self.model(image,caption,match_head='itc')
            itc_score = itc_score.cpu().detach().numpy()[0][0]
            outputs.append([itm_score, itc_score])
        return outputs[0]


def main():

    ### CLIP USAGE EXAMPLE ###
    clip_vlm = ClipVlmImplementation()

    image = clip_vlm.load_image()
    text =['a woman sitting on the beach with a dog', 'a man standing on the beach with a cat']
    similarity = clip_vlm.compute_similarity(image, text)
    print(f"CLIP outputs: {similarity}")
    ##################################

    ### BLIP USAGE EXAMPLE ###

    blip_vlm = BlipVlmImplementation()
    text = ['a woman sitting on the beach with a dog']
    image = blip_vlm.load_image()
    similarity = blip_vlm.compute_similarity(image, text)
    itm_score = similarity[0]
    itc_score = similarity[1]
    print(f"BLIP outputs:")
    print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)
    print('The image and text is matched with a probability of %.4f'%itm_score)




if __name__ == "__main__":
    main()