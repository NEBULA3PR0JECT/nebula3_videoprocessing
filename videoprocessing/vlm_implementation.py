from nebula3_videoprocessing.videoprocessing.vlm_interface import VlmInterface
from nebula3_videoprocessing.videoprocessing.utils.config import config
import typing
from PIL import Image
import requests
import torch

from transformers import CLIPProcessor, CLIPModel
from nebula3_vlmtokens_expert.vlmtokens.models.blip_itm import blip_itm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class VlmBaseImplementation(VlmInterface):

    def compute_similarity_url(self, url: str, text: list[str]):
        image = self.load_image_url(url)
        return self.compute_similarity(image, text)
    
    

class ClipVlmImplementation(VlmBaseImplementation):
    def __init__(self):
        self.model = CLIPModel.from_pretrained(config["clip_checkpoints"])
        self.processor = CLIPProcessor.from_pretrained(config["clip_checkpoints"])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device=self.device)
        self.model.eval()


    def load_image_url(self, url: str):
        return Image.open(requests.get(url, stream=True).raw)  

    def compute_similarity(self, image : Image, text : list[str]):

        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
# Need to align all the inputs to the same device 
        vv = dict()
        [vv.update({k: v.to(device=self.device)}) for k, v in inputs.items()]
        inputs = vv
        
        outputs = self.model(**inputs)
        embeds_dotproduct = (outputs.image_embeds.expand_as(outputs.text_embeds) * outputs.text_embeds).sum(dim=1)
        return embeds_dotproduct.cpu().detach().numpy()

class BlipItmVlmImplementation(VlmBaseImplementation):
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = blip_itm(pretrained=config['blip_model_url_base'], image_size=config['blip_image_size'], vit=config['blip_vit_base'])
        model.eval()
        self.model = model.to(device=self.device)
    
    def load_image_url(self, url: str):
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')  
        return image

    def load_image(self, image: Image): 
        transform = transforms.Compose([
            transforms.Resize((config['blip_image_size'], config['blip_image_size']),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(image).unsqueeze(0).to(self.device)   
        return image

    def compute_similarity(self, image: Image, text: list[str]):
        
        image = self.load_image(image)
        outputs = []
        for txt in text:
            caption = txt
            
            itm_output = self.model(image, caption, match_head='itm')
            # Change from softmax to dotproduct
            itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
            itm_score = itm_score.cpu().detach().numpy()[0]
            outputs.append(itm_score)

        return outputs[0]


class BlipItcVlmImplementation(VlmBaseImplementation):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = blip_itm(pretrained=config['blip_model_url_base'], image_size=config['blip_image_size'], vit=config['blip_vit_base'])
        model.eval()
        self.model = model.to(device=self.device)
    
    def load_image_url(self, url: str):
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB') 
        return image

    def load_image(self, image):   
        
        transform = transforms.Compose([
            transforms.Resize((config['blip_image_size'], config['blip_image_size']),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(image).unsqueeze(0).to(self.device)   
        return image

    def compute_similarity(self, image : Image, text : list[str]):
        image = self.load_image(image)
        outputs = []
        outputs = self.model(image, text, match_head='itc')

        return outputs[0]


def main():
    ### CLIP USAGE EXAMPLE ###
    clip_vlm = ClipVlmImplementation()

    url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    text =['a woman sitting on the beach with a dog', 'a man standing on the beach with a cat']
    similarity = clip_vlm.compute_similarity_url(url, text)
    print(f"CLIP outputs: {similarity}")

    ##################################

    ### BLIP ITM USAGE EXAMPLE ###

    blip_vlm = BlipItmVlmImplementation()
    text = ['a woman sitting on the beach with a dog']
    similarity = blip_vlm.compute_similarity_url(url, text)
    itm_score = similarity
    print(f"BLIP_ITM outputs:")
    print('The image and text is matched with a probability of %.4f'%itm_score)

    ## BLIP ITC USAGE EXAMPLE ###
    blip_vlm = BlipItcVlmImplementation()
    text = ['a woman sitting on the beach with a dog']
    similarity = blip_vlm.compute_similarity_url(url, text)
    itc_score = similarity
    print(f"BLIP_ITC outputs:")
    print('The image and text is matched with a probability of %.4f'%itc_score)




if __name__ == "__main__":
    main()