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
    
    def load_image(self, url):

        image = Image.open(requests.get(url, stream=True).raw)
        return image

    def compute_similarity(self, image : Image, text : list[str]):

        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)

        outputs = self.model(**inputs)
        embeds_dotproduct = (outputs.image_embeds.expand_as(outputs.text_embeds) * outputs.text_embeds).sum(dim=1)
        return embeds_dotproduct.detach().numpy()

class BlipItmVlmImplementation(VlmInterface):
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # This is COCO checkpoints, there's also Flicker
        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
        self.image_size = 384
        #@TODO: change to large
        model = blip_itm(pretrained=model_url, image_size=self.image_size, vit='base')
        model.eval()
        self.model = model.to(device=self.device)
    
    def load_image(self, url): 
        raw_image = Image.open(requests.get(url, stream=True).raw).convert('RGB')   
        
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
            # Change from softmax to dotproduct
            itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
            itm_score = itm_score.cpu().detach().numpy()[0]
            outputs.append(itm_score)

        return outputs


class BlipItcVlmImplementation(VlmInterface):
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # This is COCO checkpoints, there's also Flicker
        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
        self.image_size = 384
        #@TODO: change to large
        model = blip_itm(pretrained=model_url, image_size=self.image_size, vit='base')
        model.eval()
        self.model = model.to(device=self.device)
    
    def load_image(self, url): 
        raw_image = Image.open(requests.get(url, stream=True).raw).convert('RGB')   
        
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

            itc_output = self.model(image,caption,match_head='itc')
            # Check if its dotproduct
            itc_score = itc_output.cpu().detach().numpy()[0][0]
            outputs.append(itc_score)
        return outputs


def main():

    ### CLIP USAGE EXAMPLE ###
    clip_vlm = ClipVlmImplementation()

    image = clip_vlm.load_image(url="http://images.cocodataset.org/val2017/000000039769.jpg")
    text =['a woman sitting on the beach with a dog', 'a man standing on the beach with a cat']
    similarity = clip_vlm.compute_similarity(image, text)
    print(f"CLIP outputs: {similarity}")
    ##################################

    ### BLIP USAGE EXAMPLE ###

    blip_vlm = BlipItmVlmImplementation()
    text = ['a woman sitting on the beach with a dog']
    image = blip_vlm.load_image()
    similarity = blip_vlm.compute_similarity(image, text)
    itm_score = similarity[0]
    print(f"BLIP outputs:")
    # print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)
    print('The image and text is matched with a probability of %.4f'%itm_score)




if __name__ == "__main__":
    main()