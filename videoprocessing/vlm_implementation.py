from nebula3_videoprocessing.videoprocessing.vlm_interface import VlmInterface
import typing
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

class ClipVlmImplementation(VlmInterface):
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def compute_similarity(self, image : Image, text : list[str]):

        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        return probs.detach().numpy()[0]



def main():

    clip_vlm = ClipVlmImplementation()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    similarity = clip_vlm.compute_similarity(image, ["a photo of a cat", "a photo of a dog"])
    print(similarity)


if __name__ == "__main__":
    main()