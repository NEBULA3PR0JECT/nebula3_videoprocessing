import torch
import clip


class ClipVideoUtils:
    """
    The class provides a number of utils for CLIP based video processing
    """
    def __init__(self, model_name='RN50x4'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model_res = 640
        if model_name == 'ViT-B/32':
            self.model_res = 512
        if model_name == 'ViT-L/14':
            self.model_res = 768