import typing
from abc import ABC, abstractmethod
class VlmInterface(ABC):

    def __init__(self):
        super().__init__() 

    @abstractmethod
    def compute_similarity(self, image, text : list[str]) -> list[float]:
        pass
