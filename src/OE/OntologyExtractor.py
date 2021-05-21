import os
from abc import ABC, abstractmethod
from transformers import XLMRobertaTokenizer
from .Struct import Struct
import torch
from pymorphy2 import MorphAnalyzer


class OntologyExtractor(ABC):
    @abstractmethod
    def __init__(self):
        self.device = None
        self.model = None
        self.tokenizer = None
        self.morph = MorphAnalyzer()
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        self.data = Struct()

    def load_model(self, model: str):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    def run(self, data):
        if self.model is None:
            raise ValueError("There is no model. Please run load_model() before run()")
