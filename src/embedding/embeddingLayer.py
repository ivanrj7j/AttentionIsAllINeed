import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast

from typing import Literal

class EmbeddingLayer(nn.Module):
    def __init__(self, embeddingLayerAndPositonalENcoder:nn.Sequential, tokenizer:PreTrainedTokenizerFast, device:torch.device, maxSentenceSize:int) -> None:
        super().__init__()
        self.layers = embeddingLayerAndPositonalENcoder
        self.tokenizer = tokenizer
        self.device = device
        self.maxSentenceSize = maxSentenceSize
        self.layers.eval()
        self.eval()

        self.layers.to(self.device)

    def forward(self, x:list[str], embeddingMethod:Literal['AVG', 'FIRST']='AVG'):
        tokens = self.tokenizer(
                x,
                max_length=self.maxSentenceSize, 
                padding='max_length', 
                truncation=True, 
                add_special_tokens=True,
                return_tensors='pt'
                )['input_ids'].to(self.device)
        
        with torch.no_grad():
            output = self.layers(tokens)
            if embeddingMethod == 'FIRST':
                output = output[:, 0, :]
            else:
                output = torch.mean(output, dim=1)

            return output