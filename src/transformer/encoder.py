from .encoderDecoderBlock import MultiHeadAttention
from .encoderDecoderBlock import PositionWiseFeedForward

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, embeddingSize:int, headCount:int, dff:int, dropout:float) -> None:
        super().__init__()
        assert embeddingSize % headCount == 0, "Embedding size should be a multiple of head count"

        self.attentionLayer = MultiHeadAttention(embeddingSize, headCount)
        self.feedForward = PositionWiseFeedForward(embeddingSize, dff)

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embeddingSize)
        self.norm2 = nn.LayerNorm(embeddingSize)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        attention = self.dropout.forward(self.attentionLayer.forward(x, x, x, mask))
        x = self.norm1(x + attention)
        ff = self.dropout(self.feedForward(x))
        return self.norm2(x + ff) 