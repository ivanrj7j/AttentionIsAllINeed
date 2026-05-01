from .encoderDecoderBlock import MultiHeadAttention
from .encoderDecoderBlock import PositionWiseFeedForward

import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    ### Encoder

    #### Explanation

    This is the first part of the transformer, to understand how encoder works let's think of an transformer as a translator who is translating what the speaker is saying.

    The input embedding is the voice coming out of the speaker

    The encoder is analagous to the translator understanding what the speaker is saying, using this understanding the translator can convert the speech to any other language.
    """
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