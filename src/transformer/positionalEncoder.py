import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):  
    """
    ### Positional encoder

    ##### Explanation

    Positional encoder is the first layer of the transformer after the embedding layer. As the name suggests, it is used to encode the position of each words in the input.

    ##### Why do we need one?

    Even though we are passing a tensor of (batch, word, embedding) this is not enough because attention block cannot differentiate between a batch of (hello, my, name, is, ivan) and (ivan, is, hello, name, my) to we train a positional encoder to tackle this problem so the attention head knows that where each word belongs to.
    """    
    def __init__(self, embeddingSize:int, maximumTokens:int) -> None:
        super().__init__()

        pe = torch.zeros(maximumTokens, embeddingSize)
        position = torch.arange(0, maximumTokens, dtype=torch.float).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, embeddingSize, 2).float() * -(math.log(10000.0) / embeddingSize))

        frequency = position*divTerm

        pe[:, 0::2] = torch.sin(frequency)
        pe[:, 1::2] = torch.cos(frequency)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x:torch.Tensor):
        return x + self.pe[:, :x.size(1)]
    
if __name__ == '__main__':
    pe = PositionalEncoder(512, 256)
    inp = torch.rand((16, 128, 512))
    op = pe.forward(inp)

    assert inp.size() == op.size()
    print(inp.size(), op.size())
    print(inp[0, 0, :16])
    print(op[0, 0, :16])
    print('-'*16)

    inp = torch.zeros((16, 128, 512))
    op = pe.forward(inp)
    
    assert inp.size() == op.size()
    print(op[0, 1, :16])