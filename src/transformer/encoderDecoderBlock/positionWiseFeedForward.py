import torch.nn as nn
import torch

class PositionWiseFeedForward(nn.Module):
    """
    ### Position wise feed forward

    ##### Explanation

    Position wise feed forward is used to process the output of attention block to introduce non linearity (not to be confused with positonal encoder)
    """
    def __init__(self, embeddingSize:int, dff:int) -> None:
        super().__init__()
        self.f1 = nn.Linear(embeddingSize, dff)
        self.f2 = nn.Linear(dff, embeddingSize)
        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor):
        return self.f2.forward(self.relu.forward(self.f1.forward(x)))
    
if __name__ == "__main__":
    positionalEncoder = PositionWiseFeedForward(512, 64)
    inp = torch.rand((16, 128, 512))
    output = positionalEncoder.forward(inp)
    print("Input:", inp.size(), "Output:", output.size())