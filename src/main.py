import torch 
from transformer import Encoder

if __name__ == "__main__":
    encoder = Encoder(512, 8, 64, 0.3)
    inp = torch.rand(16, 128, 512)
    mask = torch.ones((16, 1, 1, 128))
    op = encoder.forward(inp, mask)
    assert inp.size() == op.size()

    print("Input:", inp.size(), "Output:", op.size())   