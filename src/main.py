import torch 
from transformer import Encoder, Decoder
import time

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    encoder = Encoder(512, 8, 64, 0.3).to(device)
    decoder = Decoder(512, 8, 64, 0.3).to(device)
    inp = torch.rand(16, 128, 512).to(device)
    mask = torch.ones((16, 1, 1, 128)).to(device)

    totalTime = 0
    n = 30
    for _ in range(n):
        startTime = time.time()
        enc = encoder.forward(inp, mask)
        assert inp.size() == enc.size()
        dec = decoder.forward(inp, enc, mask, mask)
        assert enc.size() == dec.size()
        totalTime += time.time() - startTime

    

    print("Total time:", round(totalTime, 2), "Average time:", round(totalTime/n, 2))