import torch 
from transformer import Encoder, Decoder, Transformer
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

    maxSeqLen = 100
    srcVocabSize, tgtVocabSize = 2000, 2000
    transformer = Transformer(1, srcVocabSize, tgtVocabSize, 256, 4, 32, 0.3, maxSeqLen)

    srcData = torch.randint(1, srcVocabSize, (64, maxSeqLen)) # (batch of 64 with maxSeqLen tokens)
    tgtData = torch.randint(1, tgtVocabSize, (64, maxSeqLen))

    startTime = time.time()
    pred = transformer.forward(srcData, tgtData)
    output = torch.softmax(pred, dim = -1)
    timeTaken = time.time() - startTime

    print("Transformer took", timeTaken, "seconds")
    print(pred.size())