import torch 
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
import time

# Training on random data TODO: use real data to train

srcVocabSize = 3500
tgtVocabSize = 4000
embeddingSize = 256
headCount = 4
numLayers = 3
dff = 128
maxSeqLen = 100
dropout = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)

transformer = Transformer(numLayers, srcVocabSize, tgtVocabSize, embeddingSize, headCount, dff, dropout, maxSeqLen, device).to(device)

# Generate random sample data
srcData = torch.randint(1, srcVocabSize, (64, maxSeqLen)).to(device)  # (batch_size, maxSeqLen)
tgtData = torch.randint(1, tgtVocabSize, (64, maxSeqLen)).to(device)  

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

totalTime = 0
for epoch in range(100):
    start = time.time()
    optimizer.zero_grad()
    output = transformer(srcData, tgtData[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgtVocabSize), tgtData[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    t = time.time() - start
    totalTime += t
    if epoch % 20 == 0:
        print(f"Epoch: {epoch+1}, Loss: {round(loss.item(), 3)} ({t}s)")
print(f"Total time {round(totalTime)}s Average time {round(totalTime/100, 3)}s")