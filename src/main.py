import torch 
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
import time

# Training on random data TODO: use real data to train

srcVocabSize = 4000
tgtVocabSize = 4000
embeddingSize = 512
headCount = 8
numLayers = 6
dff = 128
maxSeqLen = 64
dropout = 0.1
# around 3.5gb vram usage with this configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)

transformer = Transformer(numLayers, srcVocabSize, tgtVocabSize, embeddingSize, headCount, dff, dropout, maxSeqLen, device).to(device)

# Generate random sample data
srcData = torch.randint(1, srcVocabSize, (64, maxSeqLen)).to(device)  # (batch_size, maxSeqLen)
tgtData = torch.randint(1, tgtVocabSize, (64, maxSeqLen)).to(device)  

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

scaler = torch.amp.grad_scaler.GradScaler()

transformer.train()

totalTime = 0
for epoch in range(100):
    start = time.time()
    optimizer.zero_grad()

    with torch.amp.autocast_mode.autocast(device_type='cuda', dtype=torch.float16):
        output = transformer(srcData, tgtData[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgtVocabSize), tgtData[:, 1:].contiguous().view(-1))

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    t = time.time() - start
    totalTime += t
    if epoch % 20 == 0:
        print(f"Epoch: {epoch+1}, Loss: {round(loss.item(), 3)} ({t}s)")
        if epoch > 0:
            torch.save(transformer.state_dict(), f'../checkpoints/transformer-{epoch+1}.pt')
            # size of our current model is 101 mb
print(f"Total time {round(totalTime)}s Average time {round(totalTime/100, 3)}s")