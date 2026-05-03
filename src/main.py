import torch 
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerFast
from .dataset import TransformerDataset
from uuid import uuid4
import time

import config


transformer = Transformer(config.NUM_LAYERS, config.SRC_VOCAB_SIZE, config.TGT_VOCAB_SIZE, config.EMBEDDING_SIZE, config.HEAD_COUNT, config.DFF, config.DROPOUT, config.MAX_SEQ_LEN, config.TARGET_DEVICE).to(config.TARGET_DEVICE)

# Dataset and tokenizer
srcTokenizer = PreTrainedTokenizerFast(tokenizer_file=config.SRC_TOKENIZER,  
                                 bos_token="<bos>", 
                                 eos_token="<eos>",
                                 pad_token="<pad>",
                                 unk_token="<unk>")
tgtTokenizer = PreTrainedTokenizerFast(tokenizer_file=config.TGT_TOKENIZER,  
                                 bos_token="<bos>", 
                                 eos_token="<eos>",
                                 pad_token="<pad>",
                                 unk_token="<unk>")
trainingDataset = TransformerDataset(config.SRC_FILE_TRAIN, config.TGT_FILE_TRAIN, config.MAX_SENTENCE_SIZE, config.BUFFER_SIZE, srcTokenizer, tgtTokenizer)
validationDataset = TransformerDataset(config.SRC_FILE_VAL, config.TGT_FILE_VAL, config.MAX_SENTENCE_SIZE, config.BUFFER_SIZE, srcTokenizer, tgtTokenizer)
testDataset = TransformerDataset(config.SRC_FILE_TEST, config.TGT_FILE_TEST, config.MAX_SENTENCE_SIZE, config.BUFFER_SIZE, srcTokenizer, tgtTokenizer)

trainingDataLoader = trainingDataset.getDataLoader(config.BATCH_SIZE, config.TRAIN_WORKERS)
validationDataLoader = trainingDataset.getDataLoader(config.BATCH_SIZE, config.VALIDATION_WORKERS)
testDataLoader = trainingDataset.getDataLoader(config.BATCH_SIZE, config.TEST_WORKERS)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS, eps=config.EPSILON)

scaler = torch.amp.grad_scaler.GradScaler()

transformer.train()

totalTime = 0


totalSteps = config.EPOCHS * config.STEPS_PER_EPOCH

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr = config.LEARNING_RATE,
    total_steps=totalSteps,
    pct_start=config.PCT_START,
    anneal_strategy=config.ANNEAL_STRATEGY,
    div_factor=config.DIV_FACTOR,
    final_div_factor=config.FINAL_DIV_FACTOR
)

summaryWriter = SummaryWriter(config.SUMMARY_PATH)

for epoch in range(config.EPOCHS):
    start = time.time()
    optimizer.zero_grad()

    with torch.amp.autocast_mode.autocast(device_type='cuda', dtype=torch.float16):
        output = transformer(srcData, tgtData[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgtVocabSize), tgtData[:, 1:].contiguous().view(-1))

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    scheduler.step()

    t = time.time() - start
    totalTime += t
    currentLR = optimizer.param_groups[0]['lr']
    if epoch % 20 == 0:
        print(f"Epoch: {epoch+1}, Loss: {round(loss.item(), 3)} ({t}s), LR: {currentLR:.2e}")
        if epoch > 0:
            torch.save(transformer.state_dict(), f'../checkpoints/transformer-{epoch+1}.pt')
            # size of our current model is 101 mb
    summaryWriter.add_scalar('Charts/lr', currentLR, epoch)
    summaryWriter.add_scalar('Charts/loss', loss.item(), epoch)
print(f"Total time {round(totalTime)}s Average time {round(totalTime/100, 3)}s")