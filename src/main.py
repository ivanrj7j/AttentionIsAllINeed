import torch 
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from dataset import TransformerDataset
import os
import time
import random 

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
totalTime = 0
totalSteps = round(config.EPOCHS * config.TRAIN_BATCHES * 1.05)

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


def get_random_translation_sample(srcBatch, tgtBatch, srcTokenizer, tgtTokenizer):
    """
    Takes a batch of tensors and returns a random (Source, Target) string pair.
    """
    # 1. Pick a random index from the batch (dim 0)
    batch_size = srcBatch.size(0)
    random_idx = random.randint(0, batch_size - 1)

    # 2. Extract the specific sequences and convert to list
    # We move to CPU because tokenizers expect standard Python lists/arrays
    src_ids = srcBatch[random_idx].cpu().tolist()
    tgt_ids = tgtBatch[random_idx].cpu().tolist()

    # 3. Decode back to strings
    # skip_special_tokens=True removes <pad>, <bos>, <eos> automatically
    src_text = srcTokenizer.decode(src_ids, skip_special_tokens=True)
    tgt_text = tgtTokenizer.decode(tgt_ids, skip_special_tokens=True)

    return src_text.strip(), tgt_text.strip()
# get_random_translation_sample is ai generated 

def run_validation(model, dataloader, criterion, device, vocab_size, srcTokenizer, tgtTokenizer, summaryWriter, global_step, max_batches=100):
    model.eval()
    total_loss = 0
    batches_processed = 0
    
    # Initialize as None
    last_data = None

    with torch.no_grad():
        for i, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)

            with torch.amp.autocast_mode.autocast(device_type='cuda', dtype=torch.float16):
                output = model(src, tgt[:, :-1])
                loss = criterion(
                    output.contiguous().view(-1, vocab_size), 
                    tgt[:, 1:].contiguous().view(-1)
                )

            total_loss += loss.item()
            batches_processed += 1
            
            # Pack them into a tuple
            last_data = (src, tgt, output)
            
            if batches_processed >= max_batches:
                break

    # --- Sample Generation Logic ---
    # Check if last_data exists
    if last_data is not None:
        # Unpack them here - Pylance now knows they aren't None
        s_batch, t_batch, o_batch = last_data
        
        idx = random.randint(0, s_batch.size(0) - 1)
        
        # Decode using the unpacked tensors
        src_str = srcTokenizer.decode(s_batch[idx].tolist(), skip_special_tokens=True)
        tgt_str = tgtTokenizer.decode(t_batch[idx].tolist(), skip_special_tokens=True)
        
        # Use argmax on the output batch
        pred_ids = o_batch[idx].argmax(dim=-1).tolist()
        pred_str = tgtTokenizer.decode(pred_ids, skip_special_tokens=True)

        # ... (rest of your logging code) ...
        log_text = f"\n{'='*30}\n[VAL SAMPLE]\nSRC: {src_str}\nTGT: {tgt_str}\nPRD: {pred_str}\n{'='*30}"
        print(log_text)
        
        tb_log = f"**Source:** {src_str}  \n**Target:** {tgt_str}  \n**Predicted:** {pred_str}"
        summaryWriter.add_text('Validation/Samples', tb_log, global_step)

    model.train()
    return total_loss / batches_processed

# run validation is ai generated method 


# training on test data for the first time to see if the model trains properly before investing time in the wholedataset, these models will be discarded after run is verified 
if __name__ == '__main__':
    globalSteps = 0

        
    # Load one fixed batch for quick "live" checks
    val_iter = iter(validationDataLoader)
    fixed_val_src, fixed_val_tgt = next(val_iter)
    fixed_val_src = fixed_val_src.to(config.TARGET_DEVICE)
    fixed_val_tgt = fixed_val_tgt.to(config.TARGET_DEVICE)
    
    for epoch in range(config.EPOCHS):
        transformer.train()
        
        epochStart = time.time()
        bar = tqdm(enumerate(trainingDataLoader), total=config.TRAIN_BATCHES, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for i, (srcBatch, tgtBatch) in bar:
            if i >= config.TRAIN_BATCHES:
                break
            optimizer.zero_grad()
            batchStart = time.time()
            srcBatch, tgtBatch = srcBatch.to(config.TARGET_DEVICE), tgtBatch.to(config.TARGET_DEVICE)
            with torch.amp.autocast_mode.autocast(device_type='cuda', dtype=torch.float16):
                output = transformer(srcBatch, tgtBatch[:, :-1])
                loss = criterion(output.contiguous().view(-1, config.TGT_VOCAB_SIZE), tgtBatch[:, 1:].contiguous().view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            t = time.time() - batchStart
            currentLR = optimizer.param_groups[0]['lr']
            bar.set_postfix({
                "loss": loss.item(),
                "lr": currentLR
            })
            if globalSteps % config.LOG_EVERY == 0 and globalSteps > 0:
                summaryWriter.add_scalar('Charts/BatchLR', currentLR, globalSteps)
                summaryWriter.add_scalar('Charts/BatchLoss', loss.item(), globalSteps)
                summaryWriter.add_scalar('Charts/BatchTime', t, globalSteps)

            if globalSteps % config.TRANSLATE_EVERY == 0 and globalSteps > 0:
                transformer.eval()
                with torch.no_grad():
                    # Pick a random sample from our pre-loaded batch
                    idx = random.randint(0, fixed_val_src.size(0) - 1)
                    s_ten, t_ten = fixed_val_src[idx:idx+1], fixed_val_tgt[idx:idx+1]

                    with torch.amp.autocast_mode.autocast(device_type='cuda', dtype=torch.float16):
                        output = transformer(s_ten, t_ten[:, :-1])
                    
                    # Decode and clean strings
                    src_str = srcTokenizer.decode(s_ten[0].tolist(), skip_special_tokens=True)
                    tgt_str = tgtTokenizer.decode(t_ten[0].tolist(), skip_special_tokens=True)
                    pred_str = tgtTokenizer.decode(output.argmax(dim=-1)[0].tolist(), skip_special_tokens=True)

                    # Single-line output to keep the terminal clean
                    print(f"Step {globalSteps} | SRC: {src_str[:40]}... | PRD: {pred_str[:40]}...")
                    
                    summaryWriter.add_text('Samples/Validation', f"**SRC:** {src_str}  \n**TGT:** {tgt_str}  \n**PRD:** {pred_str}", globalSteps)

                transformer.train()
            globalSteps += 1

        epochTime = time.time() - epochStart
        if epoch % config.SAVE_MODEL_EVERY == 0 and config.EPOCHS > 1:
            torch.save(transformer.state_dict(), os.path.join(config.CHECKPOINT_PATH, f'transformer-{epoch+1}.pt'))

        summaryWriter.add_scalar('Charts/loss', loss.item(), epoch)
        summaryWriter.add_scalar('Charts/time', epochTime)
        summaryWriter.add_scalar('Charts/lr', currentLR)

        valLoss = run_validation(
                    transformer, 
                    validationDataLoader, 
                    criterion, 
                    config.TARGET_DEVICE, 
                    config.TGT_VOCAB_SIZE,
                    srcTokenizer,    
                    tgtTokenizer,    
                    summaryWriter,   
                    totalSteps,      
                    max_batches=20
                )
        summaryWriter.add_scalar('Charts/valLoss', valLoss, epoch)


    torch.save(transformer.state_dict(), os.path.join(config.CHECKPOINT_PATH, 'transformer-final.pt'))