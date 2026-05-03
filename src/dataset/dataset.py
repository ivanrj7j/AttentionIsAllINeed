from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import PreTrainedTokenizerFast
import random

class TransformerDataset(IterableDataset):
    def __init__(self, srcFile:str, tgtFile:str, maxSentenceSize:int, bufferSize:int, srcTokenizer:PreTrainedTokenizerFast, tgtTokenizer:PreTrainedTokenizerFast) -> None:
        super().__init__()

        self.srcFile = srcFile
        self.tgtFile = tgtFile
        self.maxSentenceSize = maxSentenceSize
        self.bufferSize = bufferSize

        self.srcTokenizer = srcTokenizer
        self.tgtTokenizer = tgtTokenizer

    def tokenizeString(self, string:str|list[str], tokenizer:PreTrainedTokenizerFast):
        return tokenizer(
            string,
            max_length=self.maxSentenceSize, 
            padding='max_length', 
            truncation=True, 
            add_special_tokens=True
        )['input_ids']
    
    def __processBatch(self, src:list[str], tgt:list[str]):
        assert len(src) == len(tgt)

        srcTokens = self.tokenizeString(src, self.srcTokenizer)
        tgtTokens = self.tokenizeString(tgt, self.tgtTokenizer)

        for a, b in zip(srcTokens, tgtTokens):
            yield (torch.tensor(a), torch.tensor(b))

    def __shuffledProcess(self, srcBuffer:list[str], tgtBuffer:list[str]):
        combined = list(zip(srcBuffer, tgtBuffer))
        random.shuffle(combined)
        s, t = zip(*combined)

        yield from self.__processBatch(list(s), list(t))
    
    def __iter__(self) -> Iterator:
        workerInfo = get_worker_info()
        srcBuffer, tgtBuffer = [], []

        with open(self.srcFile, 'r', encoding='utf-8') as srcF, \
            open(self.tgtFile, 'r', encoding='utf-8') as tgtF:
            for i, (srcLine, tgtLine) in enumerate(zip(srcF, tgtF)):
                if workerInfo is not None:
                    if i % workerInfo.num_workers != workerInfo.id:
                        continue # distributing work among each workes 

                srcBuffer.append(srcLine)
                tgtBuffer.append(tgtLine)

                if len(srcBuffer) >= self.bufferSize:
                    yield from self.__shuffledProcess(srcBuffer, tgtBuffer)

                    srcBuffer, tgtBuffer = [], []
            if len(srcBuffer) > 0:
                yield from self.__shuffledProcess(srcBuffer, tgtBuffer)

    def getDataLoader(self, batchSize:int, nWorkers:int) -> DataLoader:
        return DataLoader(dataset=self, batch_size=batchSize, num_workers=nWorkers, pin_memory=True, drop_last=True)

if __name__ == "__main__":
    import time

    srcFile = '../../dataset/engTrain.txt'
    tgtFile = '../../dataset/malTrain.txt'

    srcTokenizer = PreTrainedTokenizerFast(tokenizer_file='../../dataset/engTokenizer.json',  
                                 bos_token="<bos>", 
                                 eos_token="<eos>",
                                 pad_token="<pad>",
                                 unk_token="<unk>")
    tgtTokenizer = PreTrainedTokenizerFast(tokenizer_file='../../dataset/malTokenizer.json', 
                                 bos_token="<bos>", 
                                 eos_token="<eos>",
                                 pad_token="<pad>",
                                 unk_token="<unk>")

    dataset = TransformerDataset(srcFile, tgtFile, 64 + 2, 1000, srcTokenizer, tgtTokenizer)
    loader = dataset.getDataLoader(64, 5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    print("Starting the dataset")

    startTime = time.time()
    iterations = 0
    for i, (x, y) in enumerate(dataset):
        z = x+y
        iterations += 1
        if i % 1e4 == 0:
            print(f"{i+1} Iterations", end="\r")
    total = time.time() - startTime
    print(f"Total time to consume {iterations} datapoints: {round(total)}s. Average time: {total/iterations}s")
    # Train: Total time to consume 4884066 datapoints: 202s. Average time: 4.145243995640283e-05s
    # Validation: Total time to consume 576604 datapoints: 24s. Average time: 4.1719547734728726e-05s
    # Test: Total time to consume 287624 datapoints: 12s. Average time: 4.1570987860039006e-05s
    """

    print("Starting the dataloader")
    iterations = 0
    startTime = time.time()
    for i, (srcBatch, tgtBatch) in enumerate(loader):
        srcBatch.to(device)
        tgtBatch.to(device)
        x = srcBatch + tgtBatch
        iterations += 1
        if i % 156 == 0:
            print(f"{i+1} Batches", end="\r")
    total = time.time() - startTime
    print(f"Total time to consume {iterations} batches: {round(total)}s. Average time: {total/iterations}s")
    # Number of workers = 4
    # Test: Total time to consume 4492 batches: 20s. Average time: 0.0044110886029441335s
    # Validation: Total time to consume 9008 batches: 25s. Average time: 0.002754582102091232s
    # Train: Total time to consume 76312 batches: 126s. Average time: 0.001647827838099294s

    # Number of workers = 6
    # Train: Total time to consume 76308 batches: 127s. Average time: 0.0016587930929925123s
    # Validation: Total time to consume 9006 batches: 30s. Average time: 0.003365322401490234s
    # Test: Total time to consume 4494 batches: 24s. Average time: 0.005404491927499606s
    
    # Number of workers = 2
    # Test: Total time to consume 4494 batches: 15s. Average time: 0.003398890170618116s
    # Validation: Total time to consume 9008 batches: 24s. Average time: 0.00269696923703961s
    # Train: Total time to consume 76312 batches: 160s. Average time: 0.002096151465801763s

    # Number of workers for validation and testing: 2

    # Further tests needed to decide number of workers for training data
    
    # Number of workers = 3
    # Train: Total time to consume 76311 batches: 134s. Average time: 0.0017562201159602544s

    # Number of workers = 5
    # Train: Total time to consume 76310 batches: 125s. Average time: 0.0016363617883661506s

    # Final
    # Train: 5 workers
    # Validation and Testing: 2 workers