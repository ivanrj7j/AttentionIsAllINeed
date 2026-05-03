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



if __name__ == "__main__":
    import time

    srcFile = '../../dataset/engTest.txt'
    tgtFile = '../../dataset/malTest.txt'

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