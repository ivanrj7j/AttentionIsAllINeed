from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset
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
    
    def processBatch(self, src:list[str], tgt:list[str]):
        assert len(src) == len(tgt)

        srcTokens = self.tokenizeString(src, self.srcTokenizer)
        tgtTokens = self.tokenizeString(tgt, self.tgtTokenizer)

        for a, b in zip(srcTokens, tgtTokens):
            yield (torch.tensor(a), torch.tensor(b))
    
    def __iter__(self) -> Iterator:
        return zip('a', 'b')



if __name__ == "__main__":
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


    testSrc = ['Hello world!', "I love you!"]
    testTgt = ['ഹലോ വേൾഡ്!', 'ഞാൻ നിന്നെ സ്നേഹിക്കുന്നു']

    for x, y in dataset.processBatch(testSrc, testTgt):
        print("x:", x)
        print("y:", y)