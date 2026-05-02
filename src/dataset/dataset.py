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
        
    # def processBatch
    
    def __iter__(self) -> Iterator:
        return zip('a', 'b')



if __name__ == "__main__":
    srcFile = '../../dataset/engTest.txt'
    tgtFile = '../../dataset/malTest.txt'

    srcTokenizer = PreTrainedTokenizerFast(tokenizer_file='../../dataset/engTokenizer.json')
    tgtTokenizer = PreTrainedTokenizerFast(tokenizer_file='../../dataset/malTokenizer.json')

    srcTokenizer.post

    dataset = TransformerDataset(srcFile, tgtFile, 64 + 2, 1000, srcTokenizer, tgtTokenizer)


    with open(srcFile, 'r', encoding='utf-8') as sF, \
        open(tgtFile, 'r', encoding='utf-8') as tF:
        pass