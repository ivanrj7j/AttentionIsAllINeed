from .decoder import Decoder
from .encoder import Encoder
from .positionalEncoder import PositionalEncoder

import torch
import torch.nn as nn
import math
class Transformer(nn.Module):
    def __init__(self, numLayers:int, srcVocabSize:int, tgtVocabSize:int, embeddingSize:int, headCount:int, dff:int, dropout:float, maxSequenceLength:int) -> None:
        super().__init__()
        self.encoderEmbedding = nn.Embedding(srcVocabSize, embeddingSize)
        self.decoderEmbedding = nn.Embedding(tgtVocabSize, embeddingSize)
        # embedding layers 

        self.positionalEncoder = PositionalEncoder(embeddingSize, maxSequenceLength)
        # positonal encoder 

        self.encoders = nn.ModuleList([Encoder(embeddingSize, headCount, dff, dropout) for _ in range(numLayers)])
        self.decoders = nn.ModuleList([Decoder(embeddingSize, headCount, dff, dropout) for _ in range(numLayers)])
        # encoder and decoder 

        self.linear = nn.Linear(embeddingSize, tgtVocabSize)
        self.dropout = nn.Dropout(dropout)
        # probability prediction layer 

        self.embeddingSizeSQRT = math.sqrt(embeddingSize)

    def generateMask(self, src:torch.Tensor, tgt:torch.Tensor):
        srcMask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgtMask = (tgt != 0).unsqueeze(1).unsqueeze(2)

        sequenceLength = tgt.size(1)
        noPeakMask = (1 - torch.triu(torch.ones((1, sequenceLength, sequenceLength)), diagonal=1)).bool()

        return srcMask, tgtMask & noPeakMask
    
    def forward(self, src:torch.Tensor, tgt:torch.Tensor):
        srcMask, tgtMask = self.generateMask(src, tgt)
        
        srcPositional = self.dropout(self.positionalEncoder(self.encoderEmbedding(src))) * self.embeddingSizeSQRT
        tgtPositional = self.dropout(self.positionalEncoder(self.decoderEmbedding(tgt))) * self.embeddingSizeSQRT

        encOutput = srcPositional
        for encoder in self.encoders:
            encOutput = encoder(encOutput, srcMask)
        # going through each encoder first 

        decOutput = tgtPositional
        for decoder in self.decoders:
            decOutput = decoder(decOutput, encOutput, srcMask, tgtMask)
        # going through each decoder 

        return self.linear(decOutput)