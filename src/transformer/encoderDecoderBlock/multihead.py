import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    ### Multi Head Attention mechanism

    #### Explanation

    First part of the multihead attention is the query, key and values to understand this let's take a dictionary as an analogy. The dictionary has a key and a value the key can be the word and the value is the meaning, transformer architecture borrows from this idea.

    - Query: Each word has one query, which is compared with all keys
    - Key: The similarity between query and key is used to find the words that we are going to understand the meaning of
    - Value: The value gives the meaning of the word in relation to the query

    Note that each word(token) has a corresponding query key and a value vector

    ##### Multiple heads in the architecture

    Let's say our word embedding size is 100 element and we have 10 heads for each output from the query, key and the value we will split the query key and the value generted by the model and give it to each head, this enables parallel computation possible.

    ##### Attention calculation 

    For each head we calculate the attention of of each tokens in parallel. We find the dot product of the key and the value then we find the dot product of the result of the previous operation and find the dot product with value this is attention score. (Note that before the second dot product we divide the values with sqrt(embedding size) and find softmax)

    After we find out the attention scores we combine them together into one single unit from all the heads and the output is fed to the output combiner. the output from this will be fed to next layers.
    """
    def __init__(self, embeddingSize: int, headCount: int) -> None:
        super().__init__()
        assert embeddingSize % headCount == 0, (
            "The number of elements in the embedding should match the number"
        )

        self.dModel = embeddingSize
        self.headCount = headCount
        self.itemsPerHead = self.dModel // self.headCount
        self.itemsPerHeadSRQT = self.itemsPerHead ** 0.5 #to avoid recalculation, not necessary

        self.queryWeights = nn.Linear(self.dModel, self.dModel)
        self.keyWeights = nn.Linear(self.dModel, self.dModel)
        self.valueWeights = nn.Linear(self.dModel, self.dModel)
        self.outputCombinerWeights = nn.Linear(self.dModel, self.dModel)

    def splitHeads(self, x:torch.Tensor):
        batchSize, numberOfTokens, embeddingSize = x.size()
        return x.view(batchSize, numberOfTokens, self.headCount, self.itemsPerHead).transpose(1, 2)
    
    def combineHeads(self, x:torch.Tensor):
        batchSize, _, numberOfTokens, embeddingSize = x.size()
        return x.transpose(1, 2).contiguous().view(batchSize, numberOfTokens, self.dModel)

    # The code is commented out because it is self attention implementation and left out for understanding what is happening, but the actual implementation will be assumming cross attention
    # def calculateQKV(self, x:torch.Tensor):
    #     q = self.queryWeights(x)
    #     k = self.keyWeights(x)
    #     v = self.valueWeights(x)

    #     return (self.splitHeads(q), self.splitHeads(k), self.splitHeads(v))

    # def calculateAttention(self, x:torch.Tensor):
    #     q, k, v = self.calculateQKV(x)

    def calculateQKV(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor):
        q = self.queryWeights(Q)
        k = self.keyWeights(K)
        v = self.valueWeights(V)

        return self.splitHeads(q), self.splitHeads(k), self.splitHeads(v)
    
    def calculateAttention(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor|None=None):
        q, k, v = self.calculateQKV(Q, K, V)

        qk = torch.matmul(q, k.transpose(-2, -1)).div(self.itemsPerHeadSRQT)

        if mask is not None:
            qk = qk.masked_fill(mask == 0, -1e9)
            # Mask will be a matrix of 1's and zeros if we are provided with a mask, we will be changing the parts with 0 in the mask to -1e9

        attentionProbability = torch.softmax(qk, dim=-1)

        return torch.matmul(attentionProbability, v)
    
    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor|None=None):
        attention = self.calculateAttention(Q, K, V, mask)
        attention = self.combineHeads(attention)
        return self.outputCombinerWeights(attention) 
    


if __name__ == "__main__":
    mha = MultiHeadAttention(512, 8)
    inp = torch.rand((32, 128, 512)) # 32 batches of 128 words each of embedding size 512
    attentionScores = mha.calculateAttention(inp, inp, inp)
    op1 = mha.outputCombinerWeights(mha.combineHeads(attentionScores))
    op2 = mha(inp, inp, inp)
    
    assert op1.equal(op2)

    print(attentionScores, op2)

    print("Attentnion scores", attentionScores.size(), "Output", op2.size())