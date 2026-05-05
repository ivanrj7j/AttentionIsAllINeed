from transformer import Transformer
from embedding import EmbeddingLayer
from transformers import PreTrainedTokenizerFast
import torch

import config

print("Loading transformer...")
transformer = Transformer(config.NUM_LAYERS, config.SRC_VOCAB_SIZE, config.TGT_VOCAB_SIZE, config.EMBEDDING_SIZE, config.HEAD_COUNT, config.DFF, config.DROPOUT, config.MAX_SENTENCE_SIZE, 'cpu')
transformer.load_state_dict(torch.load('../model/translator-model.pt'))

print("Loading tokenizers...")
engEmbedding, malEmbedding = transformer.getEmbeddingLayers()
engTokenizer = PreTrainedTokenizerFast(tokenizer_file='../dataset/engTokenizer.json',  
                                 bos_token="<bos>", 
                                 eos_token="<eos>",
                                 pad_token="<pad>",
                                 unk_token="<unk>")
malTokenizer = PreTrainedTokenizerFast(tokenizer_file='../dataset/malTokenizer.json',  
                                 bos_token="<bos>", 
                                 eos_token="<eos>",
                                 pad_token="<pad>",
                                 unk_token="<unk>")

print("Creating embedding layers...")
english = EmbeddingLayer(engEmbedding, engTokenizer, config.TARGET_DEVICE, config.MAX_SENTENCE_SIZE)
malayalam = EmbeddingLayer(malEmbedding, malTokenizer, config.TARGET_DEVICE, config.MAX_SENTENCE_SIZE)

print("Calculating embedding...")
print(english(['Hello world!', 'How are you?'], 'FIRST').size())
print(english(['Hello world!', 'How are you?']).size())
print(malayalam(['കല്ലിൽ കണ്ട അതേ രൂപം.', 'എല്ലാവരും അവളെ സംസാരിക്കുന്നത്.'], 'FIRST').size())
print(malayalam(['കല്ലിൽ കണ്ട അതേ രൂപം.', 'എല്ലാവരും അവളെ സംസാരിക്കുന്നത്.']).size())

# torch.save(english.state_dict(), '../model/englishEmbedding.pt')
# torch.save(malayalam.state_dict(), '../model/malayalamEmbedding.pt')