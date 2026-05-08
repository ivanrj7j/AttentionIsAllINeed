import torch
from transformer import Transformer
from transformers import PreTrainedTokenizerFast


class Translator:
    def __init__(self, transformer: Transformer|torch.nn.DataParallel[Transformer], srcTokenizer: PreTrainedTokenizerFast, tgtTokenizer: PreTrainedTokenizerFast, device: str | torch.device, maxSentenceSize: int) -> None:
        self.transformer = transformer
        self.srcTokenizer = srcTokenizer
        self.tgtTokenizer = tgtTokenizer
        self.device = device
        self.maxSentenceSize = maxSentenceSize        

        self.bosIdx = self.tgtTokenizer.bos_token_id if self.tgtTokenizer.bos_token_id is not None else self.tgtTokenizer.cls_token_id
        self.eosIdx = self.tgtTokenizer.eos_token_id if self.tgtTokenizer.eos_token_id is not None else self.tgtTokenizer.sep_token_id

    @torch.no_grad()
    def translate(self, inp: str):
        transformer = self.transformer
        if isinstance(self.transformer, torch.nn.DataParallel):
            transformer = self.transformer.module

        assert isinstance(transformer, Transformer)
            
        transformer.eval()

        inputTokens = self.srcTokenizer(
            inp,
            max_length=self.maxSentenceSize,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )['input_ids'].to(self.device)
        # Tokenize Input
        
        srcMask = (inputTokens != 0).unsqueeze(1).unsqueeze(2).to(self.device)
        # Source Mask

        srcPositional = transformer.dropout(
            transformer.positionalEncoder(transformer.encoderEmbedding(inputTokens))
        ) * transformer.embeddingSizeSQRT
        # Source Positional Encoding & Embedding scaling exactly as implemented
        
        encOutput = srcPositional
        for encoder in transformer.encoders:
            encOutput = encoder(encOutput, srcMask)
        # Pass through encoders

        
        # AUTO-REGRESSIVE DECODING
        
        # Start target sequence with BOS token
        ys = torch.tensor([[self.bosIdx]], dtype=torch.long, device=self.device)

        for _ in range(self.maxSentenceSize - 1):
            sequenceLength = ys.size(1)
            noPeakMask = (1 - torch.triu(torch.ones((1, sequenceLength, sequenceLength)), diagonal=1)).bool().to(self.device)
            tgtMask = (ys != 0).unsqueeze(1).unsqueeze(2).to(self.device) & noPeakMask
            # Target Masking

            tgtPositional = transformer.positionalEncoder(transformer.decoderEmbedding(ys)) * transformer.embeddingSizeSQRT
            # Target Positional Encoding & Embedding
            
            decOutput = tgtPositional
            for decoder in transformer.decoders:
                decOutput = decoder(decOutput, encOutput, srcMask, tgtMask)
            # Pass through decoders

            out = transformer.linear(decOutput)
            
            next_word_logits = out[:, -1, :]
            next_word = next_word_logits.argmax(dim=-1).item()
            # Extracting the last predicted token

            ys = torch.cat([ys, torch.tensor([[next_word]], device=self.device)], dim=1)
            # Append the token to our growing sequence

            # Break if the model predicts the End-Of-Sentence token
            if next_word == self.eosIdx:
                break

        # 4. Decode the final tensor back into a human-readable string
        output_tokens = ys.squeeze(0).tolist()
        
        # Edge case: If the sequence is length 1, squeeze makes it an int instead of a list
        if isinstance(output_tokens, int): 
            output_tokens = [output_tokens]
            
        translated_text = self.tgtTokenizer.decode(output_tokens, skip_special_tokens=True)
        
        return translated_text

    
if __name__ == "__main__":
    import config

    print("Loading translator...")
    transformer = Transformer(config.NUM_LAYERS, config.SRC_VOCAB_SIZE, config.TGT_VOCAB_SIZE, config.EMBEDDING_SIZE, config.HEAD_COUNT, config.DFF, config.DROPOUT, config.MAX_SEQ_LEN, config.TARGET_DEVICE)
    transformer = torch.nn.DataParallel(transformer)
    transformer.load_state_dict(torch.load('../model/translator-model.pt'))

    print("Loading tokenizers...")
    engEmbedding, malEmbedding = transformer.module.getEmbeddingLayers()
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
    
    translator = Translator(transformer, engTokenizer, malTokenizer, config.TARGET_DEVICE, config.MAX_SENTENCE_SIZE)
    output = translator.translate("I am with you always.")
    print(output)