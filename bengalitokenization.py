# ...existing code...
import torch
from tokenizers import Tokenizer

class BengaliTokenizer:
    def __init__(self, vocab_file_path):
        self.tokenizer = Tokenizer.from_file(vocab_file_path)
        self.check_vocab()

    def check_vocab(self):
        voc = self.tokenizer.get_vocab()
        assert "<bn>" in voc, "Tokenizer must include <bn> token"
        assert "[START]" in voc and "[STOP]" in voc

    def encode(self, txt: str, language_id: str = None):
        # Normalization only; no manual <bn>, no [SPACE] replacement
        if language_id == 'bn':
            txt = self.bengali_normalize(txt)
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().tolist()
        return self.tokenizer.decode(seq, skip_special_tokens=False)

    def text_to_tokens(self, text: str, language_id: str = None):
        ids = self.encode(text, language_id)
        return torch.IntTensor(ids).unsqueeze(0)

    def bengali_normalize(self, text: str) -> str:
        import unicodedata
        return unicodedata.normalize('NFKC', text)
# ...existing code...

# # ...existing code...
# tokenizer_bn = BengaliTokenizer("tokenizer_bn_tts.json")
# sample_text = "পরিবেশ দূষণের মূল কারণগুলি হলো"
# ids = tokenizer_bn.encode(sample_text, language_id='bn')
# print(ids)
# print(tokenizer_bn.decode(ids))  # should show "<bn> [START] ... [STOP]" with correct Bengali, no mojibake
# # ...existing code...