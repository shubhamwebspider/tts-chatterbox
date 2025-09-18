import torch
from tokenizers import Tokenizer
import unicodedata

class UniversalTokenizer:
    def __init__(self, en_vocab_path, bn_vocab_path):
        self.tokenizers = {
            "en": Tokenizer.from_file(en_vocab_path),
            "bn": Tokenizer.from_file(bn_vocab_path)
        }
        self._check_vocab("en")
        self._check_vocab("bn")

    def _check_vocab(self, lang):
        voc = self.tokenizers[lang].get_vocab()
        if lang == "en":
            assert "[START]" in voc and "[STOP]" in voc
        elif lang == "bn":
            assert "<bn>" in voc and "[START]" in voc and "[STOP]" in voc

    def encode(self, text: str, language_id: str):
        if language_id == "en":
            # English: replace spaces with [SPACE]
            text = text.replace(' ', '[SPACE]')
            ids = self.tokenizers["en"].encode(text).ids
        elif language_id == "bn":
            # Bengali: normalize, no [SPACE] replacement
            text = unicodedata.normalize('NFKC', text)
            ids = self.tokenizers["bn"].encode(text).ids
        else:
            raise ValueError(f"Unsupported language_id: {language_id}")
        return ids

    def decode(self, seq, language_id: str):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().tolist()
        txt = self.tokenizers[language_id].decode(seq, skip_special_tokens=False)
        if language_id == "en":
            txt = txt.replace(' ', '').replace('[SPACE]', ' ').replace('[STOP]', '').replace('[UNK]', '')
        elif language_id == "bn":
            pass  # Bengali: ByteLevel decoder handles everything
        return txt

    def text_to_tokens(self, text: str, language_id: str):
        ids = self.encode(text, language_id)
        return torch.IntTensor(ids).unsqueeze(0)
    


tokenizer = UniversalTokenizer()
en_ids = tokenizer.encode("This is a test.", language_id="en")
bn_ids = tokenizer.encode("এইটি একটি বাংলা বাক্য।", language_id="bn")
print(tokenizer.decode(en_ids, language_id="en"))
print(tokenizer.decode(bn_ids, language_id="bn"))