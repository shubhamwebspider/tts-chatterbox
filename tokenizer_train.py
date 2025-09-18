# ...existing code...
from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing

SPECIAL_TOKENS = ["[START]", "[STOP]", "[UNK]", "[PAD]", "[SEP]", "[CLS]", "[MASK]", "<bn>"]

tok = Tokenizer(models.Unigram())
tok.normalizer = normalizers.Sequence([normalizers.NFKC()])
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tok.decoder = decoders.ByteLevel()  # critical for correct UTF‑8 reconstruction

trainer = trainers.UnigramTrainer(
    vocab_size=16000,
    special_tokens=SPECIAL_TOKENS,
    unk_token="[UNK]",
    show_progress=True,
)
tok.train(files=["bengali.txt"], trainer=trainer)

# Prepend <bn> here (do NOT also prepend it in encode)
tok.post_processor = TemplateProcessing(
    single="<bn> [START] $A [STOP]",
    pair="<bn> [START] $A [SEP] $B [STOP]",
    special_tokens=[
        ("<bn>", tok.token_to_id("<bn>")),
        ("[START]", tok.token_to_id("[START]")),
        ("[STOP]", tok.token_to_id("[STOP]")),
        ("[SEP]", tok.token_to_id("[SEP]")),
    ],
)

tok.enable_padding(pad_id=tok.token_to_id("[PAD]"), pad_token="[PAD]")
tok.save("tokenizer_bn_tts.json")
print("✅ Bengali TTS tokenizer with <bn> saved.")
# ...existing code...