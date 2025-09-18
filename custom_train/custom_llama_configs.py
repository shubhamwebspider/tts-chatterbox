from tokenizers import Tokenizer
from src.chatterbox.models.t3.llama_configs import LLAMA_CONFIGS

class T3Config:
    # Defaults (will be overwritten by bengali_accent)
    start_text_token = 255
    stop_text_token = 0
    text_tokens_dict_size = 704
    max_text_tokens = 2048

    start_speech_token = 6561
    stop_speech_token = 6562
    speech_tokens_dict_size = 8194
    max_speech_tokens = 4096

    llama_config_name = "Llama_520M"
    input_pos_emb = "learned"
    speech_cond_prompt_len = 150
    encoder_type = "voice_encoder"
    speaker_embed_size = 256
    # Perceiver is empty in repo; keep False to avoid import errors
    use_perceiver_resampler = False
    emotion_adv = True

    # Multilingual + Accent
    is_multilingual = True
    use_language_id = True
    use_accent_id = True
    num_languages = 2   # 0=English, 1=Bengali
    num_accents = 3     # 0=Standard, 1=Sylheti, 2=Chittagonian

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]

    @classmethod
    def bengali_accent(cls, tokenizer_path: str):
        cfg = cls()
        tok = Tokenizer.from_file(tokenizer_path)
        voc_size = tok.get_vocab_size()
        start_id = tok.token_to_id("[START]")
        stop_id = tok.token_to_id("[STOP]")
        cfg.text_tokens_dict_size = voc_size
        cfg.start_text_token = int(start_id)
        cfg.stop_text_token = int(stop_id)
        cfg.is_multilingual = True
        cfg.use_language_id = True
        cfg.use_accent_id = True
        cfg.num_languages = 2
        cfg.num_accents = 3
        return cfg
