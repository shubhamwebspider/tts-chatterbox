# modules/t3_config.py
from ..custom_llama_configs import LLAMA_CONFIGS


class T3Config:
    start_text_token = 255
    stop_text_token = 0
    text_tokens_dict_size = 704    # adjust if Bengali tokenizer is larger
    max_text_tokens = 2048

    start_speech_token = 6561
    stop_speech_token = 6562
    speech_tokens_dict_size = 8194
    max_speech_tokens = 4096

    llama_config_name = "Llama_520M"
    input_pos_emb = "learned"
    speech_cond_prompt_len = 150

    # Conditioning
    encoder_type = "voice_encoder"
    speaker_embed_size = 256
    use_perceiver_resampler = True
    emotion_adv = True

    # 👇 MULTILINGUAL + ACCENT CONFIG
    is_multilingual = True
    use_language_id = True
    use_accent_id = True
    num_languages = 2   # 0=English, 1=Bengali (expand as needed)
    num_accents = 3     # 0=Standard, 1=Sylheti, 2=Chittagonian

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]

    @classmethod
    def bengali_accent(cls):
        """Factory for Bengali training config."""
        cfg = cls()
        cfg.is_multilingual = True
        cfg.use_language_id = True
        cfg.use_accent_id = True
        cfg.num_languages = 2
        cfg.num_accents = 3
        return cfg