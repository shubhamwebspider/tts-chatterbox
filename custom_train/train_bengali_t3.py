# train_bengali_t3.py
import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from custom_train.custom_llama_configs import T3Config
# Use project T3 implementation
from src.chatterbox.models.t3.t3 import T3
from custom_train.custom_modules.custom_cond_enc import T3Cond
from bengalitokenization import BengaliTokenizer
from src.chatterbox.models.s3tokenizer.s3tokenizer import S3Tokenizer, S3_SR
from src.chatterbox.models.s3gen.s3gen import S3Token2Mel
from src.chatterbox.models.s3gen.utils.mel import mel_spectrogram
from src.chatterbox.models.s3gen.utils.mask import make_pad_mask
from src.chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
from datasets import load_dataset
import librosa
import numpy as np
import torch.nn.functional as F

# --- SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HYPERPARAMETERS ---
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
EPOCHS = 10
ACCUM_STEPS = 4
SAVE_EVERY = 1000
CHECKPOINT_DIR = "./checkpoints_bengali"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TOKENIZER_PATH = "tokenizer_bn_tts.json"
LANGUAGE_ID_FOR_BENGALI = 1
ACCENT_VOCAB = {"standard": 0, "sylheti": 1, "chittagonian": 2}
DEFAULT_ACCENT_ID = 0

# 24k SR for CosyVoice mel extractor
S3GEN_SR = 24000


def audio_to_tensor_pair(audio):
    wav = audio["array"]
    sr = audio["sampling_rate"]
    if sr != S3_SR:
        wav16 = librosa.resample(wav, orig_sr=sr, target_sr=S3_SR)
    else:
        wav16 = wav
    if sr != S3GEN_SR:
        wav24 = librosa.resample(wav, orig_sr=sr, target_sr=S3GEN_SR)
    else:
        wav24 = wav
    return torch.from_numpy(wav16).float(), torch.from_numpy(wav24).float()


class BengaliTTSDataset(Dataset):
    def __init__(self, hf_ds, hp: T3Config, text_tok: BengaliTokenizer, s3_tok: S3Tokenizer, ve: VoiceEncoder):
        self.ds = hf_ds
        self.hp = hp
        self.text_tok = text_tok
        self.s3_tok = s3_tok
        self.voice_encoder = ve.eval()
        for p in self.voice_encoder.parameters():
            p.requires_grad_(False)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds.iloc[idx]
        text: str = ex["transcriptions"]
        audio = ex["audio"]

        # Text tokens
        text_ids = self.text_tok.text_to_tokens(text, language_id="bn").squeeze(0).long()
        text_len = torch.tensor(int(len(text_ids)))

        # Audio at 16k and 24k
        wav16, wav24 = audio_to_tensor_pair(audio)
        wav16 = wav16[None, :]
        wav24 = wav24[None, :]

        # S3 speech tokens (raw)
        with torch.no_grad():
            s3_tokens_raw, s3_lens_raw = self.s3_tok(wavs=wav16)
        s3_tokens_raw = s3_tokens_raw[0].long()

        # 24k mel for S3Gen loss
        mel24 = mel_spectrogram(wav24, sampling_rate=S3GEN_SR).squeeze(0).transpose(0, 1)  # (T_mel, 80)

        # Align lengths: expect len(mel) ≈ 2 * len(tokens)
        desired_Tm = min(mel24.shape[0], 2 * s3_tokens_raw.shape[0])
        mel24 = mel24[:desired_Tm]
        s3_tokens_raw = s3_tokens_raw[: desired_Tm // 2]
        mel_len24 = torch.tensor(mel24.shape[0], dtype=torch.long)

        # BOS/EOS for T3 supervision
        s3_tokens_t3 = torch.cat([
            torch.tensor([self.hp.start_speech_token], dtype=torch.long),
            s3_tokens_raw,
            torch.tensor([self.hp.stop_speech_token], dtype=torch.long),
        ], dim=0)
        speech_len_t3 = torch.tensor(int(len(s3_tokens_t3)))

        # Speaker embedding from 16k
        with torch.no_grad():
            spk_emb = self.voice_encoder.embeds_from_wavs([wav16.squeeze(0).numpy()], sample_rate=S3_SR, as_spk=True)
            spk_emb = torch.from_numpy(spk_emb).float()

        lang_id = torch.tensor(LANGUAGE_ID_FOR_BENGALI, dtype=torch.long)
        if "accent" in ex and isinstance(ex["accent"], str):
            accent_id = torch.tensor(ACCENT_VOCAB.get(ex["accent"].lower(), DEFAULT_ACCENT_ID), dtype=torch.long)
        else:
            accent_id = torch.tensor(DEFAULT_ACCENT_ID, dtype=torch.long)

        return dict(
            text_tokens=text_ids,
            text_token_lens=text_len,
            speech_tokens_t3=s3_tokens_t3,
            speech_token_lens_t3=speech_len_t3,
            speech_tokens_raw=s3_tokens_raw,
            speech_token_lens_raw=torch.tensor(len(s3_tokens_raw), dtype=torch.long),
            mel24=mel24,
            mel_len24=mel_len24,
            speaker_emb=spk_emb,
            language_id=lang_id,
            accent_id=accent_id,
        )


def pad_1d(seqs, pad_value=0):
    lens = [len(s) for s in seqs]
    max_len = max(lens)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out, torch.tensor(lens, dtype=torch.long)


def pad_2d_time_major(seqs, pad_value=0.0):
    lens = [s.shape[0] for s in seqs]
    max_len = max(lens)
    feat_dim = seqs[0].shape[1]
    out = torch.full((len(seqs), max_len, feat_dim), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0], :] = s
    return out, torch.tensor(lens, dtype=torch.long)


def collate_fn(batch):
    text_padded, text_lens = pad_1d([b["text_tokens"] for b in batch], pad_value=0)
    speech_padded_t3, speech_lens_t3 = pad_1d([b["speech_tokens_t3"] for b in batch], pad_value=0)
    speech_raw_padded, speech_raw_lens = pad_1d([b["speech_tokens_raw"] for b in batch], pad_value=0)
    mel_padded, mel_lens = pad_2d_time_major([b["mel24"] for b in batch], pad_value=0.0)
    speaker_emb = torch.stack([b["speaker_emb"] for b in batch], dim=0).float()
    language_id = torch.stack([b["language_id"] for b in batch], dim=0).long()
    accent_id = torch.stack([b["accent_id"] for b in batch], dim=0).long()
    return dict(
        text_tokens=text_padded,
        text_token_lens=text_lens,
        speech_tokens_t3=speech_padded_t3,
        speech_token_lens_t3=speech_lens_t3,
        speech_tokens_raw=speech_raw_padded,
        speech_token_lens_raw=speech_raw_lens,
        mel24=mel_padded,
        mel_len24=mel_lens,
        speaker_emb=speaker_emb,
        language_id=language_id,
        accent_id=accent_id,
    )


def main():
    hp = T3Config.bengali_accent(TOKENIZER_PATH)
    text_tok = BengaliTokenizer(TOKENIZER_PATH)
    s3_tok = S3Tokenizer().to(device)
    ve = VoiceEncoder().to(device)

    # Use the user's dataset (non-streaming for random access)
    ds = load_dataset("ucalyptus/shrutilipi_bengali", split="train")
    train_ds = BengaliTTSDataset(ds, hp=hp, text_tok=text_tok, s3_tok=s3_tok, ve=ve)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Models
    t3_model = T3(hp=hp).to(device)
    s3_model = S3Token2Mel().to(device)

    optimizer = AdamW(list(t3_model.parameters()) + list(s3_model.parameters()), lr=LEARNING_RATE)
    scaler = GradScaler()

    global_step = 0
    t3_model.train(); s3_model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress_bar:
            text_tokens = batch["text_tokens"].to(device)
            text_token_lens = batch["text_token_lens"].to(device)
            speech_tokens_t3 = batch["speech_tokens_t3"].to(device)
            speech_token_lens_t3 = batch["speech_token_lens_t3"].to(device)
            speech_tokens_raw = batch["speech_tokens_raw"].to(device)
            speech_token_lens_raw = batch["speech_token_lens_raw"].to(device)
            mel24 = batch["mel24"].to(device)                  # (B, T, 80)
            mel_len24 = batch["mel_len24"].to(device)          # (B,)
            spk = batch["speaker_emb"].to(device)

            t3_cond = T3Cond(
                speaker_emb=spk,
                language_id=batch["language_id"].to(device),
                accent_id=batch["accent_id"].to(device),
            )

            with autocast():
                # T3 loss: text->speech tokens
                loss_text, loss_speech = t3_model.loss(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    text_token_lens=text_token_lens,
                    speech_tokens=speech_tokens_t3,
                    speech_token_lens=speech_token_lens_t3,
                )

                # S3 loss: tokens->mel using CFM decoder
                # 1) Project speaker emb to 80-d
                spk_norm = F.normalize(spk, dim=1)
                spk_proj = s3_model.flow.spk_embed_affine_layer(spk_norm)

                # 2) Token embeddings and encoder forward to get mu
                mask_tok = (~make_pad_mask(speech_token_lens_raw)).float().unsqueeze(-1).to(device)
                tok_emb = s3_model.flow.input_embedding(torch.clamp(speech_tokens_raw, min=0)) * mask_tok
                h, _ = s3_model.flow.encoder(tok_emb, speech_token_lens_raw)
                h = s3_model.flow.encoder_proj(h)             # (B, Tm, 80)

                # 3) Align lengths between h and mel
                Tm = torch.min(torch.tensor(h.shape[1], device=device), mel_len24.min())
                Tm = Tm.item()
                h = h[:, :Tm, :].transpose(1, 2).contiguous()         # (B, 80, T)
                feat = mel24[:, :Tm, :].transpose(1, 2).contiguous()  # (B, 80, T)
                mask_mel = (~make_pad_mask(torch.full_like(mel_len24, Tm))).unsqueeze(1).float().to(device)

                # 4) No prompt conditions during training
                cond = torch.zeros_like(feat)

                loss_s3, _y = s3_model.flow.decoder.compute_loss(
                    x1=feat,
                    mask=mask_mel,
                    mu=h,
                    spks=spk_proj,
                    cond=cond,
                )

                total_loss = loss_text + loss_speech + loss_s3

            scaler.scale(total_loss).backward()
            if (global_step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += total_loss.item()
            global_step += 1
            progress_bar.set_postfix({
                "L_t3_txt": f"{loss_text.item():.4f}",
                "L_t3_sp": f"{loss_speech.item():.4f}",
                "L_s3": f"{loss_s3.item():.4f}",
                "L": f"{total_loss.item():.4f}",
            })

            if global_step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"t3_s3_bn_step{global_step}.pt")
                torch.save({
                    "t3_model": t3_model.state_dict(),
                    "s3_model": s3_model.state_dict(),
                    "hp": hp.__dict__,
                }, ckpt_path)

        logger.info(f"Epoch {epoch+1} avg loss: {epoch_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    main()