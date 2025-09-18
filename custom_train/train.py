import os
import json
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path

from torch.utils.data import Dataset
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from datasets import load_dataset

# =============================
# Custom chatterbox imports
# =============================
from src.chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.t3 import T3, T3Cond, T3Config
# from chatterbox.models.s3.constants import S3_SR
S3_SR = 16_000

# =============================
# Arguments
# =============================
@dataclass
class ModelArguments:
    model_dir: str = field(
        metadata={"help": "Path to local pretrained ChatterboxTTS model directory."}
    )

@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Hugging Face dataset name for Bengali speech."}
    )
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to local dataset containing wav + txt."}
    )
    train_split: str = field(default="train")
    eval_split: str = field(default="validation")
    audio_column: str = field(default="audio")
    text_column: str = field(default="text")
    max_text_len: int = field(default=400)
    max_speech_len: int = field(default=2500)
    prompt_duration_s: float = field(default=3.0)

# =============================
# Dataset class
# =============================
class BengaliTTSDataset(Dataset):
    def __init__(self, dataset, chatterbox_model: ChatterboxTTS, args: DataArguments):
        self.ds = dataset
        self.args = args
        self.s3_sr = S3_SR
        self.text_tokenizer = chatterbox_model.tokenizer
        self.speech_tokenizer = chatterbox_model.s3gen.tokenizer
        self.voice_encoder = chatterbox_model.ve
        self.prompt_len_samples = int(args.prompt_duration_s * self.s3_sr)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        # --- Load audio ---
        if isinstance(item[self.args.audio_column], str):
            wav, sr = librosa.load(item[self.args.audio_column], sr=None, mono=True)
        else:
            wav = item[self.args.audio_column]["array"]
            sr = item[self.args.audio_column]["sampling_rate"]
        if sr != self.s3_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.s3_sr)
        wav = wav.astype(np.float32)

        # --- Speaker embedding ---
        speaker_emb = self.voice_encoder.embeds_from_wavs([wav], sample_rate=self.s3_sr)
        speaker_emb = torch.from_numpy(speaker_emb[0])

        # --- Text tokens ---
        norm_text = punc_norm(item[self.args.text_column])
        text_tokens = self.text_tokenizer.text_to_tokens(norm_text).squeeze(0)
        text_tokens = F.pad(text_tokens, (1, 0), value=self.text_tokenizer.start_token_id)
        text_tokens = torch.cat([text_tokens, torch.tensor([self.text_tokenizer.stop_token_id])])
        text_tokens = text_tokens[: self.args.max_text_len]
        text_len = torch.tensor(len(text_tokens))

        # --- Speech tokens ---
        speech_tokens, lens = self.speech_tokenizer.forward([wav])
        speech_tokens = speech_tokens.squeeze(0)[: lens[0]]
        speech_tokens = F.pad(speech_tokens, (1, 0), value=self.speech_tokenizer.start_token_id)
        speech_tokens = torch.cat([speech_tokens, torch.tensor([self.speech_tokenizer.stop_token_id])])
        speech_tokens = speech_tokens[: self.args.max_speech_len]
        speech_len = torch.tensor(len(speech_tokens))

        # --- Prompt tokens ---
        cond_audio = wav[: self.prompt_len_samples]
        if len(cond_audio) == 0:
            prompt_tokens = torch.zeros(self.prompt_len_samples, dtype=torch.long)
        else:
            prompt_tokens, _ = self.speech_tokenizer.forward([cond_audio], max_len=self.prompt_len_samples)
            prompt_tokens = prompt_tokens.squeeze(0)

        return {
            "text_tokens": text_tokens,
            "text_len": text_len,
            "speech_tokens": speech_tokens,
            "speech_len": speech_len,
            "speaker_emb": speaker_emb,
            "prompt_tokens": prompt_tokens,
        }

# =============================
# Data collator
# =============================
@dataclass
class BengaliCollator:
    text_pad_id: int
    speech_pad_id: int
    prompt_len: int

    def __call__(self, batch: List[Dict]):
        # Pad text
        max_text = max([len(x["text_tokens"]) for x in batch])
        max_speech = max([len(x["speech_tokens"]) for x in batch])

        text_tokens = torch.stack([
            F.pad(x["text_tokens"], (0, max_text - len(x["text_tokens"])), value=self.text_pad_id)
            for x in batch
        ])
        text_lens = torch.stack([x["text_len"] for x in batch])

        speech_tokens = torch.stack([
            F.pad(x["speech_tokens"], (0, max_speech - len(x["speech_tokens"])), value=self.speech_pad_id)
            for x in batch
        ])
        speech_lens = torch.stack([x["speech_len"] for x in batch])

        speaker_embs = torch.stack([x["speaker_emb"] for x in batch])
        prompts = torch.stack([x["prompt_tokens"] for x in batch])

        # Labels: shift left, mask pads & prompt
        labels_speech = speech_tokens[:, 1:].clone()
        labels_speech[labels_speech == self.speech_pad_id] = -100
        labels_speech[:, : self.prompt_len] = -100

        labels_text = text_tokens[:, 1:].clone()
        labels_text[labels_text == self.text_pad_id] = -100

        return {
            "text_tokens": text_tokens,
            "text_lens": text_lens,
            "speech_tokens": speech_tokens,
            "speech_lens": speech_lens,
            "speaker_emb": speaker_embs,
            "prompt_tokens": prompts,
            "labels_text": labels_text,
            "labels_speech": labels_speech,
        }

# =============================
# Model wrapper
# =============================
class T3Wrapper(torch.nn.Module):
    def __init__(self, t3: T3, config: T3Config):
        super().__init__()
        self.t3 = t3
        self.config = config

    def forward(self, text_tokens, text_lens, speech_tokens, speech_lens,
                speaker_emb, prompt_tokens, labels_text=None, labels_speech=None):

        cond = T3Cond(
            speaker_emb=speaker_emb,
            cond_prompt_speech_tokens=prompt_tokens,
            emotion_adv=torch.tensor([0.5], device=self.t3.device)
        )

        loss_text, loss_speech, logits = self.t3.loss(
            cond,
            text_tokens=text_tokens,
            text_token_lens=text_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_lens,
            labels_text=labels_text,
            labels_speech=labels_speech,
        )
        return {"loss": loss_text + loss_speech, "logits": logits}

# =============================
# Main
# =============================
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # --- Load base model ---
    chatterbox_model = ChatterboxTTS.from_local(model_args.model_dir, device="cpu")
    t3_model = chatterbox_model.t3
    t3_model.train()

    # --- Load dataset ---
    if data_args.dataset_name:
        ds = load_dataset(data_args.dataset_name)
        train_ds = ds[data_args.train_split]
        eval_ds = ds[data_args.eval_split]
    else:
        raise ValueError("Provide a dataset_name for Bengali dataset.")

    train_dataset = BengaliTTSDataset(train_ds, chatterbox_model, data_args)
    eval_dataset = BengaliTTSDataset(eval_ds, chatterbox_model, data_args)

    collator = BengaliCollator(
        text_pad_id=chatterbox_model.tokenizer.pad_token_id,
        speech_pad_id=chatterbox_model.s3gen.tokenizer.pad_token_id,
        prompt_len=int(data_args.prompt_duration_s * S3_SR),
    )

    # --- Wrap model ---
    model = T3Wrapper(t3_model, chatterbox_model.t3_cfg)

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # --- Train ---
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()