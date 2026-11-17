"""Model B: Audio Spectrogram Transformer, fine-tuned on FMA genres (reduced setup for a 6 GB GPU).

Input: a batch of 16 kHz waveform crops [B, samples] (datasets.WaveDataset, data.kind: wave).
Features are computed inside forward() exactly like transformers' ASTFeatureExtractor
(Kaldi fbank, 128 mel bins, hanning window, padded/truncated to 1024 frames, then
(x - mean) / (2 * std) with AudioSet statistics), but on the GPU and always in float32.

Config (cfg["model"]): checkpoint, gradient_checkpointing, freeze_layers (embeddings + the first N
transformer layers are frozen to save time and memory), lr_backbone, lr_head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import register

MEAN, STD = -4.2677393, 4.5689974   # ASTFeatureExtractor defaults (AudioSet)
SAMPLE_RATE, N_MELS, MAX_FRAMES = 16_000, 128, 1024


class ASTGenre(nn.Module):
    def __init__(self, n_classes: int, checkpoint: str, gradient_checkpointing: bool = True,
                 freeze_layers: int = 0, lr_backbone: float = 1e-5, lr_head: float = 1e-3):
        super().__init__()
        from transformers import ASTForAudioClassification

        self.model = ASTForAudioClassification.from_pretrained(
            checkpoint, num_labels=n_classes, ignore_mismatched_sizes=True)
        self.lr_backbone, self.lr_head = lr_backbone, lr_head
        backbone = self.model.audio_spectrogram_transformer
        if freeze_layers > 0:
            for p in backbone.embeddings.parameters():
                p.requires_grad = False
            for layer in backbone.encoder.layer[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False
        if gradient_checkpointing:
            # non-reentrant: with frozen early layers the reentrant variant would give the trainable
            # layers no gradients (their inputs would not require grad).
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ------------------------------------------------------------------ features
    @staticmethod
    def features(wave: torch.Tensor) -> torch.Tensor:
        """[B, samples] float waveform -> [B, 1024, 128] normalized log-mel filterbanks."""
        import torchaudio.compliance.kaldi as ta_kaldi

        with torch.autocast(device_type=wave.device.type, enabled=False):
            out = []
            for w in wave.float():
                fb = ta_kaldi.fbank(w.unsqueeze(0), sample_frequency=SAMPLE_RATE,
                                    window_type="hanning", num_mel_bins=N_MELS)
                n = fb.shape[0]
                if n < MAX_FRAMES:
                    fb = nn.functional.pad(fb, (0, 0, 0, MAX_FRAMES - n))
                out.append(fb[:MAX_FRAMES])
            x = torch.stack(out)
        return (x - MEAN) / (STD * 2)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return self.model(input_values=self.features(wave)).logits

    def embed(self, wave: torch.Tensor) -> torch.Tensor:
        """Pooled 768-d representation (for UMAP plots)."""
        return self.model.audio_spectrogram_transformer(input_values=self.features(wave)).pooler_output

    # --------------------------------------------------------------- optimizer
    def param_groups(self, cfg: dict) -> list[dict]:
        head = [p for n, p in self.model.named_parameters() if n.startswith("classifier") and p.requires_grad]
        backbone = [p for n, p in self.model.named_parameters()
                    if not n.startswith("classifier") and p.requires_grad]
        return [{"params": backbone, "lr": self.lr_backbone}, {"params": head, "lr": self.lr_head}]


@register("ast")
def build(cfg: dict, n_classes: int) -> nn.Module:
    m = cfg["model"]
    return ASTGenre(n_classes, m.get("checkpoint", "MIT/ast-finetuned-audioset-10-10-0.4593"),
                    m.get("gradient_checkpointing", True), m.get("freeze_layers", 0),
                    float(m.get("lr_backbone", 1e-5)), float(m.get("lr_head", 1e-3)))
