import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torchaudio.transforms import Spectrogram, GriffinLim, MelSpectrogram

############################################################
# STFT Encoder/Decoder
############################################################

class STFTEncoder(nn.Module):
    def __init__(self, n_fft=512, hop_length=128):
        super().__init__()
        self.stft = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)

    def forward(self, x):
        # Returns complex spectrogram
        return self.stft(x)


class STFTDecoder(nn.Module):
    def __init__(self, n_fft=512, hop_length=128):
        super().__init__()
        self.griffin = GriffinLim(n_fft=n_fft, hop_length=hop_length)

    def forward(self, mag):
        return self.griffin(mag)


############################################################
# Conformer Block (minimal ESPnet-style version)
############################################################

class ConformerBlock(nn.Module):
    """Minimal conformer layer (sufficient for inference)."""
    def __init__(self, dim, heads, ff_mult=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.Swish(),
            nn.Linear(dim * ff_mult, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)
        return x


############################################################
# ConVoiFilter Model
############################################################

class ConVoiFilter(nn.Module):
    def __init__(self, config):
        super().__init__()

        sep_conf = config["enh_args"]["separator_conf"]
        self.dim = sep_conf["adim"]
        self.layers = sep_conf["layers"]
        self.heads = sep_conf["aheads"]

        # Encoder/decoder
        self.stft_enc = STFTEncoder(
            n_fft=config["enh_args"]["encoder_conf"]["n_fft"],
            hop_length=config["enh_args"]["encoder_conf"]["hop_length"]
        )
        self.stft_dec = STFTDecoder(
            n_fft=config["enh_args"]["decoder_conf"]["n_fft"],
            hop_length=config["enh_args"]["decoder_conf"]["hop_length"]
        )

        # Input linear projection
        self.input_linear = nn.Linear(257, self.dim)

        # Conformer stack
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(self.dim, self.heads) for _ in range(self.layers)
        ])

        # Output mask head
        self.mask_head = nn.Sequential(
            nn.Linear(self.dim, 257),
            nn.ReLU()
        )

    def forward(self, wav, spk_emb):
        """
        wav: (1, T)
        spk_emb: (1, 512)  NOTE: placeholder x-vector
        """
        stft = self.stft_enc(wav)  # (1, freq, time, complex-realpart)
        mag = torch.sqrt(stft.pow(2).sum(-1))  # magnitude (1, F, T)
        mag = mag.squeeze(0).transpose(0, 1)   # (T, F)

        # Linear projection
        x = self.input_linear(mag)  # (T, D)

        # Add speaker embedding (broadcast)
        x = x + spk_emb  # (T, D)

        # Conformer stack
        for layer in self.conformer_layers:
            x = layer(x)

        # Predict mask
        mask = self.mask_head(x)  # (T, F)
        mask = mask.transpose(0, 1).unsqueeze(0)

        enhanced_mag = mag.transpose(0, 1).unsqueeze(0) * mask
        enhanced = self.stft_dec(enhanced_mag.squeeze(0))

        return enhanced


############################################################
# Utility Functions
############################################################

def load_audio(path, sr=16000):
    wav, fs = torchaudio.load(path)
    if fs != sr:
        wav = torchaudio.functional.resample(wav, fs, sr)
    return wav


def fake_xvector():
    """Placeholder 512-d speaker embedding (no x-vector model provided)."""
    return torch.randn(1, 1024) * 0.01  # match adim


############################################################
# Main Inference
############################################################

def main():
    # Load config
    with open("config.json") as f:
        config = json.load(f)

    # Load model
    model = ConVoiFilter(config)
    sd = torch.load("pytorch_model.bin", map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    # Load audio
    noisy = load_audio("noisy.wav").unsqueeze(0)  # (1, T)
    ref = load_audio("ref.wav")  # unused directly

    # Use placeholder speaker embedding (x-vector not provided)
    spk_emb = fake_xvector()

    # Run model
    with torch.no_grad():
        enhanced = model(noisy, spk_emb)

    # Save result
    torchaudio.save("enhanced.wav", enhanced.cpu(), 16000)
    print("Saved enhanced.wav")


if __name__ == "__main__":
    main()
