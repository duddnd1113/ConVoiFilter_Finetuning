# ==============================================================================================
# Finetuning.py — ConVoiFilter FineTuning (Conformer + Only-Attn + Speaker FFN + Pre-FFN Option)
# ==============================================================================================

# ---------- Warning Suppression ----------
import warnings, logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="istft will require a complex-valued input tensor")
logging.getLogger().setLevel(logging.ERROR)

# ---------- Imports ----------
import os
import math
import glob
import csv
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from functools import partial

from src.model.modeling_enh import VoiceFilter
from src.model.configuration_voicefilter import VoiceFilterConfig


# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", device)


# ============================================================
# HF-compatible wav loader
# ============================================================
def load_wav(path, sr=16000):
    try:
        wav, sr_ = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr_ != sr:
            wav = librosa.resample(wav, orig_sr=sr_, target_sr=sr)
        return wav.astype(np.float32)
    except:
        wav, _ = librosa.load(path, sr=sr, mono=True)
        return wav.astype(np.float32)


# ============================================================
# Dataset
# ============================================================
class ConVoiFilterDataset(Dataset):
    def __init__(self, root_dir, sr=16000):
        self.sr = sr

        self.mix_files = sorted(glob.glob(os.path.join(root_dir, "Mix/*.wav")))
        self.clean_files = sorted(glob.glob(os.path.join(root_dir, "Clean/*.wav")))
        self.ref_files = sorted(glob.glob(os.path.join(root_dir, "Target/*.wav")))

        assert len(self.mix_files) == len(self.clean_files) == len(self.ref_files)

        print(f"[Dataset] Loaded {root_dir}: {len(self.mix_files)} samples")

    def __getitem__(self, idx):
        mix = load_wav(self.mix_files[idx], self.sr)
        clean = load_wav(self.clean_files[idx], self.sr)
        ref = load_wav(self.ref_files[idx], self.sr)

        T = min(len(mix), len(clean))
        mix = mix[:T]
        clean = clean[:T]

        return {"mix": torch.tensor(mix),
                "clean": torch.tensor(clean),
                "ref": torch.tensor(ref)}

    def __len__(self):
        return len(self.mix_files)


# ============================================================
# Collate fn
# ============================================================
def collate_fn(batch, chunk_size):
    mixes = [b["mix"] for b in batch]
    cleans = [b["clean"] for b in batch]
    refs = [b["ref"] for b in batch]
    lengths = torch.tensor([len(x) for x in mixes])

    max_len = lengths.max().item()
    pad_len = math.ceil(max_len / chunk_size) * chunk_size

    mix_batch, clean_batch = [], []

    for m, c in zip(mixes, cleans):
        pad = pad_len - len(m)
        if pad > 0:
            m = torch.cat([m, torch.zeros(pad)])
            c = torch.cat([c, torch.zeros(pad)])
        mix_batch.append(m)
        clean_batch.append(c)

    return {
        "mix": torch.stack(mix_batch).unsqueeze(1),
        "clean": torch.stack(clean_batch).unsqueeze(1),
        "lengths": lengths,
        "refs": refs,
    }


# ============================================================
# Speaker embedding calculator
# ============================================================
def calc_spk_emb(model, ref_list, sr, device):
    emb_list = []
    max_len = sr * 5

    for ref in ref_list:
        wav = ref.numpy().astype(np.float32)

        chunks = []
        for i in range(0, len(wav), max_len):
            w = wav[i:i+max_len]
            if len(w) < max_len:
                w = np.concatenate([w, np.zeros(max_len-len(w), np.float32)])
            chunks.append(w)

        chunks = torch.tensor(chunks).float().unsqueeze(1).to(device)

        with torch.no_grad():
            emb = model.xvector_model(chunks).mean(dim=0)

        emb_list.append(emb)

    return torch.stack(emb_list, dim=0)


# ============================================================
# Load pretrained + Finetune control
# ============================================================
def load_local_voicefilter(depth, type="full", open_spk_ffn=0, open_pre_ffn=0):

    config = VoiceFilterConfig.from_pretrained("pretrained/config.json")
    model = VoiceFilter(config)

    state = torch.load("pretrained/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state, strict=False)

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # ---- Conformer blocks ----
    encoders = model.enh_model.separator.conformer.encoders
    total_blocks = len(encoders)
    start_idx = total_blocks - depth

    print(f"[FT Mode] {type}")
    print(f"[FT Blocks] {list(range(start_idx, total_blocks))}")

    # ---------------------------
    # Conformer FT
    # ---------------------------
    if type == "full":
        for idx in range(start_idx, total_blocks):
            for p in encoders[idx].parameters():
                p.requires_grad = True

    elif type == "attn":
        for idx in range(start_idx, total_blocks):
            block = encoders[idx]
            # Self-attention only
            for p in block.self_attn.parameters():
                p.requires_grad = True

            # Freeze the rest
            for name, module in block.named_children():
                if name != "self_attn":
                    for p in module.parameters():
                        p.requires_grad = False

    # --------------------------------------------------------
    # (NEW) Pre-Conformer FFN / Conv 열기
    # --------------------------------------------------------
    if open_pre_ffn == 1:
        print("[PRE] Pre-Conformer FFN/Conv → Trainable")

        possible_pre = [
            "linear",
            "pre_encoder",
            "input_layer",
            "conv_in",
            "prenet",
            "preprocessor",
        ]

        for name in possible_pre:
            if hasattr(model.enh_model.separator, name):
                module = getattr(model.enh_model.separator, name)
                print(f"  → Open: separator.{name}")
                for p in module.parameters():
                    p.requires_grad = True
    else:
        print("[PRE] Pre-Conformer FFN/Conv → Frozen")

    # --------------------------------------------------------
    # Speaker Encoder FFN 열기
    # --------------------------------------------------------
    if open_spk_ffn == 1:
        print("[SPK] Speaker Encoder FFN → Trainable")
        for name, module in model.xvector_model.named_modules():
            if isinstance(module, nn.Linear):
                for p in module.parameters():
                    p.requires_grad = True
    else:
        print("[SPK] Speaker Encoder FFN → Frozen")

    # count trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Trainable Params] {trainable} / {total}")

    return model, config


# ============================================================
# Train step
# ============================================================
def train_step(model, batch, optimizer, sr):
    model.train()

    mix = batch["mix"].to(device)
    clean = batch["clean"].to(device)
    lengths = batch["lengths"].to(device)
    refs = batch["refs"]

    spk_emb = calc_spk_emb(model, refs, sr, device)

    optimizer.zero_grad()
    out = model(
        speech=mix.squeeze(1),
        speech_lengths=lengths,
        target_speech=clean.squeeze(1),
        target_spk_embedding=spk_emb,
    )

    loss = out.loss
    loss.backward()
    optimizer.step()

    return loss.item()


# ============================================================
# Eval step
# ============================================================
def eval_step(model, batch, sr):
    model.eval()

    mix = batch["mix"].to(device)
    clean = batch["clean"].to(device)
    lengths = batch["lengths"].to(device)
    refs = batch["refs"]

    spk_emb = calc_spk_emb(model, refs, sr, device)

    with torch.no_grad():
        out = model(
            speech=mix.squeeze(1),
            speech_lengths=lengths,
            target_speech=clean.squeeze(1),
            target_spk_embedding=spk_emb,
        )
    return out.loss.item()


# ============================================================
# Inference
# ============================================================
def run_test_inference(model, test_mix_path, test_ref_path, sr, save_path):
    mix = load_wav(test_mix_path, sr)
    ref = load_wav(test_ref_path, sr)

    mix_t = torch.tensor(mix).float().to(device)
    ref_t = torch.tensor(ref).float().unsqueeze(0).to(device)

    with torch.no_grad():
        spk_emb = model.xvector_model(ref_t).mean(dim=0, keepdim=True)
        enhanced = model.do_enh(mix_t, spk_emb)

    sf.write(save_path, enhanced.cpu().numpy(), sr)
    print("[Test Saved]", save_path)


# ============================================================
# Train loop
# ============================================================
def train_model(train_loader, val_loader, test_loader,
                model, config, epochs, patience, lr,
                ckpt_dir, plot_dir, result_dir, log_file):

    def log_print(*args):
        print(*args)
        with open(log_file, "a") as f:
            print(*args, file=f)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val = float("inf")
    patience_cnt = 0

    sr = config.sample_rate

    train_hist, val_hist = [], []

    log_print("===== Start Training =====")

    for epoch in range(1, epochs + 1):
        log_print(f"\n===== Epoch {epoch}/{epochs} =====")

        train_losses = []
        for batch in tqdm(train_loader, desc="Train"):
            train_losses.append(train_step(model, batch, optimizer, sr))
        avg_train = float(np.mean(train_losses))

        val_losses = []
        for batch in tqdm(val_loader, desc="Val"):
            val_losses.append(eval_step(model, batch, sr))
        avg_val = float(np.mean(val_losses))

        log_print(f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        train_hist.append(avg_train)
        val_hist.append(avg_val)

        scheduler.step()

        # Best model save
        if avg_val < best_val:
            best_val = avg_val
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.bin"))
            log_print(f" → New Best! (Val = {best_val:.4f})")
        else:
            patience_cnt += 1
            log_print(f" → No improvement ({patience_cnt}/{patience})")
            if patience_cnt >= patience:
                log_print("Early stopping triggered!")
                break

    # ============================================================
    # Final Test
    # ============================================================
    log_print("\n===== Final Test Evaluation =====")
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_model.bin")))
    model.eval()

    test_losses = []
    for batch in tqdm(test_loader, desc="Test"):
        test_losses.append(eval_step(model, batch, sr))
    final_test_loss = float(np.mean(test_losses))
    log_print(f"[Final Test Loss] {final_test_loss:.4f}")

    run_test_inference(
        model,
        "Dataset/Test_Dataset/Mix/mix_003.wav",
        "Dataset/Test_Dataset/Target/enrollment_003.wav",
        sr,
        os.path.join(result_dir, "test_final_003.wav")
    )

    # Plot
    epochs_range = list(range(1, len(train_hist) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_hist, label="Train")
    plt.plot(epochs_range, val_hist, label="Val")
    plt.axhline(final_test_loss, color="red", linestyle="--", label=f"Test Avg: {final_test_loss:.4f}")
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.savefig(os.path.join(plot_dir, "loss_history.png"))

    # CSV
    csv_path = os.path.join(plot_dir, "loss_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "final_test_loss"])
        for ep, (tr, va) in enumerate(zip(train_hist, val_hist), start=1):
            if ep == len(train_hist):
                writer.writerow([ep, tr, va, final_test_loss])
            else:
                writer.writerow([ep, tr, va, ""])

    log_print(f"Saved plot and CSV to {plot_dir}")


# ============================================================
# Main
# ============================================================
import argparse
from datetime import timedelta, timezone

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--type", type=str, default="full", choices=["full", "attn"])
    parser.add_argument("--open_spk_ffn", type=int, default=0)
    parser.add_argument("--open_pre_ffn", type=int, default=0,
                        help="1 = Conformer 이전 FFN/Conv도 FT")

    args = parser.parse_args()

    # Directory setup
    KST = timezone(timedelta(hours=9))
    timestamp = datetime.now(KST).strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("FineTuning", timestamp)

    ckpt_dir = os.path.join(base_dir, "checkpoints")
    plot_dir = os.path.join(base_dir, "plots")
    result_dir = os.path.join(base_dir, "results")
    log_file = os.path.join(base_dir, "log.txt")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print(f"[Save Dir] {base_dir}")

    # Dataset
    train_ds = ConVoiFilterDataset("Dataset/Train_Dataset")
    val_ds   = ConVoiFilterDataset("Dataset/Val_Dataset")
    test_ds  = ConVoiFilterDataset("Dataset/Test_Dataset")

    # Model load
    model, config = load_local_voicefilter(
        depth=args.depth,
        type=args.type,
        open_spk_ffn=args.open_spk_ffn,
        open_pre_ffn=args.open_pre_ffn
    )
    model.to(device)

    sr = config.sample_rate
    chunk_size = config.enh_chunk_size * sr
    batch_size = 12

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=partial(collate_fn, chunk_size=chunk_size))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=partial(collate_fn, chunk_size=chunk_size))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=partial(collate_fn, chunk_size=chunk_size))

    train_model(
        train_loader, val_loader, test_loader,
        model, config,
        epochs=50, patience=5, lr=1e-5,
        ckpt_dir=ckpt_dir, plot_dir=plot_dir,
        result_dir=result_dir, log_file=log_file
    )