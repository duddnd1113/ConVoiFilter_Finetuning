#-----------------------------------------------------------------------------
# 경고문 안 뜨게 처리
#-----------------------------------------------------------------------------
import logging
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


#-----------------------------------------------------------------------------
# Train.py  — HF from_pretrained와 완전 동일 파이프라인
#-----------------------------------------------------------------------------

import os
import glob
import json
import torch
import soundfile as sf
import librosa
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from src.model.modeling_enh import VoiceFilter
from src.model.configuration_voicefilter import VoiceFilterConfig


#-----------------------------------------------------------------------------
# 0. GPU 체크
#-----------------------------------------------------------------------------
use_gpu = True
if not torch.cuda.is_available():
    use_gpu = False
device = torch.device("cuda" if use_gpu else "cpu")


#-----------------------------------------------------------------------------
# 1. HF inference-style WAV loader
#-----------------------------------------------------------------------------
def load_wav_hf(path, target_sr=16000):
    """
    HF inference와 동일한 방식으로 wav를 로드
    (soundfile → librosa fallback + float32 + mono + resample)
    """
    try:
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = librosa.resample(wav, sr, target_sr)
        return wav.astype(np.float32)
    except:
        wav, sr = librosa.load(path, sr=target_sr, mono=True)
        return wav.astype(np.float32)


#-----------------------------------------------------------------------------
# 2. HF inference-style padding
#-----------------------------------------------------------------------------
def pad_to_chunk(wav, chunk_size):
    rem = len(wav) % chunk_size
    if rem == 0:
        return wav
    pad_len = chunk_size - rem
    return np.concatenate([wav, np.zeros(pad_len, dtype=np.float32)])


#-----------------------------------------------------------------------------
# 3. HF inference-style xvector embedding
#-----------------------------------------------------------------------------
def cal_xvector_sincnet_embedding(xvector_model, ref_wav, sr=16000, max_length=5):
    chunk_len = max_length * sr
    chunks = []

    for i in range(0, len(ref_wav), chunk_len):
        w = ref_wav[i:i+chunk_len]
        if len(w) < chunk_len:
            w = np.concatenate([w, np.zeros(chunk_len - len(w))])
        chunks.append(w)

    chunks = torch.tensor(chunks, dtype=torch.float32).unsqueeze(1)
    if use_gpu:
        chunks = chunks.cuda()

    with torch.no_grad():
        emb = xvector_model(chunks)

    return emb.mean(dim=0).cpu()


#-----------------------------------------------------------------------------
# 4. Dataset – HF inference와 동일하게 수행
#-----------------------------------------------------------------------------
class ConVoiFilterDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000):
        self.sample_rate = sample_rate

        self.clean_files = sorted(glob.glob(os.path.join(root_dir, "Clean/*.wav")))
        self.mix_files   = sorted(glob.glob(os.path.join(root_dir, "Mix/*.wav")))
        self.ref_files   = sorted(glob.glob(os.path.join(root_dir, "Target/*.wav")))

        assert len(self.mix_files) == len(self.clean_files) == len(self.ref_files)
        print("=== Dataset Loaded ===")
        print(f"[Dataset Loaded] {root_dir} → {len(self.mix_files)} samples")

    def __len__(self):
        return len(self.mix_files)

    def __getitem__(self, idx):
        mix  = load_wav_hf(self.mix_files[idx])
        clean = load_wav_hf(self.clean_files[idx])
        ref   = load_wav_hf(self.ref_files[idx])

        # mix / clean 길이 동일하게 맞춤
        T = min(len(mix), len(clean))
        mix = mix[:T]
        clean = clean[:T]

        mix  = torch.tensor(mix, dtype=torch.float32).unsqueeze(0)
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        ref   = torch.tensor(ref, dtype=torch.float32).unsqueeze(0)

        return {"mix": mix, "clean": clean, "ref": ref}


#-----------------------------------------------------------------------------
# 5. Local model loader that mimics HF from_pretrained()
#-----------------------------------------------------------------------------
def load_voicefilter_model_local():
    config_path = "pretrained/config.json"
    ckpt_path   = "pretrained/pytorch_model.bin"

    # HF와 동일한 config 로드
    config = VoiceFilterConfig.from_pretrained(config_path)

    # HF와 동일한 방식으로 모델 생성
    model = VoiceFilter(config)

    # strict=False 로드 → HF behavior 동일
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # print("[Local Model Load] Missing:", missing)
    # print("[Local Model Load] Unexpected:", unexpected)

    model.eval()

    # xvector freeze & eval (HF behavior)
    model.xvector_model.eval()
    for p in model.xvector_model.parameters():
        p.requires_grad = False

    print("\n=== Local ConVoiFilter Loaded ===")
    print("Local ConVoiFilter loaded EXACTLY like HF.\n")

    return model


#-----------------------------------------------------------------------------
# 6. Inference wrapper (도_enh() 그대로 사용)
#-----------------------------------------------------------------------------
def enhance_audio(model, mix_wav, ref_wav, sr=16000):
    chunk_size = model.wav_chunk_size

    mix_wav = pad_to_chunk(mix_wav, chunk_size)
    ref_wav = pad_to_chunk(ref_wav, chunk_size)

    mix_tensor = torch.tensor(mix_wav, dtype=torch.float32).to(device)
    ref_tensor = torch.tensor(ref_wav, dtype=torch.float32).to(device)

    # speaker embedding
    with torch.no_grad():
        spk_emb = cal_xvector_sincnet_embedding(model.xvector_model,
                                                ref_tensor.cpu().numpy(),
                                                sr=sr)
        spk_emb = spk_emb.to(device)

    # enhancement
    with torch.no_grad():
        enhanced = model.do_enh(mix_tensor, spk_emb)

    return enhanced.cpu().numpy()


#-----------------------------------------------------------------------------
# 7. Training Loop (학습 전용) 
# !!!! 제일 중요 !!!!
#-----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer):
    model.train()

    chunk_size = model.wav_chunk_size
    total_loss = 0

    for batch in dataloader:
        mix   = batch["mix"].to(device)     # (B,1,T)
        clean = batch["clean"].to(device)   # (B,1,T)
        ref   = batch["ref"].to(device)     # (B,1,T_ref)


        # ===== 1) numpy 변환해서 padding =====
        # numpy로 변환해서 1D로 flatten
        mix_np = mix.squeeze().cpu().numpy().astype(np.float32)
        clean_np = clean.squeeze().cpu().numpy().astype(np.float32)

        # padding
        rem = mix_np.shape[-1] % chunk_size
        if rem != 0:
            pad_len = chunk_size - rem
            pad = np.zeros(pad_len, dtype=np.float32)
            mix_np = np.concatenate([mix_np, pad])
            clean_np = np.concatenate([clean_np, pad])


        # back to tensor
        mix_tensor   = torch.tensor(mix_np,   dtype=torch.float32).unsqueeze(0).to(device)
        clean_tensor = torch.tensor(clean_np, dtype=torch.float32).unsqueeze(0).to(device)

        mix_len = torch.tensor([mix_tensor.shape[-1]]).to(device)

        # ===== 2) reference embedding =====
        with torch.no_grad():
            spk_emb = model.xvector_model(ref)
            spk_emb = spk_emb.mean(dim=0, keepdim=True)  # (1, emb_dim)

        # ===== 3) model forward =====
        output = model(
            speech=mix_tensor.squeeze(1),
            speech_lengths=mix_len,
            target_speech=clean_tensor.squeeze(1),
            target_spk_embedding=spk_emb
        )

        loss = output.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    return total_loss / len(dataloader)



#-----------------------------------------------------------------------------
# 8. Main
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    # === Dataset ===
    train_dataset = ConVoiFilterDataset("Dataset/Train_Dataset")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # === Model ===
    model = load_voicefilter_model_local().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # === Training 1 epoch (예시) ===
    print("\n=== Training Start ===")
    loss = train_one_epoch(model, train_loader, optimizer)
    print(f"[Epoch 1] Loss = {loss:.4f}\n")

    # === Save checkpoint ===
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model_epoch1.bin")
    print("Saved checkpoint: checkpoints/model_epoch1.bin")

    # === Inference test ===
    test_mix  = load_wav_hf("test_data/침착맨+한로로.wav")
    
    test_ref  = load_wav_hf("test_data/침착맨 타겟.wav")

    enhanced = enhance_audio(model, test_mix, test_ref, sr=16000)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    out_path = f"results/enhanced_{ts}.wav"
    sf.write(out_path, enhanced, 16000)

    print(f"\n🎉 Done! Enhanced audio saved at: {out_path}\n")
