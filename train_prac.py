# -----------------------------
# Step 0 : 먼저 데이터셋을 불러오기 
# -----------------------------

import os
import glob
import torchaudio
import torch
from torch.utils.data import Dataset
import numpy as np

# GPU 체크
use_gpu = True
if not torch.cuda.is_available():
    use_gpu = False

class ConVoiFilterDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000):
        self.sample_rate = sample_rate

        clean_dir = os.path.join(root_dir, "Clean")
        mix_dir = os.path.join(root_dir, "Mix")
        target_dir = os.path.join(root_dir, "Target")

        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
        self.mix_files = sorted(glob.glob(os.path.join(mix_dir, "*.wav")))
        self.ref_files = sorted(glob.glob(os.path.join(target_dir, "*.wav")))

        assert len(self.clean_files) == len(self.mix_files) == len(self.ref_files)
        print(f"[Dataset Loaded] {root_dir} → {len(self.mix_files)} samples")


    def _load_wav_hf(self, path):
        # HF inference와 동일한 방식
        import soundfile as sf
        import librosa

        try:
            wav, sr = sf.read(path)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != self.sample_rate:
                wav = librosa.resample(wav, sr, self.sample_rate)
            return wav.astype(np.float32)
        except:
            wav, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            return wav.astype(np.float32)


    def __getitem__(self, idx):
        mix = self._load_wav_hf(self.mix_files[idx])
        clean = self._load_wav_hf(self.clean_files[idx])
        ref = self._load_wav_hf(self.ref_files[idx])

        # len match like inference code
        T = min(len(mix), len(clean))
        mix = mix[:T]
        clean = clean[:T]

        # torch format
        mix = torch.tensor(mix).unsqueeze(0)
        clean = torch.tensor(clean).unsqueeze(0)
        ref = torch.tensor(ref).unsqueeze(0)

        return {
            "mix": mix,      
            "clean": clean,  
            "ref": ref
        }


    def __len__(self):
        return len(self.mix_files)



from torch.utils.data import DataLoader

train_dataset = ConVoiFilterDataset("Dataset/Train_Dataset")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

for batch in train_loader:
    mix = batch["mix"]
    clean = batch["clean"]
    ref = batch["ref"]

    print("mix:", mix.shape)
    print("clean:", clean.shape)
    print("ref:", ref.shape)
    break


# ------------------------------------------------------------
# Step 1 : 이전 Checkpoint에서 학습된 모델 가중치 및 구조 불러오기
#          (Checkpoint가 없다면 사전학습 가중치 그대로 가져오도록 해야함) 
# ------------------------------------------------------------
import json
import torch
from src.model.modeling_enh import VoiceFilter
from src.model.configuration_voicefilter import VoiceFilterConfig
from voice_filter_infer_modify import *

def load_voicefilter_model():
    config_path = "pretrained/config.json"
    checkpoint_path = "pretrained/pytorch_model.bin"

    # 1) Load JSON dictionary
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # 2) Convert dict → VoiceFilterConfig
    config = VoiceFilterConfig.from_dict(config_dict)

    # 3) Initialize model
    model = VoiceFilter(config)

    # 4) Load weights
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)

    model.eval()
    print("ConVoiFilter model loaded successfully!")
    return model

def load_voicefilter_model_local():
    config_path = "pretrained/config.json"
    checkpoint_path = "pretrained/pytorch_model.bin"

    # 1) HF-style config loading
    from src.model.configuration_voicefilter import VoiceFilterConfig
    config = VoiceFilterConfig.from_pretrained(config_path)

    # 2) HF-style model initialization
    model = VoiceFilter(config)

    # 3) HF-style weight loading
    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # 4) Set eval mode (HF behavior)
    model.eval()

    # 5) HF also freezes xvector model
    model.xvector_model.eval()
    for p in model.xvector_model.parameters():
        p.requires_grad = False

    print("Local VoiceFilter loaded EXACTLY like HF.")
    return model

# ------------------------------------------------------------
# Step 2 : Forward 호출 + Loss 출력 테스트
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = load_voicefilter_model_local().to(device)

# 데이터 1개 뽑아 테스트
for batch in train_loader:
    mix = batch["mix"].to(device)      # (1, 1, T)
    clean = batch["clean"].to(device)  # (1, 1, T)
    ref = batch["ref"].to(device)      # (1, 1, T_ref)
    break

# ------------------------------------------------------------
# Step 2 : Fast Forward + Save & Listen
# ------------------------------------------------------------
import soundfile as sf
import os

print("\n[Fast Forward Test Start]\n")

# 1개 배치 가져오기 (이미 위에서 mix, clean, ref 가져왔지?)
mix = mix.squeeze(1)       # (1, T)
clean = clean.squeeze(1)
ref = ref.squeeze(1)

# numpy 변환
mix_np = mix.squeeze().cpu().detach().numpy()
ref_np = ref.squeeze().cpu().detach().numpy()

# chunk size 얻기
chunk_size = model.wav_chunk_size

# pad 함수
def pad_to_chunk(wav, chunk):
    rem = len(wav) % chunk
    if rem == 0:
        return wav
    pad_len = chunk - rem
    return np.concatenate([wav, np.zeros(pad_len, dtype=wav.dtype)])

# padding
mix_np = pad_to_chunk(mix_np, chunk_size)
ref_np = pad_to_chunk(ref_np, chunk_size)

# tensor 변환
mix_tensor = torch.tensor(mix_np, dtype=torch.float32).to(device)
ref_tensor = torch.tensor(ref_np, dtype=torch.float32).to(device)

# speaker embedding
with torch.no_grad():
    speaker_embedding = cal_xvector_sincnet_embedding(
        model.xvector_model,
        ref_tensor.cpu().numpy(),
        sr=16000
    )
    speaker_embedding = speaker_embedding.to(device)

# enhancement
with torch.no_grad():
    enhanced = model.do_enh(mix_tensor, speaker_embedding)

enhanced_np = enhanced.cpu().numpy()

# Save folder
out_dir = "./fast_test_output2"
os.makedirs(out_dir, exist_ok=True)

sf.write(os.path.join(out_dir, "mix.wav"),     mix_np,      16000)
sf.write(os.path.join(out_dir, "clean.wav"),   clean.squeeze().cpu().detach().numpy(), 16000)
sf.write(os.path.join(out_dir, "enhanced.wav"), enhanced_np, 16000)

print("Saved mix.wav, clean.wav, enhanced.wav in fast_test_output/")
print("You can now open and listen to enhanced.wav")
