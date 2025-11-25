#-----------------------------------------------------------------------------
# 경고문 안 뜨게 처리
#-----------------------------------------------------------------------------
import logging
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#-----------------------------------------------------------------------------

import os
import json
import torch
import soundfile as sf
import librosa
import numpy as np
from datetime import datetime

from src.model.modeling_enh import VoiceFilter
from src.model.configuration_voicefilter import VoiceFilterConfig


#-----------------------------------------------------------------------------
# 0. GPU 체크
#-----------------------------------------------------------------------------
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")


#-----------------------------------------------------------------------------
# 1. HF inference-style WAV loader
#-----------------------------------------------------------------------------
def load_wav_hf(path, target_sr=16000):
    """HF inference와 동일한 방식으로 wav 로드."""
    try:
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        return wav.astype(np.float32)
    except:
        wav, sr = librosa.load(path, sr=target_sr, mono=True)
        return wav.astype(np.float32)


#-----------------------------------------------------------------------------
# 2. Padding (HF와 동일)
#-----------------------------------------------------------------------------
def pad_to_chunk(wav, chunk_size):
    rem = len(wav) % chunk_size
    if rem == 0:
        return wav
    pad_len = chunk_size - rem
    return np.concatenate([wav, np.zeros(pad_len, dtype=np.float32)])


#-----------------------------------------------------------------------------
# 3. HF-style xvector embedding
#-----------------------------------------------------------------------------
def cal_xvector_sincnet_embedding(xvector_model, ref_wav, sr=16000, max_length=5):
    chunk_len = max_length * sr
    chunks = []

    for i in range(0, len(ref_wav), chunk_len):
        w = ref_wav[i:i + chunk_len]
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
# 4. 로컬 모델 로더 (HF from_pretrained 완벽 재현)
#-----------------------------------------------------------------------------
def load_voicefilter_model_local():
    config_path = "pretrained/config.json"
    ckpt_path   = "pretrained/pytorch_model.bin"

    # config 로드
    config = VoiceFilterConfig.from_pretrained(config_path)

    # 모델 생성
    model = VoiceFilter(config)

    # 가중치 로드
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)

    # print("[Local Model Load] Missing:", missing)
    # print("[Local Model Load] Unexpected:", unexpected)
    print("\n=== Local ConVoiFilter Loaded ===")

    # inference mode
    model.eval()

    # xvector freeze
    model.xvector_model.eval()
    for p in model.xvector_model.parameters():
        p.requires_grad = False

    return model


#-----------------------------------------------------------------------------
# 5. Inference wrapper (do_enh 그대로)
#-----------------------------------------------------------------------------
def enhance_audio(model, mix_wav, ref_wav, sr=16000):
    chunk_size = model.wav_chunk_size

    mix_wav = pad_to_chunk(mix_wav, chunk_size)
    ref_wav = pad_to_chunk(ref_wav, chunk_size)

    mix_tensor = torch.tensor(mix_wav, dtype=torch.float32).to(device)
    ref_tensor = torch.tensor(ref_wav, dtype=torch.float32).to(device)

    # embedding
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
# 6. Main – Test only
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. 모델 로드
    model = load_voicefilter_model_local().to(device)

    # 2. 테스트 파일 로드
    mix_path = "test_data/침착맨+한로로.wav"
    ref_path = "test_data/한로로 타겟.wav"

    mix_wav = load_wav_hf(mix_path)
    ref_wav = load_wav_hf(ref_path)

    # 3. 음성 향상 실행
    enhanced_audio = enhance_audio(model, mix_wav, ref_wav, sr=16000)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = f"./results/{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    out_path = os.path.join(result_dir, "enhanced_output.wav")
    sf.write(out_path, enhanced_audio, 16000)

    print(f"🎉 Done! Enhanced audio saved at:\n➡  {out_path}\n")