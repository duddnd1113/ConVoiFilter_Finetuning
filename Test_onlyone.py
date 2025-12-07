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
print(device)

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

    chunks = np.array(chunks, dtype=np.float32)
    chunks = torch.from_numpy(chunks).unsqueeze(1)
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
    ckpt_path   = "/root/VoiceFiltering_finetuning/FineTuning/2025-12-04_20-12-28/checkpoints/best_model_val.bin"
    # ckpt_path   = "pretrained/pytorch_model.bin"

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

def calc_inference_loss(model, mix_wav, clean_wav, spk_emb, sr=16000):
    """
    Train/Val에서 쓰는 loss 계산 방식 그대로 사용.
    """
    model.eval()
    mix_wav = pad_to_chunk(mix_wav, model.wav_chunk_size)
    clean_wav = pad_to_chunk(clean_wav, model.wav_chunk_size)

    # tensor shapes 맞추기
    mix_t = torch.tensor(mix_wav).float().unsqueeze(0).to(device)     # [1, T]
    clean_t = torch.tensor(clean_wav).float().unsqueeze(0).to(device) # [1, T]
    lengths = torch.tensor([mix_t.size(-1)]).to(device)

    if spk_emb.dim() == 1:
        spk_emb = spk_emb.unsqueeze(0).to(device)   # [1, D]
    else:
        spk_emb = spk_emb.to(device)

    with torch.no_grad():
        out = model(
            speech=mix_t,
            speech_lengths=lengths,
            target_speech=clean_t,
            target_spk_embedding=spk_emb,
        )

    return out.loss.item()


#-----------------------------------------------------------------------------
# 6. Main – Test only
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. 모델 로드
    model = load_voicefilter_model_local().to(device)

    # 2. 테스트 파일 로드
    mix_path = "/root/VoiceFiltering_finetuning/Dataset/Test_Dataset/Mix/mix_648.wav" # 노이지 데이터
    ref_path = "/root/VoiceFiltering_finetuning/Dataset/Test_Dataset/Target/enrollment_648.wav" # 타겟 데이터
    clean_path = "/root/VoiceFiltering_finetuning/Dataset/Test_Dataset/Clean/target_648.wav" # 정답 데이터

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

    mix_wav = load_wav_hf(mix_path)
    clean_wav = load_wav_hf(clean_path)
    ref_wav = load_wav_hf(ref_path)

    # speaker embedding
    with torch.no_grad():
        spk_emb = cal_xvector_sincnet_embedding(model.xvector_model,
                                                ref_wav, sr=16000).to(device)

    # === Loss (Train/Val과 동일한 방식) ===
    test_loss = calc_inference_loss(model,
                                    mix_wav,
                                    clean_wav,
                                    spk_emb,
                                    sr=16000)

    print(f"[Inference SI-SNR Loss] {test_loss:.4f}")
