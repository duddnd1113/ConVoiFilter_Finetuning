from src.model.modeling_enh import VoiceFilter
import torch
import numpy as np
import soundfile as sf
import os
from datetime import datetime
import librosa



# GPU 체크
use_gpu = True
if not torch.cuda.is_available():
    use_gpu = False


# ---------------------------
# Utility functions
# ---------------------------

def load_wav(path, target_sr=16000):
    """
    1차로 soundfile로 시도하고,
    안 되면 librosa로 fallback.
    """
    try:
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        # 필요하면 리샘플
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return wav, sr
    except Exception as e:
        print("[sf.read 실패, librosa로 재시도]")
        wav, sr = librosa.load(path, sr=target_sr, mono=True)
        return wav, sr


def save_wav(path, audio, sr):
    sf.write(path, audio, sr)


def pad_wav_to_chunk(wav, chunk_size):
    """Make wav length exactly a multiple of chunk_size."""
    length = len(wav)
    remainder = length % chunk_size
    if remainder == 0:
        return wav
    pad_len = chunk_size - remainder
    return np.concatenate([wav, np.zeros(pad_len, dtype=wav.dtype)])


def cal_xvector_sincnet_embedding(xvector_model, ref_wav, sr=16000, max_length=5):
    """
    ref wav을 5초 단위로 나누어 xvector를 계산하고 평균을 내는 함수.
    """
    wavs = []
    chunk_len = max_length * sr

    for i in range(0, len(ref_wav), chunk_len):
        wav = ref_wav[i:i + chunk_len]
        if len(wav) < chunk_len:
            wav = np.concatenate([wav, np.zeros(chunk_len - len(wav))])
        wavs.append(wav)

    wavs = torch.from_numpy(np.stack(wavs)).float()  # (N, T)

    if use_gpu:
        wavs = wavs.cuda()

    with torch.no_grad():
        embed = xvector_model(wavs.unsqueeze(1))      # (N, 1, T) → (N, emb_dim)

    return torch.mean(embed, dim=0).cpu()             # (emb_dim,)


# ---------------------------
# Main inference
# ---------------------------
if __name__ == "__main__":
    repo_id = 'nguyenvulebinh/voice-filter'

    # 1. 모델 로드
    enh_model = VoiceFilter.from_pretrained(repo_id, cache_dir='./cache')

    if use_gpu:
        enh_model = enh_model.cuda()

    print("\nModel loaded.\n")

    print("Model state_dict keys and shapes:")
    for k, v in enh_model.state_dict().items():
        print(f"{k}: {tuple(v.shape)}")

    # 2. 테스트 파일 지정
    noisy_wav_path = "/Users/youngwoong/Desktop/YONSEI/2025-2/딥러닝과 응용/teamproject_git/VoiceFiltering_finetuning/test_data/침착맨+한로로.wav"
    ref_speaker_path = "/Users/youngwoong/Desktop/YONSEI/2025-2/딥러닝과 응용/teamproject_git/VoiceFiltering_finetuning/test_data/침착맨 타겟.wav"

    noisy, sr = load_wav(noisy_wav_path)
    ref, _ = load_wav(ref_speaker_path)

    chunk_size = enh_model.wav_chunk_size

    # 3. Audio padding → chunk 크기로 맞춤
    noisy = pad_wav_to_chunk(noisy, chunk_size)
    ref = pad_wav_to_chunk(ref, chunk_size)

    # 4. Tensor 변환
    noisy_tensor = torch.tensor(noisy).float()
    ref_tensor = torch.tensor(ref).float()

    if use_gpu:
        noisy_tensor = noisy_tensor.cuda()
        ref_tensor = ref_tensor.cuda()

    # 5. Reference speaker embedding 계산
    with torch.no_grad():
        speaker_embedding = cal_xvector_sincnet_embedding(
            enh_model.xvector_model, 
            ref_tensor.cpu().numpy(),
            sr=sr
        )
        speaker_embedding = speaker_embedding.cuda() if use_gpu else speaker_embedding

    print(speaker_embedding.mean(), speaker_embedding.std())
    # 6. do_enh() 사용 → 전체 forward 자동 처리
    with torch.no_grad():
        enhanced_audio = enh_model.do_enh(noisy_tensor, speaker_embedding)

    enhanced_audio = enhanced_audio.cpu().numpy()

    # 7. 날짜 기반 results 폴더 생성
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = f"./results/{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    out_path = os.path.join(result_dir, "enhanced_output.wav")
    save_wav(out_path, enhanced_audio, sr)

    print(f"🎉 Done! Enhanced audio saved at:\n➡  {out_path}\n")
