from src.model.modeling_enh import VoiceFilter
import torch
from huggingface_hub import hf_hub_download
import os
import glob
import csv
from tqdm import tqdm
import librosa
import numpy as np
import soundfile as sf
import os
from datetime import datetime

use_gpu = True
if use_gpu:
    if not torch.cuda.is_available():
        use_gpu = False
        
def cal_xvector_sincnet_embedding(xvector_model, ref_wav, max_length=5, sr=16000):
    wavs = []
    for i in range(0, len(ref_wav), max_length*sr):
        wav = ref_wav[i:i + max_length*sr]
        wav = np.concatenate([wav, np.zeros(max(0, max_length * sr - len(wav)))])
        wavs.append(wav)
    wavs = torch.from_numpy(np.stack(wavs))
    if use_gpu:
        wavs = wavs.cuda()
    embed = xvector_model(wavs.unsqueeze(1).float())
    return torch.mean(embed, dim=0).detach().cpu()


def load_wav(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # stereo → mono
    return wav, sr

def save_wav(path, audio, sr):
    sf.write(path, audio, sr)

def pad_wav_to_chunk(wav, chunk_size):
    length = len(wav)
    if length % chunk_size == 0:
        return wav
    pad_len = chunk_size - (length % chunk_size)
    return np.concatenate([wav, np.zeros(pad_len, dtype=wav.dtype)])

if __name__ == "__main__":
    # Load models
    repo_id = 'nguyenvulebinh/voice-filter'

    enh_model = VoiceFilter.from_pretrained(repo_id, cache_dir='./cache')
    if use_gpu:
        enh_model = enh_model.cuda()

    print("Model loaded.")

    # --------------------------------------------------------
    # 1. 여기에 테스트할 noisy 음성
    # --------------------------------------------------------
    noisy_wav_path = "/Users/youngwoong/Desktop/YONSEI/2025-2/딥러닝과 응용/teamproject_git/VoiceFiltering_finetuning/data/mix_fixed.wav"       # 여기에 너 파일 경로 넣기
    ref_speaker_path = "/Users/youngwoong/Desktop/YONSEI/2025-2/딥러닝과 응용/teamproject_git/VoiceFiltering_finetuning/data/침착맨1_fixed.wav"    # 타겟 사람 목소리 파일
    

    chunk_size = enh_model.wav_chunk_size


    noisy, sr = load_wav(noisy_wav_path)
    ref, _ = load_wav(ref_speaker_path)

    noisy = pad_wav_to_chunk(noisy, chunk_size)
    ref = pad_wav_to_chunk(ref, chunk_size)

    # 텐서 변환
    noisy_tensor = torch.tensor(noisy).float().unsqueeze(0)  # (1, T)
    ref_tensor = torch.tensor(ref).float().unsqueeze(0)

    T_noisy = noisy_tensor.shape[1]
    T_ref = ref_tensor.shape[1]

    speech_lengths = torch.tensor([T_noisy], dtype=torch.long)
    target_speech_lengths = torch.tensor([T_ref], dtype=torch.long)

    if use_gpu:
        noisy_tensor = noisy_tensor.cuda()
        ref_tensor = ref_tensor.cuda()

    # --------------------------------------------------------
    # 2. 모델 inference
    # --------------------------------------------------------
    with torch.no_grad():
        enhanced = enh_model(
            noisy_tensor,
            speech_lengths,
            ref_tensor,          # target_spk_embedding
            ref_tensor,          # target_speech
            target_speech_lengths
        )
    
    enhanced = enhanced.squeeze().cpu().numpy()

    # --------------------------------------------------------
    # 3. 저장
    # --------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = f"./results/{timestamp}"

    os.makedirs(result_dir, exist_ok=True)

    # 저장할 파일 경로
    out_path = os.path.join(result_dir, "enhanced_output.wav")

    save_wav(out_path, enhanced, sr)

    print(f"Done! -> Enhanced audio saved at: {out_path}")   