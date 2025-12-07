#-----------------------------------------------------------------------------
# 경고문 제거
#-----------------------------------------------------------------------------
import logging
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import os
import numpy as np
import torch
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from src.model.modeling_enh import VoiceFilter
from src.model.configuration_voicefilter import VoiceFilterConfig

#-----------------------------------------------------------------------------
# GPU 체크
#-----------------------------------------------------------------------------
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print(f"[Device] {device}")


#-----------------------------------------------------------------------------
# WAV Loader
#-----------------------------------------------------------------------------
def load_wav_hf(path, target_sr=16000):
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
# Padding
#-----------------------------------------------------------------------------
def pad_to_chunk(wav, chunk_size):
    rem = len(wav) % chunk_size
    if rem == 0:
        return wav
    return np.concatenate([wav, np.zeros(chunk_size - rem, dtype=np.float32)])


#-----------------------------------------------------------------------------
# x-vector Embedding
#-----------------------------------------------------------------------------
def cal_xvector_sincnet_embedding(xvector_model, ref_wav, sr=16000, max_length=5):
    chunk_len = max_length * sr
    chunks = []

    for i in range(0, len(ref_wav), chunk_len):
        w = ref_wav[i:i + chunk_len]
        if len(w) < chunk_len:
            w = np.concatenate([w, np.zeros(chunk_len - len(w))])
        chunks.append(w)

    chunks = torch.tensor(np.array(chunks), dtype=torch.float32).unsqueeze(1)
    if use_gpu:
        chunks = chunks.cuda()

    with torch.no_grad():
        emb = xvector_model(chunks)

    return emb.mean(dim=0).cpu()


#-----------------------------------------------------------------------------
# Audio Enhancement
#-----------------------------------------------------------------------------
def enhance_audio(model, mix_wav, ref_wav, sr=16000):
    mix_wav = pad_to_chunk(mix_wav, model.wav_chunk_size)
    ref_wav = pad_to_chunk(ref_wav, model.wav_chunk_size)

    mix_tensor = torch.tensor(mix_wav, dtype=torch.float32).to(device)

    with torch.no_grad():
        spk_emb = cal_xvector_sincnet_embedding(model.xvector_model,
                                                ref_wav, sr=sr).to(device)
        enhanced = model.do_enh(mix_tensor, spk_emb)
    return enhanced.cpu().numpy()


#-----------------------------------------------------------------------------
# Loss 계산 함수
#-----------------------------------------------------------------------------
def calc_inference_loss(model, mix_wav, clean_wav, spk_emb, sr=16000):
    mix_wav = pad_to_chunk(mix_wav, model.wav_chunk_size)
    clean_wav = pad_to_chunk(clean_wav, model.wav_chunk_size)

    mix_t = torch.tensor(mix_wav).float().unsqueeze(0).to(device)
    clean_t = torch.tensor(clean_wav).float().unsqueeze(0).to(device)
    lengths = torch.tensor([mix_t.size(-1)]).to(device)

    if spk_emb.dim() == 1:
        spk_emb = spk_emb.unsqueeze(0)
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
# 모델 로더
#-----------------------------------------------------------------------------
def load_model(config_path, ckpt_path):
    config = VoiceFilterConfig.from_pretrained(config_path)
    model = VoiceFilter(config)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    model.eval()
    model.xvector_model.eval()
    for p in model.xvector_model.parameters():
        p.requires_grad = False

    return model.to(device)


#-----------------------------------------------------------------------------
# 단일 모델 Test 수행 + 오디오 저장
#-----------------------------------------------------------------------------
def run_model_test_and_save_audio(model, model_name, result_dir,
                                  mix_dir, clean_dir, ref_dir, sr=16000):

    mix_files = sorted([f for f in os.listdir(mix_dir) if f.endswith(".wav")])

    losses = []
    names = []

    print(f"\n▶ Running test for: {model_name}")

    for mix_file in tqdm(mix_files, desc=f"{model_name} evaluating", ncols=100):
        base = mix_file.replace("mix_", "").replace(".wav", "")

        mix_path = os.path.join(mix_dir, mix_file)
        clean_path = os.path.join(clean_dir, f"target_{base}.wav")
        ref_path = os.path.join(ref_dir, f"enrollment_{base}.wav")

        if not (os.path.exists(clean_path) and os.path.exists(ref_path)):
            continue

        mix_wav = load_wav_hf(mix_path)
        clean_wav = load_wav_hf(clean_path)
        ref_wav = load_wav_hf(ref_path)

        # Speaker embedding
        with torch.no_grad():
            spk_emb = cal_xvector_sincnet_embedding(model.xvector_model,
                                                    ref_wav, sr=sr).to(device)

        # Loss 계산
        loss = calc_inference_loss(model, mix_wav, clean_wav, spk_emb)
        losses.append(loss)
        names.append(base)

        # 오디오 향상
        enhanced_audio = enhance_audio(model, mix_wav, ref_wav, sr=sr)

        # 저장 파일명
        save_path = os.path.join(result_dir, f"{base}_{model_name}.wav")
        sf.write(save_path, enhanced_audio, sr)

    avg_loss = float(np.mean(losses))
    print(f"→ {model_name} Avg Loss = {avg_loss:.4f}")

    return names, losses, avg_loss


#-----------------------------------------------------------------------------
# 두 모델 비교
#-----------------------------------------------------------------------------
def compare_two_models(pretrained_model, finetuned_model,
                       mix_dir="Dataset/Test_Dataset/Mix",
                       clean_dir="Dataset/Test_Dataset/Clean",
                       ref_dir="Dataset/Test_Dataset/Target",
                       sr=16000):

    # 결과 저장 폴더 생성
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = f"./test_all_results/{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    # Pretrained
    names, pre_losses, pre_avg = run_model_test_and_save_audio(
        pretrained_model, "pretrained", result_dir,
        mix_dir, clean_dir, ref_dir, sr
    )

    # Fine-tuned
    _, fin_losses, fin_avg = run_model_test_and_save_audio(
        finetuned_model, "finetuned", result_dir,
        mix_dir, clean_dir, ref_dir, sr
    )


    #-----------------------------------------
    # 📈 Loss per sample plot (향상된 부분은 빨간색)
    #-----------------------------------------
    plt.figure(figsize=(14, 6))

    plt.plot(pre_losses, marker='o', label="Pretrained", linewidth=2)
    plt.plot(fin_losses, marker='o', label="Fine-tuned", linewidth=2)

    # x축 이름
    x_positions = np.arange(len(names))

    # x축 라벨을 조건부 색상으로 설정
    xtick_colors = []
    for i in range(len(names)):
        if fin_losses[i] < pre_losses[i]:
            xtick_colors.append('red')   # 🔥 개선된 경우 빨간색
        else:
            xtick_colors.append('black') # 기본값

    # 실제 ticks 적용
    plt.xticks(ticks=x_positions, labels=names, rotation=45)

    # tick 색상 적용
    ax = plt.gca()
    for tick_label, color in zip(ax.get_xticklabels(), xtick_colors):
        tick_label.set_color(color)

    plt.ylabel("SI-SNR Loss")
    plt.title("Loss Comparison per Test Sample (Improved samples in RED)")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(result_dir, "0_loss_comparison_plot.png")
    plt.savefig(plot_path)


    #-----------------------------------------
    # 📊 Average Loss bar chart
    #-----------------------------------------
    plt.figure(figsize=(6, 5))
    plt.bar(["Pretrained", "Fine-tuned"], [pre_avg, fin_avg])
    plt.ylabel("Average SI-SNR Loss")
    plt.title("Average Loss Comparison")
    plt.grid(axis='y')
    plt.tight_layout()

    avg_plot_path = os.path.join(result_dir, "0_average_loss_comparison.png")
    plt.savefig(avg_plot_path)

    print("\n================= RESULT =================")
    print(f"Pretrained Avg Loss : {pre_avg:.4f}")
    print(f"Fine-tuned Avg Loss : {fin_avg:.4f}")
    print(f"Results saved in folder: {result_dir}")
    print("================================================")

    return result_dir


#-----------------------------------------------------------------------------
# MAIN
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    PRETRAINED_CKPT = "pretrained/pytorch_model.bin"
    PRETRAINED_CONFIG = "pretrained/config.json"

    FINETUNED_CKPT = "/root/VoiceFiltering_finetuning/FineTuning/2025-12-05_02-49-41/checkpoints/best_model.bin"
    FINETUNED_CONFIG = "pretrained/config.json"

    pretrained_model = load_model(PRETRAINED_CONFIG, PRETRAINED_CKPT)
    print("✓ Pretrained model loaded.")

    finetuned_model = load_model(FINETUNED_CONFIG, FINETUNED_CKPT)
    print("✓ Fine-tuned model loaded.")

    compare_two_models(pretrained_model, finetuned_model)
