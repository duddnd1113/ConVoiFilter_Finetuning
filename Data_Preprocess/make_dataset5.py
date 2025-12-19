import os
import random
from glob import glob

import numpy as np
import librosa
import soundfile as sf


def rms_db(audio: np.ndarray, eps: float = 1e-10) -> float:
    rms = np.sqrt(np.mean(audio ** 2) + eps)
    return 20.0 * np.log10(rms + eps)


def rms(audio: np.ndarray, eps: float = 1e-10) -> float:
    return np.sqrt(np.mean(audio**2) + eps)


def random_segment_from_array(audio: np.ndarray, seg_len: int) -> np.ndarray:
    """
    audio에서 길이 seg_len짜리 구간을 랜덤으로 잘라서 반환.
    너무 짧으면 타일링해서 길이 확보 후 자르기.
    """
    if len(audio) < seg_len:
        rep = int(np.ceil(seg_len / len(audio)))
        audio = np.tile(audio, rep)
    max_start = len(audio) - seg_len
    start = random.randint(0, max_start)
    return audio[start:start + seg_len]


# clean 화자 (data_normalized: 이미 RMS 맞춰둔 상태)
norm_dir   = "data_normalized"      

# 같은 ID의 enrollment
enroll_dir = "data_enrollment"      

# 배경 노이즈(카페 등) - 선택
noise_dir  = r"C:\Users\ejh99\deleteplease\hasibal"     # 폴더 경로

# 출력 루트 폴더 (split별 하위 폴더 생성)
base_out_root = "fine_tune"

#오디오 / 세그먼트 설정
target_sr   = 16000         # VoiceFilter sample_rate
seg_len_sec = 5.0
seg_len     = int(seg_len_sec * target_sr)

# target 한 명당 mixture 개수
mixtures_per_target = 40

# interferer 화자 수 범위
min_interferer = 1
max_interferer = 2

# interferer 볼륨 스케일 (지금은 비율로 맞추기 때문에 alpha는 사용 안 함)
alpha_min = 0.5
alpha_max = 0.8

# 난이도별 interferer 비율(EASY / HARD)
easy_ratio_min, easy_ratio_max = 0.4, 0.6   # target 대비 비율
hard_ratio_min, hard_ratio_max = 0.8, 1.0   # target 대비 비율
easy_prob = 0.5                             # easy/hard 섞는 비율

# noise는 target 대비 고정 비율
noise_ratio_target = 0.5                    # target RMS의 0.5배 정도

# train/valid/test 비율 (화자 단위 split)
train_ratio = 0.8
valid_ratio = 0.1
test_ratio  = 0.1

random.seed(2025)
np.random.seed(2025)



norm_paths = sorted(glob(os.path.join(norm_dir, "*.wav")))
if not norm_paths:
    raise RuntimeError(f"[ERROR] '{norm_dir}' 안에 wav 파일이 없습니다.")

ids = [os.path.splitext(os.path.basename(p))[0] for p in norm_paths]  
id_to_norm = {id_: os.path.join(norm_dir, id_ + ".wav") for id_ in ids}

id_to_enroll = {}
for id_ in ids:
    epath = os.path.join(enroll_dir, id_ + ".wav")
    if os.path.exists(epath):
        id_to_enroll[id_] = epath
    else:
        print(f"[WARN] enrollment for ID={id_} not found. fallback to normalized clean.")
        id_to_enroll[id_] = id_to_norm[id_]

print("[INFO] Preloading clean/enroll wavs into memory...")
clean_wavs = {}
enroll_wavs = {}
for id_ in ids:
    c_wav, _ = librosa.load(id_to_norm[id_], sr=target_sr, mono=True)
    e_wav, _ = librosa.load(id_to_enroll[id_], sr=target_sr, mono=True)
    clean_wavs[id_] = c_wav.astype(np.float32)
    enroll_wavs[id_] = e_wav.astype(np.float32)

# noise wav도 미리 다 읽어두기
noise_paths = sorted(glob(os.path.join(noise_dir, "*.wav")))
noise_wavs = []
if not noise_paths:
    print(f"[INFO] '{noise_dir}' 안에 노이즈가 없습니다. 화자 interferer만 사용합니다.")
else:
    print(f"[INFO] background noise files: {len(noise_paths)}개")
    for npath in noise_paths:
        n_wav, _ = librosa.load(npath, sr=target_sr, mono=True)
        noise_wavs.append(n_wav.astype(np.float32))

print(f"[INFO] speakers in data_normalized : {len(ids)}명")
print(f"[INFO] mixtures_per_target         : {mixtures_per_target}")
print(f"[INFO] segment length (sec)        : {seg_len_sec}")
print(f"[INFO] interferer scale alpha      : {alpha_min} ~ {alpha_max}")



ids_shuffled = ids.copy()
random.shuffle(ids_shuffled)

n = len(ids_shuffled)
n_train = int(n * train_ratio)
n_valid = int(n * valid_ratio)
# 나머지는 전부 test로
n_test  = n - n_train - n_valid

train_ids = ids_shuffled[:n_train]
valid_ids = ids_shuffled[n_train:n_train + n_valid]
test_ids  = ids_shuffled[n_train + n_valid:]

print("\n[INFO] Speaker split:")
print(f"  train: {len(train_ids)}명")
print(f"  valid: {len(valid_ids)}명")
print(f"  test : {len(test_ids)}명")

splits = {
    "train": train_ids,
    "valid": valid_ids,
    "test":  test_ids,
}


for split_name, split_id_list in splits.items():
    if not split_id_list:
        print(f"\n[WARN] split '{split_name}' 에 할당된 화자가 없습니다. 스킵합니다.")
        continue

    print(f"\n=== [{split_name.upper()}] {len(split_id_list)} speakers ===")

    # split별 출력 폴더
    out_root = os.path.join(base_out_root, split_name)
    out_mix = os.path.join(out_root, "mix")
    out_tgt = os.path.join(out_root, "target")
    out_enr = os.path.join(out_root, "enrollment")
    os.makedirs(out_mix, exist_ok=True)
    os.makedirs(out_tgt, exist_ok=True)
    os.makedirs(out_enr, exist_ok=True)

    # split마다 파일 번호 001부터 시작
    sample_idx = 1

    for t_idx, target_id in enumerate(split_id_list, start=1):
        print(f"[{split_name}] Target {t_idx}/{len(split_id_list)}  ID={target_id}")

        t_all = clean_wavs[target_id]
        e_all = enroll_wavs[target_id]

        # target/enroll 길이가 너무 짧으면 타일링
        if len(t_all) < seg_len * 2:
            rep = int(np.ceil((seg_len * 2) / len(t_all)))
            t_all = np.tile(t_all, rep)
        if len(e_all) < seg_len:
            rep = int(np.ceil(seg_len / len(e_all)))
            e_all = np.tile(e_all, rep)

        for k in range(mixtures_per_target):
            # 4-1) target / enrollment segment
            target_seg = random_segment_from_array(t_all, seg_len)
            enroll_seg = random_segment_from_array(e_all, seg_len)

            rms_t = rms(target_seg)

            # 4-2) interferer speech (같은 split 안의 다른 화자들만 사용)
            interferer_speech = np.zeros_like(target_seg)
            other_ids = [i for i in split_id_list if i != target_id]
            num_interferer = 0

            if other_ids:
                num_interferer = random.randint(min_interferer, max_interferer)
                for _ in range(num_interferer):
                    iid = random.choice(other_ids)
                    i_all = clean_wavs[iid]
                    i_seg = random_segment_from_array(i_all, seg_len)
                    interferer_speech += i_seg

            # 4-3) background noise segment (아직 스케일X)
            noise_seg = np.zeros_like(target_seg)
            if noise_wavs:
                n_all = random.choice(noise_wavs)
                noise_seg = random_segment_from_array(n_all, seg_len)

            # 4-4) easy / hard 모드 결정 + interferer 스케일 비율
            if random.random() < easy_prob:
                mode = "easy"
                r_min, r_max = easy_ratio_min, easy_ratio_max
            else:
                mode = "hard"
                r_min, r_max = hard_ratio_min, hard_ratio_max

            interferer_ratio = np.random.uniform(r_min, r_max)  # target 대비 비율

            rms_i = rms(interferer_speech)
            if rms_t > 1e-6 and rms_i > 1e-6:
                gain_inter = interferer_ratio * rms_t / rms_i
                interferer_scaled = interferer_speech * gain_inter
            else:
                interferer_scaled = interferer_speech

            # 4-5) noise는 target 기준 noise_ratio_target 배로 스케일
            rms_n = rms(noise_seg)
            if rms_t > 1e-6 and rms_n > 1e-6:
                gain_noise = noise_ratio_target * rms_t / rms_n
                noise_scaled = noise_seg * gain_noise
            else:
                noise_scaled = noise_seg

            # 4-6) interferer_total = 화자 interferer + noise
            interferer_total = interferer_scaled + noise_scaled

            # 거의 무음이면 최소한의 노이즈 추가
            if np.max(np.abs(interferer_total)) < 1e-6:
                interferer_total += 1e-4 * np.random.randn(*interferer_total.shape).astype(np.float32)

            # 4-7) 최종 mix = target + interferer_total
            mix = target_seg + interferer_total

            # 4-8) 클리핑 방지
            max_abs = np.max(np.abs(mix))
            if max_abs > 1.0:
                scale = 0.999 / max_abs
                mix            *= scale
                target_seg     *= scale
                enroll_seg     *= scale
                interferer_total *= scale

            mix_rms = rms_db(mix)
            # 너무 작은 RMS면 스킵
            if mix_rms < -50.0:
                continue

            # 4-9) 파일 저장
            sample_id = f"{sample_idx:03d}"
            mix_name = f"mix_{sample_id}.wav"
            tgt_name = f"target_{sample_id}.wav"
            enr_name = f"enrollment_{sample_id}.wav"

            sf.write(os.path.join(out_mix, mix_name), mix,        target_sr)
            sf.write(os.path.join(out_tgt, tgt_name), target_seg, target_sr)
            sf.write(os.path.join(out_enr, enr_name), enroll_seg, target_sr)

            if sample_idx % 50 == 0:
                print(
                    f"  -> [{split_name}] ID={sample_id} (target={target_id}), "
                    f"mode={mode}, "
                    f"interferer_ratio={interferer_ratio:.2f}, "
                    f"noise_ratio={noise_ratio_target:.2f}, "
                    f"interferer_cnt={num_interferer}, "
                    f"mix_rms={mix_rms:.1f} dB"
                )

            sample_idx += 1

print("\n[Done] fine_tune/train, fine_tune/valid, fine_tune/test 생성 완료!")
