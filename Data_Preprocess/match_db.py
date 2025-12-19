import librosa
import soundfile as sf
import numpy as np
import os
import numpy as np
from glob import glob

def rms_db(audio: np.ndarray, eps: float = 1e-10) -> float:
    """
    audio: float형 numpy array, [-1, 1] 범위
    반환: RMS 기준 dB (20 * log10(rms))
    """
    rms = np.sqrt(np.mean(audio**2) + eps)
    return 20 * np.log10(rms + eps)



def match_to_target_db(in_path: str, out_path: str, target_db: float = -23.0):
    # 1) 로드 (원본 SR 유지, mono=False로 스테레오도 그대로)
    audio, sr = librosa.load(in_path, sr=None, mono=False)

    # librosa: mono -> (T,), stereo -> (C, T)
    if audio.ndim == 1:
        audio_for_rms = audio
    else:
        # 여러 채널일 때는 전체 채널 기준 RMS
        audio_for_rms = audio.reshape(-1)

    # 2) 현재 RMS dB 계산
    cur_db = rms_db(audio_for_rms)

    # 3) 필요한 gain 계산
    db_change = target_db - cur_db
    gain = 10 ** (db_change / 20.0)

    audio_out = audio * gain

    # 4) 클리핑 방지 ([-1, 1] 넘어가면 스케일 다운)
    max_abs = np.max(np.abs(audio_out))
    if max_abs > 1.0:
        audio_out = audio_out / max_abs * 0.999

    # 5) 저장 (stereo면 (T, C)로 transpose)
    if audio_out.ndim == 1:
        sf.write(out_path, audio_out, sr)
    else:
        sf.write(out_path, audio_out.T, sr)

    print(f"{os.path.basename(in_path)}: {cur_db:.2f} dB -> {target_db:.2f} dB (gain={gain:.3f})")

def batch_match_examples_folder(
    input_dir: str = "examples",
    output_dir: str = "examples_normalized",
    target_db: float = -20.0,
):
    """
    input_dir: 원본 wav 파일들이 있는 폴더 (예: 'examples')
    output_dir: 정규화된 wav를 저장할 폴더
    target_db: 맞추고 싶은 RMS dB
    """
    os.makedirs(output_dir, exist_ok=True)

    # 여기서는 .wav만 대상으로 함. 필요하면 패턴 추가 가능.
    wav_paths = glob(os.path.join(input_dir, "*.wav"))

    if not wav_paths:
        print("입력 폴더에 wav 파일이 없습니다.")
        return

    print(f"{input_dir} 안의 {len(wav_paths)}개 wav 파일을 target_db={target_db} dB로 맞춥니다.\n")

    for p in wav_paths:
        fname = os.path.basename(p)
        out_path = os.path.join(output_dir, fname)
        match_to_target_db(p, out_path, target_db=target_db)

if __name__ == "__main__":
    # 여기에서 target_db만 바꿔서 쓰면 됨 (예: -18.0, -20.0 등)
    batch_match_examples_folder(
        input_dir= r"C:\Users\ejh99\deleteplease\target",
        output_dir="data_normalized",
        target_db=-20.0,
    )
