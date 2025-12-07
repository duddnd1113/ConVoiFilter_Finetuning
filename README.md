# Convoifilter Fine-Tuning Project

이 Repository는 **ConVoiFilter** 모델을 기반으로 한 **목표 화자 음성 추출(Target Speaker Voice Extraction)** 파인튜닝 프로젝트입니다.  
원본 모델 및 방법론은 아래 논문을 참고합니다

📄 **"ConVoiFilter: An End-to-End Target Speaker Voice Filtering Model"**  
🔗 https://arxiv.org/pdf/2308.11380.pdf  

본 프로젝트의 목표는 **ConVoiFilter 모델을 실제 환경에 더 적합하도록 파인튜닝하고**,  
복잡한 소음 속에서도 목표 화자를 안정적으로 분리할 수 있도록 모델을 개선하는 것입니다. 

---

## 프로젝트 배경

이 프로젝트는 연세대학교

**딥러닝과 응용 (IIE4123.01-00)**  
수업의 팀 프로젝트로 진행되었습니다.
 
목표는 실제 환경(소음 포함)에서 목표 화자를 더 안정적으로 분리하기 위한 파인튜닝 실험입니다.

---

## 주요 목표
- 목표 화자 음성 분리 성능 향상  
- 복잡한 배경 소음 상황에서 강건성 증가  
- 실제 환경 음색에 맞춘 Fine-Tuning  

---

# 사용 방법 (Usage)

## 1. Pretrained Weight 다운로드

본 프로젝트는 HuggingFace에 공개된 ConVoiFilter 사전학습 가중치를 기반으로 합니다

🔗 https://huggingface.co/nguyenvulebinh/voice-filter  

다운로드 후 아래 폴더 구조로 배치해야합니다. 
```
pretrained/
└── pytorch_model.bin
```
---


## 2. Dataset 구조

아래 폴더 구조를 따라야 합니다
```
Dataset/
├── Train_Dataset/
│ ├── Clean/
│ ├── Mix/
│ └── Target/
├── Val_Dataset/
│ ├── Clean/
│ ├── Mix/
│ └── Target/
└── Test_Dataset/
├── Clean/
├── Mix/
└── Target/
```

각 폴더 내부는 **동일한 index 이름을 사용**해야 합니다.

Ex)

Clean/clean_000001.wav

Mix/mix_000001.wav

Target/enrollment_000001.wav


---

## ⚠️ 중요한 Dataset Split 규칙

**같은 화자가 Train/Test에 동시에 포함되면 안 됩니다.**
즉, 반드시 “화자 단위 분리”가 필요합니다.  

ConVoiFilter는 Target Speaker conditioning 구조이기 때문에  
같은 화자가 Train/Test에 포함되면 테스트 성능이 과대평가됩니다.

---

## 3. Fine-Tuning 실행 방법 및 전략 설명

본 프로젝트에서는 아래 argument들을 조합하여 다양한 파인튜닝 설정을 적용할 수 있습니다

```bash
--depth:          Conformer block 몇 개를 업데이트할지 (0=freeze, 1=하나 열기 …)
--type:           'full' = 전체 업데이트, 'attn' = attention 계열만 업데이트
--open_spk_ffn:   Speaker FFN 업데이트 여부 (1=업데이트)
--open_pre_ffn:   Conformer 이전 FFN/Conv 업데이트 여부 (1=업데이트)
```

예시 실행
```bash
python Finetuning3.py --depth 2 --type attn --open_spk_ffn 1
```
각 옵션은 모델의 특정 모듈만 선택적으로 업데이트할 수 있어 다양한 실험 전략을 손쉽게 구성하도록 합니다.

---
## 4. Hyperparameter 설정

코드 내부에서 아래 요소를 직접 설정할 수 있습니다

- batch size
- learning rate
- optimizer
- scheduler (예: cosine learning rate scheduler)
- early stopping

사용자도 argument 또는 코드 값만 바꿔 다양한 파인튜닝 실험을 직접 재현할 수 있습니다.

