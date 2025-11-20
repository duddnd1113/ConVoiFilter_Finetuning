# VoiceFiltering Fine-Tuning Project

This repository contains our implementation and fine-tuning pipeline for the **ConVoiFilter** model, originally proposed for target-speaker voice filtering.  
The base model and methodology reference the official work from:  
**"ConVoiFilter: An End-to-End Target Speaker Voice Filtering Model"**  
🔗 https://arxiv.org/pdf/2308.11380.pdf

Our project adapts the original model to our own domain-specific environment and data.

---

## 🎓 Project Context

This work is conducted as part of the course:

**Deep Learning and Applications (IIE4123.01-00)**  
Yonsei University  

The objective of our team project is to **fine-tune ConVoiFilter to better match our specific target domain**, improving performance under realistic acoustic conditions while maintaining real-time feasibility.

We use the publicly shared ConVoiFilter pretrained model as a baseline and extend it through additional domain-adapted fine-tuning.

---

## 🚀 Goals of This Project

- Fine-tune ConVoiFilter for real-world target speaker extraction  
- Improve robustness to complex background noise  
- Adapt the model to our domain-specific speech characteristics  
- Optimize the model for real-time usage (low latency & lightweight)  
- Provide easy-to-use inference scripts for evaluation  

---


---

# 🇰🇷 VoiceFiltering Fine-Tuning Project (Korean Version)

이 저장소는 **ConVoiFilter** 모델을 기반으로 한 **목표 화자 음성 필터링(Target Speaker Voice Filtering)** 파인튜닝 프로젝트입니다.  
원본 모델 및 방법론은 아래 논문을 참고합니다:  
📄 **"ConVoiFilter: An End-to-End Target Speaker Voice Filtering Model"**  
🔗 https://arxiv.org/pdf/2308.11380.pdf

본 프로젝트는 원본 모델을 **우리 도메인에 맞춘 환경 및 데이터셋**으로 재학습하여 성능을 향상시키는 것을 목표로 합니다.

---

## 🎓 프로젝트 배경

이 프로젝트는 연세대학교:

**딥러닝과 응용 (IIE4123.01-00)**  
수업의 팀 프로젝트로 진행되었습니다.

우리 팀의 목표는 **ConVoiFilter 모델을 실제 환경에 더 적합하도록 파인튜닝하고**,  
복잡한 소음 속에서도 목표 화자를 안정적으로 분리할 수 있도록 모델을 개선하는 것입니다.

또한 실시간 사용 가능성과 경량화를 고려하여 모델을 재구성하고,  
추론 및 실험을 위한 편리한 스크립트도 제공합니다.

---

## 🚀 프로젝트 목표

- 실제 환경에서 목표 화자 음성 추출 성능 개선  
- 복잡하고 다양한 배경 소음 상황에서 모델 강건성 향상  
- 도메인 특화 음색 및 데이터에 맞춘 파인튜닝  
- 실시간 적용이 가능하도록 모델 경량화  
- 평가 및 추론을 위한 간단한 스크립트 제공  

---


