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

## 📦 Project Structure

