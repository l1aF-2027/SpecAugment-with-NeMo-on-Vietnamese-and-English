# 🗣️ SpecAugment with NeMo on Vietnamese and English

## 📘 Project Overview

This project revisits and extends the work presented in:

> [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://www.isca-archive.org/interspeech_2019/park19e_interspeech.pdf)
> *Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, Quoc V. Le. Interspeech 2019*

Instead of using the LAS (Listen, Attend and Spell) model and **Shallow Fusion** as in the original paper, we use:

* **QuartzNet 15x5** from NVIDIA NeMo for both English and Vietnamese ASR
* **Deep Fusion** for language model integration

We trained from scratch on English data (LibriSpeech-100) and fine-tuned a pretrained English model for Vietnamese (VIVOS Corpus).

> ⚠ Due to limited computing resources (e.g., 12-hour limit on Kaggle P100), training was split across multiple sessions and may not reach optimal performance and the experiment with Deep Fussion can not do on English dataset.

## 🧪 Methods

### 🔁 SpecAugment

A simple but powerful data augmentation method applied directly to **Mel spectrograms**, including:

* **Time Masking**
* **Frequency Masking**

We did not use **Time Warping** due to its high computational cost and low effectiveness.

Configuration in NeMo:

```yaml
spec_augment:
  _target_: nemo.collections.asr.modules.SpectrogramAugmentation
  freq_masks: 1
  freq_width: 27
  time_masks: 1
  time_width: 50
```

### 🧠 Model Architecture

We used **QuartzNet 15x5**, a lightweight 1D convolutional end-to-end ASR model with CTC loss.

For Vietnamese, we fine-tuned the pretrained `stt_en_quartznet15x5` model from NeMo.


## 📊 Datasets

| Dataset         | Language   | Training Hours | Test Hours |
| --------------- | ---------- | -------------- | ---------- |
| LibriSpeech-100 | English    | 100            | 5.4        |
| VIVOS           | Vietnamese | \~15           | \~0.75     |

## 🧪 Experiments & Results

| Model              | No Fusion           |                    | With Fusion        |                    |
|--------------------|---------------------|--------------------|--------------------|--------------------|
|                    | LibreSpeech-100     | VIVOS              | LibreSpeech-100    | VIVOS              |
| QuartzNet          | 0.1833              | 0.3006             | -                  | 0.2739             |
| QuartzNet+SpecAug  | 0.1664              | 0.2723             | -                  | 0.2310             |


## 🔁 Fusion Techniques

We replaced **Shallow Fusion** with **Deep Fusion** to integrate the language model directly into training rather than only during decoding.

In this study, we propose a hybrid architecture that combines the strengths of the **QuartzNet encoder** and the **RNN-T decoder**. Traditionally, QuartzNet is paired with a CTC decoder, but we substitute the decoder with the **RNN-T architecture**, which includes a **prediction network** and a **joint network**.

- The **QuartzNet encoder** remains unchanged with **15 layers** of **separable convolutions**, which efficiently extract acoustic features.
- The **RNN-T decoder** consists of:
  - A **prediction network** with **1 layer and 640 hidden units**, functioning as an integrated language model.
  - A **joint network** with **640 hidden units**, which merges the outputs from the encoder and the prediction network to generate the final probability distribution.

This design enables **end-to-end joint training** instead of relying on shallow fusion at inference time. By deeply integrating the language model during training, the model leverages:
- The parallel processing ability of the CNN-based QuartzNet encoder,
- The powerful sequence modeling capabilities of the RNN-T decoder, and
- The **streaming inference** support inherent in the RNN-T framework.

This approach not only improves the alignment between acoustic and linguistic representations but also simplifies the deployment process by eliminating the need for external language model fusion during decoding.


## 🔍 Observations

* SpecAugment acts as a **strong regularizer**.
* Even with limited resources, it improved performance noticeably.
* QuartzNet is a viable LAS alternative with available pretrained weights and lower training cost.


## 📌 Conclusion

* ✅ SpecAugment significantly boosts generalization for ASR models.
* ✅ QuartzNet with SpecAugment achieves promising results on both English and Vietnamese datasets.
* ✅ Deep Fusion offers an alternative to Shallow Fusion by incorporating the language model during training and gains a higher performance on Vietnamese dataset.
* ⚠ Results are affected by limited compute time and multi-stage training.

## 📂 Structure

```
root/
├── manifests/
├── configs/
├── scripts/
├── checkpoints/
├── logs/
└── README.md
```

## 📚 References

1. Park, D. S., Chan, W., Zhang, Y., et al. (2019). [SpecAugment](https://www.isca-archive.org/interspeech_2019/park19e_interspeech.pdf). *Interspeech*.
2. NVIDIA NeMo Toolkit: [https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)
3. LibriSpeech Dataset: [http://www.openslr.org/12/](http://www.openslr.org/12/)
4. VIVOS Corpus: [https://ailab.hcmus.edu.vn/vivos](https://ailab.hcmus.edu.vn/vivos)

Let me know if you want a Vietnamese version of this `README.md` as well or want to add badges, setup commands, or a `requirements.txt` section.
