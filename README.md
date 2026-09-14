# 🌍 EarthSwin

**EarthSwin** is a deep learning based landslide detection system for satellite imagery from **Himachal Pradesh**.

The project combines **multi-scale spatial features**, **spatial relationships**, and **temporal information** from consecutive weeks to identify potential landslide events.

---

## 🏔️ Overview

Landslides are influenced by both **where** they occur and **how environmental conditions change over time**. Looking at a single satellite image may not provide enough context.

EarthSwin addresses this by combining:

* 🧩 **Patch Pyramid** for multi-scale spatial information
* 🧠 **CNN + Swin Transformer** for feature extraction
* 🔗 **Cross-modal Attention** to combine CNN and Swin features
* 🕸️ **Graph Attention Network (GAT)** to model spatial relationships between nearby regions
* ⏳ **Temporal Transformer** to capture changes across consecutive weeks
* 🎯 **Classification Head** for landslide prediction

---

## 🧠 Architecture

The overall pipeline is:

```text
        🛰️ Satellite / Environmental Data
                       ↓
              📐 Multi-scale Patches
             64 × 64 | 128 × 128 | 256 × 256
                       ↓
                CNN + Swin Encoder
                       ↓
              🔗 Cross-modal Attention
                       ↓
                  Scale Fusion
                       ↓
                 🕸️ Spatial Graph
                       ↓
                      GAT
                       ↓
              ⏳ Temporal Transformer
                       ↓
              🎯 Classification Head
                       ↓
             📊 Landslide Probability
```

---

## 🌱 Input Data

The model uses three environmental variables:

* 🛰️ **Sentinel-1 VV backscatter**
* 🌧️ **CHIRPS rainfall**
* ⛰️ **SRTM slope**

Missing-value masks are also provided to the model.

This gives **6 channels per week**:

```text
3 environmental channels
        +
3 missingness masks
        =
6 channels / week
```

The model processes a sequence of **3 consecutive weeks** for each prediction.

---

## 🕸️ Spatial Graph

Spatial relationships are represented using a graph where each spatial anchor is connected to its **8 nearest neighbours**.

**Euclidean distance** is used to select neighbouring anchors, while edge weights incorporate environmental similarity based on:

* 📡 **VV backscatter**
* 🌧️ **Rainfall**
* ⛰️ **Slope**

The graph is constructed **separately for the training, validation, and test splits** to avoid spatial information leaking across dataset splits.

---

## 📦 Dataset

The current dataset is based on **weekly observations from 2016 in Himachal Pradesh**.

The preprocessing pipeline includes:

* 🗺️ Raster merging
* 🧩 Multi-scale patch extraction
* 🚫 Missing-value handling
* 📏 Normalization
* ⏳ Temporal sequence construction
* ✂️ Spatial train/validation/test splitting
* 🔄 Positive-sample augmentation
* 🕸️ Split-specific graph construction

The processed dataset is **not included in this repository** because of its size. The required data is shared separately.

---

## 📁 Project Structure

```text
EarthSwin/
│
├── earthswin/
│   └── Swin-Transformer/      # 🧠 Swin Transformer implementation
│
├── models/
│   ├── attention.py           # 🔗 Cross-modal attention
│   ├── cnn.py                 # 🧠 CNN feature extractor
│   ├── fusion.py              # 🔀 Multi-scale feature fusion
│   ├── gat.py                 # 🕸️ Graph Attention Network
│   ├── graph.py               # 📍 Graph construction
│   ├── patch_pyramid.py       # 📐 Multi-scale feature extraction
│   ├── swin.py                # 🧠 Swin feature extractor
│   ├── temporal.py            # ⏳ Temporal Transformer
│   └── earthswin_model.py     # 🌍 Complete EarthSwin model
│
├── preprocessing/
│   ├── create_pytorch_dataset.py
│   └── ...
│
├── config.py
├── diagnose.py
├── merge.py
├── train.py
├── environment.yml
└── README.md
```

---

## ⚙️ Installation

Create the Conda environment using:

```bash
conda env create -f environment.yml
conda activate landslide
```

The exact **PyTorch installation** may depend on whether the system uses CPU or CUDA.

---

## 🚀 Training

The training pipeline currently uses:

* **BCEWithLogitsLoss**
* **AdamW optimizer**
* **Gradient clipping**
* **Validation loss** for model selection

Because landslide events are highly imbalanced, the training supervision uses **balanced positive and negative samples**, while validation and test sets retain their natural distribution.

---

## ✅ Current Status

The main **EarthSwin architecture has been implemented and tested component-by-component**.

### Implemented & Tested

* ✅ Multi-scale Patch Pyramid
* ✅ CNN feature extraction
* ✅ Swin Transformer feature extraction
* ✅ Cross-modal attention
* ✅ Scale fusion
* ✅ Spatial graph construction
* ✅ Graph Attention Network
* ✅ Temporal Transformer
* ✅ End-to-end forward pass
* ✅ End-to-end backward / gradient flow

### 🔬 In Progress

**Model training and final evaluation** are currently in progress.

---

## 📜 Third-party Code

This project uses the official **Microsoft Swin Transformer** implementation as part of the model backbone.

The original Swin Transformer implementation is licensed under the **MIT License**.

**Copyright © Microsoft Corporation.**

The original license and copyright notices are retained with the corresponding third-party code.

---

## 👩‍💻 Author

**Advitiya Prakash**


---

⭐ *EarthSwin — combining spatial, temporal, and multi-scale information for landslide detection.*
