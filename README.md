# Chikankari Design Generator using Generative AI  
_Capstone Project – Generative AI for Unique cultural heitage_

This is an end to end Gen AI project that uses **Stable Diffusion v1.5** fine-tuned with **Advanced LoRA** to generate beautiful **Chikankari embroidery patterns** from text prompts like _"floral motifs with paisley"_. It includes a visually appealing **Streamlit interface** and advanced evaluation metrics (FID, LPIPS, CLIPScore, t-SNE).

---
## Features

- Text-to-image generation for Chikankari embroidery
- Fine-tuned with LoRA on 4000+ embroidery images
- Streamlit-based interactive UI 
- Evaluation: FID, CLIPScore, LPIPS, t-SNE visualization
- Latent space interpolation
- Dataset preprocessing & augmentation included

---  
## Highlights

| Pillar | Details |
| ------ | ------- |
| **Model** | Stable-Diffusion v1.5 **fine-tuned with Advanced LoRA** on **4000 +** curated Chikankari images |
| **Training** | Hugging Face Diffusers workflow – data aug, checkpointing, **Advanced LoRA** |

---

##  Repository Structure
```text
├─ app.py                     # Streamlit interface
├─ script.ipynb               # Pre processing and Data Augmentation
├─ mainmodel.ipynb            # Fine tuning
├─ data/
│  ├─ rawData/                # original reference photos
│  ├─ preprocessedData/       # colour-balanced 512² crops
│  ├─ postprocessedData/      # diffusion outputs cleaned 
│  ├─ lpips_comparison/       # LPIPS CSV + preview grids
├─ finalDerivedData/          # curated generated designs (PNG)
├─ requirements.txt
```
## Requirements

### Hardware

> **GPU Required**  

### Software

- Python 3.8+
- PyTorch with CUDA
- diffusers, transformers, accelerate, safetensors
- Streamlit
  
##  Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/INFO-698-InfoSci-Capstone/cultural-heritage-ai.git
```
### 2. Set up the environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 3.Download base model & LoRA weights
Base model: CompVis/stable-diffusion-v1-5 from HuggingFace

Place your LoRA weights at:
./outputs/chikankari_lora/pytorch_lora_weights.safetensors

## Running the App
Launch the Streamlit interface:
```bash

streamlit run app.py
```
Then open your browser to:
```bash
http://localhost:8501
```
> ⚠️ **GPU REQUIRED**  
> This project uses **Stable Diffusion v1.5 + LoRA**, which **requires a CUDA-compatible GPU**. CPU-only systems will likely fail or be extremely slow.
