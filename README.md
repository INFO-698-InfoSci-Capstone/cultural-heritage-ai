# Chikankari Design Generator using Generative AI  
_Capstone Project – Generative AI for Unique cultural heitage_

A full pipeline that **learns the aesthetics of hand-stitched Chikankari embroidery** and lets anyone create new patterns from plain-language prompts through an elegant Streamlit web app.

---

## Highlights

| Pillar | Details |
| ------ | ------- |
| **Model** | Stable-Diffusion v1.5 **fine-tuned with Advanced LoRA** on **4 000 +** curated Chikankari images |
| **Training** | Hugging Face Diffusers workflow – data aug, checkpointing, **Advanced LoRA** |
| **Eval** | *FID* + *CLIPScore* on held-out set, **LPIPS** vs. originals |
| **App** | Streamlit UI with smart prompt helper & stylish card layout |


---

##  Repository Structure
```text
├─ app.py                     # Streamlit interface
├─ script.ipynb               # Pre processing and Data Augmentation
├─ mainmodel.ipynb            # Fine tuning
├─ data/
│  ├─ rawData/                # original reference photos
│  ├─ preprocessedData/       # colour-balanced 512² crops
│  ├─ postprocessedData/      # diffusion outputs cleaned w/ ESRGAN
│  ├─ lpips_comparison/       # LPIPS CSV + preview grids
├─ finalDerivedData/       # curated generated designs (PNG)
├─ requirements.txt
