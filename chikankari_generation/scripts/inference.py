
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os

# Load base model
base_model_id = "runwayml/stable-diffusion-v1-5"
lora_weights_path = "../models/lora-output"

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.load_lora_weights(lora_weights_path)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Prompt for generation
prompt = "a photo of sks chikankari embroidery"

# Generate and save 5 sample images
os.makedirs("../derived", exist_ok=True)
for i in range(5):
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(f"../derived/generated_sample_{i+1}.png")

print("✅ Inference complete: 5 images saved in derived/")
