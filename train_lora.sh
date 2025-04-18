#!/bin/bash

accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="./chikankari_final_dataset" \
  --output_dir="./outputs/chikankari_lora" \
  --instance_prompt="a detailed Chikankari embroidery design" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=10 \
  --checkpointing_steps=500 \
  --mixed_precision="fp16" \
  --seed=1337 \
  --report_to="tensorboard"

