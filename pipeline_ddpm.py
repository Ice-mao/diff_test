from diffusers import DiffusionPipeline
from diffusers import DDPMScheduler, UNet2DModel
import os
import torch
from PIL import Image
import numpy as np

model_path = os.path.expanduser("~/data/models/ddpm-cat-256/")

# ddpm = DiffusionPipeline.from_pretrained(model_path, use_safetensors=True).to("cuda")
# image = ddpm(num_inference_steps=25).images[0]
# image.save("ddpm_generated_image.png")

scheduler = DDPMScheduler.from_pretrained(model_path)
model = UNet2DModel.from_pretrained(model_path, use_safetensors=True).to("cuda")
scheduler.set_timesteps(200)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
input = noise

for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
    previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
    input = previous_noisy_sample


image = (input / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
image = Image.fromarray(image)
image.save("ddpm_generated_image.png")



