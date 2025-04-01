from diffusers import DDPMPipeline
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.utils import make_image_grid
import os
import torch
from dataclasses import dataclass

# @dataclass
# class TrainingConfig:
#     image_size = 128  # the generated image resolution
#     train_batch_size = 8
#     eval_batch_size = 8  # how many images to sample during evaluation
#     num_epochs = 50
#     gradient_accumulation_steps = 1
#     learning_rate = 1e-4
#     lr_warmup_steps = 500
#     save_image_epochs = 10
#     save_model_epochs = 30
#     mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
#     output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub

#     push_to_hub = False  # whether to upload the saved model to the HF Hub
#     hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
#     hub_private_repo = None
#     overwrite_output_dir = True  # overwrite the old model when re-running the notebook
#     seed = 0


# config = TrainingConfig()

# def evaluate(config, pipeline):
#     # Sample some images from random noise (this is the backward diffusion process).
#     # The default pipeline output type is `List[PIL.Image]`
#     images = pipeline(
#         batch_size=16,
#         generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
#     ).images

#     # Make a grid out of the images
#     image_grid = make_image_grid(images, rows=4, cols=config.eval_batch_size // 4)

#     # Save the images
#     test_dir = os.path.join(config.output_dir, "samples")
#     os.makedirs(test_dir, exist_ok=True)

model_path = os.path.expanduser("/home/dell-t3660tow/data/manipulation/diff_test/ddpm-butterflies-128/")
unet = UNet2DModel.from_pretrained(model_path+'unet', use_safetensors=True)
scheduler = DDPMScheduler.from_pretrained(model_path+'scheduler')
pipeline = DDPMPipeline(unet, scheduler)
image = pipeline(num_inference_steps=100).images[0]
image.save("ddpm_generated_image.png")
# evaluate(config, pipeline)