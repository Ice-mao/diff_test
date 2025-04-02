from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class TrainingConfig:
    dataset_name = "/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/trajs_dam/"
    # train parameters
    train_batch_size = 16
    eval_batch_size = 8  # how many images to sample during evaluation
    num_epochs = 2
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 1
    # distribution config
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    gpu_ids = None
    # model save config
    output_dir = "auv_tracking_diffusion_policy"  # the model name locally and on the HF Hub
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = None
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 42


config = TrainingConfig()
import datasets
from auv_track_launcher.dataset.holoocean_image_dataset import HoloOceanImageDataset

# config.dataset_name = "/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/trajs_dam/"
dataset_0 = datasets.load_from_disk(config.dataset_name+"traj_1")
dataset_1 = datasets.load_from_disk(config.dataset_name+"traj_2")
dataset_2 = datasets.load_from_disk(config.dataset_name+"traj_3")
# dataset_3 = datasets.load_from_disk(config.dataset_name+"traj_2")
_dataset = datasets.concatenate_datasets([dataset_0, dataset_1, dataset_2])
dataset = HoloOceanImageDataset(_dataset,
                                horizon=16,
                                obs_horizon=4,
                                pred_horizon=12)
del _dataset

import torch

train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.train_batch_size,
    num_workers=2,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

from unet_1d_condition_dev import UNet1DConditionModel
from torch import nn
from auv_track_launcher.networks.diffusion_vision_encoder import Encoder

vision_encoder = Encoder(num_channels=[512, 256])
model = UNet1DConditionModel(
    input_dim=3,
    local_cond_dim=None,
    global_cond_dim=1024,
    cross_attention_dim=512,
    time_embedding_type="fourier",
    flip_sin_to_cos= True,
    freq_shift= 0,
    block_out_channels=[256,512,1024],
    kernel_size=3,
    n_groups=8,
    act_fn="mish",
    cond_predict_scale=False,
)
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'model': model
})

sample_image = dataset[0]["obs"].unsqueeze(0).float() 
image_features = nets['vision_encoder'](sample_image) # (4, 3, 128, 128) -> [4, 2048]
sample_action = dataset[0]["action"].unsqueeze(0).float() 
print("Input image shape:", sample_image.shape)
print("Input shape:", sample_action.shape)

print("Output shape:", model(sample_action, timestep=torch.tensor([10]), encoder_hidden_states=image_features).sample.shape)
print("dataset ready")

import torch
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_action.shape)
timesteps = torch.LongTensor([50])
noisy_action = noise_scheduler.add_noise(sample_action, noise, timesteps)

import torch.nn.functional as F

noise_pred = model(noisy_action, timesteps, encoder_hidden_states=image_features).sample
loss = F.mse_loss(noise_pred, noise)

from diffusers.optimization import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(nets.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

from diffusers import DDPMPipeline
import os

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    # images = pipeline(
    #     batch_size=config.eval_batch_size,
    #     generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    # ).images

    # # Save the images
    # test_dir = os.path.join(config.output_dir, "samples")
    # os.makedirs(test_dir, exist_ok=True)
    print("eval")
print("evaluate function ready")

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
        device_placement=True,
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    from diffusers.training_utils import EMAModel
    ema_model = EMAModel(
        parameters=accelerator.unwrap_model(model),
        power=0.75,
    )
    ema_model.to(accelerator.device)
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_actions = batch["action"].float().to(accelerator.device)
            sample_image = batch["obs"].float().to(accelerator.device)
            # Sample noise to add to the images
            noise = torch.randn(clean_actions.shape, device=clean_actions.device)
            bs = clean_actions.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_actions.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_actions = noise_scheduler.add_noise(clean_actions, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_dict = accelerator.unwrap_model(model)
                image_features = model_dict['vision_encoder'](sample_image)
                noise_pred = model_dict['model'](noisy_actions, timesteps, encoder_hidden_states=image_features).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                ema_model.step(model.parameters())
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            unet = accelerator.unwrap_model(model)
            ema_model.store(unet.parameters())
            ema_model.copy_to(unet.parameters())
            ema_model.restore(unet.parameters())
            pipeline = DDPMPipeline(unet=unet["model"], scheduler=noise_scheduler)
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)
                torch.save(unet.state_dict(), os.path.join(config.output_dir, "unet_ema"))


from accelerate import notebook_launcher
args = (config, nets, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
notebook_launcher (train_loop, args, num_processes=1)