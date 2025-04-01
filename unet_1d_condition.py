# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_1d_blocks import get_down_block, get_mid_block, get_out_block, get_up_block
from diffusers.models.activations import get_activation


@dataclass
class UNet1DConditionOutput(BaseOutput):
    """
    The output of [`UNet1DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, sample_size)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None


class UNet1DConditionModel(ModelMixin, ConfigMixin):
    r"""
    A conditional 1D UNet model that takes a noisy sample, conditional state, and a timestep and returns a sample
    shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for its generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int`, *optional*, defaults to 65536): Default length of input sequence.
        in_channels (`int`, *optional*, defaults to 2): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 2): Number of channels in the output.
        extra_in_channels (`int`, *optional*, defaults to 0): Number of additional channels added to the input.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`): Whether to flip the sin to cos in the time embedding.
        freq_shift (`float`, *optional*, defaults to 0.0): The frequency shift to apply to the time embedding.
        time_embedding_type (`str`, *optional*, defaults to "fourier"): Type of time embedding to use.
        use_timestep_embedding (`bool`, *optional*, defaults to `False`): Whether to use learned timestep embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D")`):
            Tuple of downsample block types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock1D"`): Block type for middle of UNet.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip")`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(32, 32, 64)`): Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to 1): The number of layers per block.
        act_fn (`str`, *optional*, defaults to `None`): Optional activation function in UNet blocks.
        norm_num_groups (`int`, *optional*, defaults to 8): The number of groups for normalization.
        encoder_hid_dim (`int`, *optional*, defaults to 512): The dimension of the encoder hidden states.
        cross_attention_dim (`int`, *optional*, defaults to 512): The dimension of the cross attention features.
        downsample_each_block (`bool`, *optional*, defaults to `False`): Experimental feature for using UNet without upsampling.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: int = 65536,
        in_channels: int = 2,
        out_channels: int = 2,
        extra_in_channels: int = 0,
        flip_sin_to_cos: bool = True,
        freq_shift: float = 0.0,
        time_embedding_type: str = "fourier",
        down_block_types: Tuple[str] = ("DownBlock1D", "DownBlock1D", "DownBlock1D"),
        mid_block_type: str = "UNetMidBlock1D",
        up_block_types: Tuple[str] = ("UpBlock1D", "UpBlock1D", "UpBlock1D"),
        out_block_type: str = None,
        block_out_channels: Tuple[int] = (32, 32, 64),
        layers_per_block: int = 1,
        act_fn: str = None,
        norm_num_groups: int = 8, 
        encoder_hid_dim: int = 512,
        cross_attention_dim: int = 512,
        downsample_each_block: bool = False,
    ):
        super().__init__()
        self.sample_size = sample_size

        # Time embedding
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=8, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = 2 * 8
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=freq_shift
            )
            timestep_input_dim = block_out_channels[0]

        time_embed_dim = block_out_channels[0] * 4
        
        self.time_embedding = TimestepEmbedding(
            in_channels=timestep_input_dim,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=block_out_channels[0],
        )
        
        # Encoder projection - projects image features to required dimension
        self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        
        # Optional conditioning embedding
        self.cond_embedding = nn.Sequential(
            nn.Linear(cross_attention_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Build U-Net structure
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.out_block = None

        # Down blocks
        output_channel = in_channels
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            if i == 0:
                input_channel += extra_in_channels

            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_downsample=not is_final_block or downsample_each_block,
            )
            self.down_blocks.append(down_block)

        # Mid block
        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            embed_dim=block_out_channels[0],
            num_layers=layers_per_block,
            add_downsample=downsample_each_block,
        )

        # Up blocks
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        
        if out_block_type is None:
            final_upsample_channels = out_channels
        else:
            final_upsample_channels = block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = (
                reversed_block_out_channels[i + 1] if i < len(up_block_types) - 1 else final_upsample_channels
            )

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # Output block
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.out_block = get_out_block(
            out_block_type=out_block_type,
            num_groups_out=num_groups_out,
            embed_dim=block_out_channels[0],
            out_channels=out_channels,
            act_fn=act_fn,
            fc_dim=block_out_channels[-1] // 4,
        )

    def process_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Process the encoder hidden states (image features) through a projection layer.
        
        Args:
            encoder_hidden_states (`torch.Tensor`): Image features with shape `(batch_size, feature_dim)`.
            
        Returns:
            `torch.Tensor`: Processed encoder hidden states.
        """
        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        return encoder_hidden_states

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[UNet1DConditionOutput, Tuple]:
        r"""
        The [`UNet1DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the shape `(batch_size, num_channels, sample_size)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch_size, feature_dim)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~UNet1DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~UNet1DConditionOutput`] or `tuple`:
                If `return_dict` is True, a [`~UNet1DConditionOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        # 1. Time embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # Project time embeddings
        t_emb = self.time_proj(timesteps)
        temb = self.time_embedding(t_emb)
        
        # 2. Condition embedding
        # Process encoder hidden states (image features)
        encoder_hidden_states = self.process_encoder_hidden_states(encoder_hidden_states)
        
        # Create conditioning embedding and add to time embedding
        temb = torch.cat([
            temb, encoder_hidden_states
        ], axis=-1)
        temb = temb.unsqueeze(-1)  # 变为 [batch_size, time_embed_dim, 1]
        # 3. Down blocks
        down_block_res_samples = ()
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=temb)
            down_block_res_samples += res_samples

        # 4. Mid block
        sample = self.mid_block(sample, temb)

        # 5. Up blocks
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            sample = upsample_block(sample, res_hidden_states_tuple=res_samples, temb=temb)

        # 6. Output block
        if self.out_block:
            sample = self.out_block(sample, temb)

        if not return_dict:
            return (sample,)

        return UNet1DConditionOutput(sample=sample)
