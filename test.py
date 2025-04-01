from unet_1d_condition_dev import UNet1DConditionModel
import torch
from tqdm import tqdm

# 创建模型
# model = UNet1DConditionModel(
#     sample_size=16,           # 动作序列长度
#     in_channels=3,              # 动作维度
#     out_channels=3,             # 输出维度
#     encoder_hid_dim=1024,        # 图像编码器输出维度
#     cross_attention_dim=512,     # 内部交叉注意力维度
#     act_fn="mish",              # 激活函数
#     # norm_num_groups=3,
#     layers_per_block=2,  # how many ResNet layers to use per UNet block
#     block_out_channels=(128, 256, 512),  # the number of output channels for each UNet block
#     down_block_types=(
#         "DownResnetBlock1D",  # a regular ResNet downsampling block
#         "DownResnetBlock1D",
#         "DownResnetBlock1D",
#     ),
#     up_block_types=(
#         "UpResnetBlock1D",  # a regular ResNet upsampling block
#         "UpResnetBlock1D",
#         "UpResnetBlock1D",
#     ),
# )

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

# 前向传播
action_sequence = torch.randn(1, 12, 3)          # (batch_size, seq_len, action_dim)
timesteps = torch.tensor([10, 10])                 # 随机 timestep
image_features = torch.randn(1, 1024)               # 图像特征
for t in tqdm(timesteps):
    output = model(action_sequence, t, image_features)
    print(output.sample.shape)