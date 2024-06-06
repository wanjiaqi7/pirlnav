# 这段代码定义了几个神经网络组件，用于视觉特征提取，特别是针对Habitat环境
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from habitat import logger
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import RunningMeanAndVar
from torch import Tensor

from pirlnav.policy.models import resnet

#  这个类将输入张量展平，从第一个维度（即批量维度）之后的所有维度展平为一个维度
class Flatten(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.flatten(x, start_dim=1)

# 定义了一个名为 ResNetEncoder 的类，它继承自 nn.Module，用于在基于观察空间的强化学习环境中实现图像编码器，
# ResNetEncoder 类主要功能是通过骨干网络处理多种输入数据，并通过压缩层将高维特征压缩成更紧凑的表示
# 这个编码器主要基于 ResNet 架构，并且可以处理多种类型的输入（如 RGB 图像、深度图像和语义图像）
class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,  # 输入的观察空间，通常包含不同类型的图像数据（如 RGB、深度、语义）
        baseplanes: int = 32,  # 基础平面数，用于控制网络的通道数
        ngroups: int = 32,  # 用于 GroupNorm 的分组数量
        spatial_size: int = 128,  # 初始的空间尺寸
        make_backbone=None,  # 创建骨干网络的函数
        normalize_visual_inputs: bool = False,  # 是否对视觉输入进行归一化
        sem_embedding_size=4,  # 语义嵌入的大小
        dropout_prob: float = 0.0  # Dropout 的概率，用于正则化
    ):
        super().__init__()
        #  处理 RGB 输入：获取 RGB 图像的尺寸和通道数，如果没有 RGB 图像，则设置通道数为 0
        if "rgb" in observation_space.spaces:
            self._frame_size = tuple(observation_space.spaces["rgb"].shape[:2])
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            # spatial_size = observation_space.spaces["rgb"].shape[:2] // 2
            spatial_size = observation_space.spaces["rgb"].shape[:2]
        else:
            self._n_input_rgb = 0
        #  处理深度输入：获取深度图像的尺寸和通道数，如果没有深度图像，则设置通道数为 0
        if "depth" in observation_space.spaces:
            self._frame_size = tuple(observation_space.spaces["depth"].shape[:2])
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            # spatial_size = observation_space.spaces["depth"].shape[:2] // 2
            spatial_size = observation_space.spaces["depth"].shape[:2]
        else:
            self._n_input_depth = 0
        #  处理语义输入：获取语义图像的尺寸，并设置语义嵌入的大小，如果没有语义图像，则设置通道数为 0
        if "semantic" in observation_space.spaces:
            self._frame_size = tuple(observation_space.spaces["semantic"].shape[:2])
            self._n_input_semantics = sem_embedding_size # observation_space.spaces["semantic"].shape[2]
        else:
            self._n_input_semantics = 0
        #  调整空间尺寸：根据输入图像的尺寸调整 spatial_size
        if self._frame_size == (256, 256):
            spatial_size = (128, 128)
        elif self._frame_size == (240, 320):
            spatial_size = (120, 108)
        elif self._frame_size == (480, 640):
            spatial_size = (120, 108)
        elif self._frame_size == (640, 480):
            spatial_size = (108, 120)
        #  初始化归一化模块：如果需要归一化视觉输入，则使用 RunningMeanAndVar 模块，否则使用空的 nn.Sequential() 模块
        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()
        #  is_blind 方法用于检查模型是否接收任何视觉输入
        #  创建骨干网络：
        if not self.is_blind:   #  如果模型不是盲目的（即有视觉输入），则根据输入通道数创建骨干网络
            input_channels = self._n_input_depth + self._n_input_rgb + self._n_input_semantics  # input_channels 是所有输入（RGB、深度和语义）通道数的总和
            self.backbone = make_backbone(input_channels, baseplanes, ngroups, dropout_prob=dropout_prob)  # 使用 make_backbone 函数创建，这个函数定义了实际的 ResNet 架构
            # 压缩空间计算
            # 计算最终的空间尺寸 final_spatial ：通过乘以 self.backbone.final_spatial_compress（骨干网络的空间压缩因子）来计算
            final_spatial = np.array([math.ceil(
                d * self.backbone.final_spatial_compress
            ) for d in spatial_size])
            after_compression_flat_size = 2048  # 压缩后的平面大小，设置为 2048
            num_compression_channels = int(
                round(after_compression_flat_size / np.prod(final_spatial))
            )  # 是压缩通道数，通过压缩后的平面大小除以压缩空间的大小计算得到
            #  压缩层
            self.compression = nn.Sequential(
                nn.Conv2d(                             #  一个 Conv2d 层，调整通道数
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),   # 一个 GroupNorm 层，进行分组归一化
                nn.ReLU(True),    
            )

            self.output_shape = (              # 是压缩后特征图的形状，包括通道数和空间大小
                num_compression_channels,
                final_spatial[0],
                final_spatial[1],
            )

    @property
    def is_blind(self):   # 用于判断是否没有任何输入通道，如果 RGB、深度和语义通道数总和为 0，则认为模型没有输入（即“盲”状态）
        return self._n_input_rgb + self._n_input_depth + self._n_input_semantics == 0
    # 网络层初始化
    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:  # 检查是否没有输入
            return None
        # 处理 RGB 输入
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)
        # 处理深度输入
        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)
        # 处理语义输入
        if self._n_input_semantics > 0:
            semantic_observations = observations["semantic"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            semantic_observations = semantic_observations.permute(0, 3, 1, 2)

            cnn_input.append(semantic_observations)

        x = torch.cat(cnn_input, dim=1)
        if self._frame_size == (256, 256):
            x = F.avg_pool2d(x, 2)
        elif self._frame_size == (240, 320):
            x = F.avg_pool2d(x, (2, 3), padding=(0, 1)) # 240 x 324 -> 120 x 108
        elif self._frame_size == (480, 640):
            x = F.avg_pool2d(x, (4, 5))
        elif self._frame_size == (640, 480):
            x = F.avg_pool2d(x, (5, 4))
        # 归一化和特征提取
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x

# 是一个基于 ResNet 的深度图编码器。它可以加载预训练的权重，并根据需求输出紧凑的特征向量或空间特征图
class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=128,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"depth": observation_space.spaces["depth"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint)

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        obs_depth = observations["depth"]
        if len(obs_depth.size()) == 5:
            observations["depth"] = obs_depth.contiguous().view(
                -1, obs_depth.size(2), obs_depth.size(3), obs_depth.size(4)
            )

        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)
# 一个基于 ResNet 的 RGB 图像编码器。它可以处理 RGB 图像，提取特征，并根据需求输出紧凑的特征向量或空间特征图
class ResnetRGBEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=256,
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()

        backbone_split = backbone.split("_")
        logger.info("backbone: {}".format(backbone_split))
        make_backbone = getattr(resnet, backbone_split[0])

        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"rgb": observation_space.spaces["rgb"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=make_backbone,
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)
    
    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        obs_rgb = observations["rgb"]
        if len(obs_rgb.size()) == 5:
            observations["rgb"] = obs_rgb.contiguous().view(
                -1, obs_rgb.size(2), obs_rgb.size(3), obs_rgb.size(4)
            )

        if "rgb_features" in observations:
            x = observations["rgb_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)
