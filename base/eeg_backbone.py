"""
固定维度版本的 EEG Backbone
- 在初始化时就计算并创建完整网络（不延迟）
- 自动适配 EEG 和 MEG 数据集
- 不需要预初始化
- 训练稳定
"""

import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import os
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return  x + self.f(x)
    

class EEGProjectLayer(nn.Module):
    def __init__(self,  z_dim,c_num, timesteps, drop_proj=0.3):
        super(EEGProjectLayer, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1]-self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


def _calculate_conv_output_dim(c_num, timesteps):
    """
    使用实际测试来确定各个 backbone 的输出维度
    这比手动计算更可靠
    
    Args:
        c_num: 通道数
        timesteps: [start, end] 时间窗
    
    Returns:
        各个 backbone 的 embedding_dim
    """
    t_len = timesteps[1] - timesteps[0]
    
    # 使用已知的维度表（从实际测试得出）
    # EEG: c_num=17, t_len=250
    # MEG: c_num=271, t_len=201
    
    if c_num == 17 and t_len == 250:
        # EEG 配置
        return {
            'shallownet': 1440,
            'tsconv': 1440,
            'deepnet': 1400,
            'eegnet': 1248,
        }
    elif c_num == 271 and t_len == 201:
        # MEG 配置（从之前测试得出）
        return {
            'shallownet': 1040,
            'tsconv': 1040,
            'deepnet': 800,
            'eegnet': 864,
        }
    else:
        # 其他配置：动态计算（使用dummy forward）
        # 返回 None，在 BaseModel 中会触发动态计算
        return {
            'shallownet': None,
            'tsconv': None,
            'deepnet': None,
            'eegnet': None,
        }


class BaseModel(nn.Module):
    """
    固定维度版本的 BaseModel
    - 对于已知配置（EEG/MEG），在 __init__ 时就创建完整的 project 层
    - 对于未知配置，使用 lazy initialization（向后兼容）
    - 不需要预初始化
    """
    def __init__(self, z_dim, c_num, timesteps, backbone_type='shallownet'):
        super(BaseModel, self).__init__()
        
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.backbone_type = backbone_type.lower()
        
        # 计算该配置下的正确 embedding_dim
        dims = _calculate_conv_output_dim(c_num, timesteps)
        embedding_dim = dims.get(self.backbone_type, None)
        
        self.backbone = None  # 子类会设置
        
        if embedding_dim is not None:
            # 已知配置：立即创建 project 层
            self.project = nn.Sequential(
                FlattenHead(),
                nn.Linear(embedding_dim, z_dim),
                ResidualAdd(nn.Sequential(
                    nn.GELU(),
                    nn.Linear(z_dim, z_dim),
                    nn.Dropout(0.5))),
                nn.LayerNorm(z_dim))
            print(f"[{self.backbone_type.upper()}] c_num={c_num}, timesteps={timesteps}, embedding_dim={embedding_dim} ✅")
        else:
            # 未知配置：延迟创建（但会打印警告）
            self.project = None
            print(f"[{self.backbone_type.upper()}] c_num={c_num}, timesteps={timesteps}, ⚠️  Unknown config, will use lazy init")
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
    
    def _build_projector(self, in_features: int):
        """仅用于未知配置的延迟初始化"""
        self.project = nn.Sequential(
            FlattenHead(),
            nn.Linear(in_features, self.z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(self.z_dim, self.z_dim),
                nn.Dropout(0.5))),
            nn.LayerNorm(self.z_dim))
        # 确保在正确的设备上
        if self.backbone is not None:
            self.project.to(next(self.backbone.parameters()).device)

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        
        # 如果是未知配置，第一次 forward 时创建
        if self.project is None:
            with torch.no_grad():
                flat_dim = x.contiguous().view(x.size(0), -1).size(1)
            self._build_projector(flat_dim)
            print(f"[{self.backbone_type.upper()}] Lazy init with embedding_dim={flat_dim}")
        
        x = self.project(x)
        return x


class Shallownet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, backbone_type='shallownet')
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.Dropout(0.5),
            )
    

class Deepnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, backbone_type='deepnet')
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 25, (1, 10), (1, 1)),
                nn.Conv2d(25, 25, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(25),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(25, 50, (1, 10), (1, 1)),
                nn.BatchNorm2d(50),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(50, 100, (1, 10), (1, 1)),
                nn.BatchNorm2d(100),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(100, 200, (1, 10), (1, 1)),
                nn.BatchNorm2d(200),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
            )
        

class EEGnet(BaseModel):
    def __init__(self,  z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, backbone_type='eegnet')
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 8, (1, 64), (1, 1)),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
                nn.Conv2d(16, 16, (1, 16), (1, 1)),
                nn.BatchNorm2d(16), 
                nn.ELU(),
                nn.Dropout2d(0.5)
            )
        

class TSconv(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, backbone_type='tsconv')
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(0.5),
            )


if __name__ == "__main__":
    print("="*80)
    print("测试固定维度版本 - EEG 配置")
    print("="*80)
    
    backbones = {
        "Shallownet": Shallownet,
        "Deepnet": Deepnet,
        "EEGnet": EEGnet,
        "TSconv": TSconv,
    }
    
    # EEG 配置
    z_dim = 1024
    c_num_eeg = 17
    timesteps_eeg = [0, 250]
    
    print("\n📊 EEG 数据集 (c_num=17, timesteps=250):")
    print("-"*80)
    for name, BackboneClass in backbones.items():
        model = BackboneClass(z_dim=z_dim, c_num=c_num_eeg, timesteps=timesteps_eeg)
        total_params = sum(p.numel() for p in model.parameters())
        
        # 测试前向传播
        with torch.no_grad():
            x = torch.randn(2, c_num_eeg, timesteps_eeg[1]-timesteps_eeg[0])
            out = model(x)
        
        print(f"  {name:<15} 参数: {total_params:>10,}  输出: {out.shape}")
    
    # MEG 配置
    c_num_meg = 271
    timesteps_meg = [0, 201]
    
    print("\n📊 MEG 数据集 (c_num=271, timesteps=201):")
    print("-"*80)
    for name, BackboneClass in backbones.items():
        model = BackboneClass(z_dim=z_dim, c_num=c_num_meg, timesteps=timesteps_meg)
        total_params = sum(p.numel() for p in model.parameters())
        
        # 测试前向传播
        with torch.no_grad():
            x = torch.randn(2, c_num_meg, timesteps_meg[1]-timesteps_meg[0])
            out = model(x)
        
        print(f"  {name:<15} 参数: {total_params:>10,}  输出: {out.shape}")
    
    print("\n" + "="*80)
    print("✅ 固定维度版本测试通过！")
    print("   - 在初始化时就创建完整网络")
    print("   - 不需要预初始化")
    print("   - 同时支持 EEG 和 MEG")
    print("="*80)

