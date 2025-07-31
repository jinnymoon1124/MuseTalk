import torch
import torch.nn as nn
import math
import json

from diffusers import UNet2DConditionModel
import sys
import time
import numpy as np
import os

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=384, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, seq_len, d_model = x.size()
        pe = self.pe[:, :seq_len, :]
        x = x + pe.to(x.device)
        return x
    
class UNet():
    def __init__(self, 
                 unet_config,
                 model_path,
                 use_float16=False,
                 device=None
        ):
        with open(unet_config, 'r') as f:
            unet_config = json.load(f)
        self.model = UNet2DConditionModel(**unet_config)
        self.pe = PositionalEncoding(d_model=384)
        # 워커 프로세스인지 확인하여 올바른 디바이스 사용 (다중 환경변수 체크)
        import os
        worker_gpu_id = os.environ.get('WORKER_GPU_ID') or os.environ.get('FORCE_WORKER_GPU')
        
        print(f"🔍 [UNet] 환경 변수 체크:")
        print(f"   - WORKER_GPU_ID: {os.environ.get('WORKER_GPU_ID')}")
        print(f"   - FORCE_WORKER_GPU: {os.environ.get('FORCE_WORKER_GPU')}")
        print(f"   - 입력 device: {device}")
        
        if device != None:
            # 워커 프로세스 여부 확인
            is_worker = os.environ.get('MUSETALK_WORKER_PROCESS') == 'TRUE'
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            # 워커 프로세스에서 디바이스 조정
            if is_worker and worker_gpu_id is not None:
                # CUDA_VISIBLE_DEVICES로 인해 항상 cuda:0 사용
                worker_device = torch.device("cuda:0")
                print(f"🔥 [UNet] 워커 프로세스 디바이스: {worker_device} (실제 물리 GPU: {worker_gpu_id})")
                print(f"🔥 [UNet] CUDA_VISIBLE_DEVICES={cuda_visible}로 매핑됨")
                self.device = worker_device
            else:
                self.device = device
                print(f"🔧 [UNet] 디바이스 설정: {self.device}")
            
            # 멀티 GPU 환경에서 명시적 디바이스 설정
            if self.device.type == 'cuda':
                torch.cuda.set_device(self.device)
                print(f"🔧 [UNet] 디바이스 강제 설정: {self.device}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        weights = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(weights)
        if use_float16:
            self.model = self.model.half()
        
        # 모델을 지정된 디바이스로 이동 (재확인)
        print(f"🔧 [UNet] 모델을 {self.device}로 이동 중...")
        self.model.to(self.device)
    
if __name__ == "__main__":
    unet = UNet()
