# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the given model."""

import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

warnings.filterwarnings('ignore', '`resume_download` is deprecated')

#----------------------------------------------------------------------------
# Configuration presets.

model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'

config_presets = {
    'edm2-cifar10-fid':        dnnlib.EasyDict(net=f'{model_root}/network-snapshot-0125829-0.100.pkl'),  # fid = 3.53
    'edm2-img512-xs-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.135.pkl'),  # fid = 3.53
    'edm2-img512-s-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.130.pkl'),   # fid = 2.56
    'edm2-img512-m-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.100.pkl'),   # fid = 2.25
    'edm2-img512-l-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.085.pkl'),   # fid = 2.06
    'edm2-img512-xl-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.085.pkl'),  # fid = 1.96
    'edm2-img512-xxl-fid':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.070.pkl'), # fid = 1.91
    'edm2-img64-s-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.075.pkl'),    # fid = 1.58
    'edm2-img64-m-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-m-2147483-0.060.pkl'),    # fid = 1.43
    'edm2-img64-l-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-l-1073741-0.040.pkl'),    # fid = 1.33
    'edm2-img64-xl-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img64-xl-0671088-0.040.pkl'),   # fid = 1.33
    'edm2-img512-xs-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.200.pkl'),  # fd_dinov2 = 103.39
    'edm2-img512-s-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.190.pkl'),   # fd_dinov2 = 68.64
    'edm2-img512-m-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.155.pkl'),   # fd_dinov2 = 58.44
    'edm2-img512-l-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.155.pkl'),   # fd_dinov2 = 52.25
    'edm2-img512-xl-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.155.pkl'),  # fd_dinov2 = 45.96
    'edm2-img512-xxl-dino':      dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.150.pkl'), # fd_dinov2 = 42.84
    'edm2-img512-xs-guid-fid':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.045.pkl',   gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.045.pkl', guidance=1.4), # fid = 2.91
    'edm2-img512-s-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.025.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.025.pkl', guidance=1.4), # fid = 2.23
    'edm2-img512-m-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.030.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.2), # fid = 2.01
    'edm2-img512-l-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.015.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.2), # fid = 1.88
    'edm2-img512-xl-guid-fid':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.020.pkl',   gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.020.pkl', guidance=1.2), # fid = 1.85
    'edm2-img512-xxl-guid-fid':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',  gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.2), # fid = 1.81
    'edm2-img512-xs-guid-dino':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.150.pkl',   gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.150.pkl', guidance=1.7), # fd_dinov2 = 79.94
    'edm2-img512-s-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.085.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.085.pkl', guidance=1.9), # fd_dinov2 = 52.32
    'edm2-img512-m-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.015.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=2.0), # fd_dinov2 = 41.98
    'edm2-img512-l-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.035.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.035.pkl', guidance=1.7), # fd_dinov2 = 38.20
    'edm2-img512-xl-guid-dino':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.030.pkl',   gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.7), # fd_dinov2 = 35.67
    'edm2-img512-xxl-guid-dino': dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',  gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.7), # fd_dinov2 = 33.09
}

#----------------------------------------------------------------------------
# EDM sampler from the paper
# "Elucidating the Design Space of Diffusion-Based Generative Models",
# extended to support classifier-free guidance.

def edm_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
    data_mean=None,
    eps_scaler=1.,
    **sampler_kwargs,
):
    # Guided denoiser.
    def denoise(x, t, t_next):
        Dx = net(x, t, t_next, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / max(num_steps - 1, 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    afs = False
    if data_mean is None:
        data_mean = torch.zeros_like(x_next)
    Dx_prev = data_mean if afs else None
    w = 0.8

    def inner(z, t_steps):
        Dx_list = []
        for (t_cur, t_next) in zip(t_steps[:-1], t_steps[1:]):
            Dx = denoise(z, t_cur, t_next)
            z = z + (t_next - t_cur) * (z - Dx) / t_cur
            Dx_list.append(Dx)
        return Dx_list[-1], Dx_list[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        if afs and i == 0:
            Dx = data_mean # torch.zeros_like(x_hat)
        else:
            Dx = denoise(x_hat, t_hat, t_next.clamp_min(sigma_min))

        if Dx_prev is not None:
            d_cur = (x_hat - ((1+w)*Dx-w*Dx_prev)) / t_hat
        else:
            d_cur = (x_hat - Dx) / t_hat

        if i < num_steps - 1:
            d_cur = d_cur / eps_scaler
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Dx_prev = Dx

        # Apply 2nd order correction.
        if i < num_steps - 1:
            Dx_next = denoise(x_next, t_next, t_steps[i + 2] if i + 2 < len(t_steps) else t_steps[-1])
            d_prime = (x_next - Dx_next) / t_next
            d_prime = d_prime / eps_scaler
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def euler_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
    data_mean=None,
    eps_scaler=1.,
    sg_weight=0.,
    **sampler_kwargs,
):
    # Guided denoiser.
    def denoise(x, t, t_next):
        Dx = net(x, t, t_next, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / max(num_steps - 1, 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    afs = False
    if data_mean is None:
        data_mean = torch.zeros_like(x_next)
    Dx_prev = data_mean if afs else None

    def inner(z, t_steps):
        Dx_list = []
        for (t_cur, t_next) in zip(t_steps[:-1], t_steps[1:]):
            Dx = denoise(z, t_cur, t_next)
            z = z + (t_next - t_cur) * (z - Dx) / t_cur
            Dx_list.append(Dx)
        return Dx_list[-1], Dx_list[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        if afs and i == 0:
            Dx = data_mean # torch.zeros_like(x_hat)
        else:
            Dx = denoise(x_hat, t_hat, t_next.clamp_min(sigma_min))

        if Dx_prev is not None:
            d_cur = (x_hat - ((1+sg_weight)*Dx-sg_weight*Dx_prev)) / t_hat
        else:
            d_cur = (x_hat - Dx) / t_hat

        if i < num_steps - 1:
            d_cur = d_cur / eps_scaler
        x_next = x_hat + (t_next - t_hat) * d_cur

        Dx_prev = Dx

    return x_next

def double_euler_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
    data_mean=None,
    eps_scaler=1.,
    sg_weight=0.,
    **sampler_kwargs,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / max(num_steps - 1, 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    afs = False
    if data_mean is None:
        data_mean = torch.zeros_like(x_next)
    Dx_prev = data_mean if afs else None

    def inner(z, t_steps):
        Dx_list = []
        for (t_cur, t_next) in zip(t_steps[:-1], t_steps[1:]):
            Dx = denoise(z, t_cur)
            z = z + (t_next - t_cur) * (z - Dx) / t_cur
            Dx_list.append(Dx)
        return Dx_list[-1], Dx_list[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        if afs and i == 0:
            Dx = data_mean # torch.zeros_like(x_hat)
        else:
            Dx, _ = inner(x_hat, t_steps[i:])

        if Dx_prev is not None:
            d_cur = (x_hat - ((1+sg_weight)*Dx-sg_weight*Dx_prev)) / t_hat
        else:
            d_cur = (x_hat - Dx) / t_hat

        if i < num_steps - 1:
            d_cur = d_cur / eps_scaler
        x_next = x_hat + (t_next - t_hat) * d_cur

        Dx_prev = Dx

    return x_next

def random_euler_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
    data_mean=None,
    eps_scaler=1.,
    sg_weight=0.,
    **sampler_kwargs,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / max(num_steps - 1, 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    afs = False
    if data_mean is None:
        data_mean = torch.zeros_like(x_next)
    Dx_prev = data_mean if afs else None

    def inner(z, t_steps):
        Dx_list = []
        for (t_cur, t_next) in zip(t_steps[:-1], t_steps[1:]):
            Dx = denoise(z, t_cur)
            z = z + (t_next - t_cur) * (z - Dx) / t_cur
            Dx_list.append(Dx)
        return Dx_list[-1], Dx_list[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_next = inner(x_next, t_steps[i:])[0] + randn_like(x_next) * t_next

    return x_next

def unipc_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    order=2, variant='bh1', # UniPC 專屬參數: order (階數), variant ('bh1' 或 'bh2')
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, # UniPC 是常微分方程 (ODE) 求解器，SDE 的隨機參數將被忽略
    dtype=torch.float32, randn_like=torch.randn_like,
    data_mean=None,
    **sampler_kwargs,
):
    """基於 EDM2 風格實作的 UniPC 採樣器"""
    
    if S_churn > 0:
        import warnings
        warnings.warn("UniPC 是一個 ODE 求解器。S_churn 等隨機性注入參數將在此採樣器中被忽略。")

    # 分類器自由引導 (Classifier-Free Guidance) 去噪函數
    def denoise(x, t_val):
        Dx = net(x, t_val, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t_val, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # 建立時間步離散化 (Time step discretization，與 EDM 相同)
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # 進入主採樣迴圈
    x = noise.to(dtype) * t_steps[0]
    
    model_prev_list = []
    t_prev_list = []
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        
        # 在第一步計算初始去噪結果並緩存
        if i == 0:
            model_cur = denoise(x, t_cur)
            model_prev_list.append(model_cur)
            t_prev_list.append(t_cur)
            
        # 維護緩衝區大小不超過指定階數 (order)
        if len(model_prev_list) > order:
            model_prev_list.pop(0)
            t_prev_list.pop(0)
            
        step_order = len(model_prev_list)
        
        # EDM 的最後一步目標時間點 t_next 為 0
        # 根據 UniPC (DPM-Solver) 解析解，若目標 t 為 0，最佳解即為當前模型的乾淨影像預測結果
        if t_next == 0:
            x = model_prev_list[-1]
            break
            
        # 判斷是否使用 Corrector
        use_corrector = True
        if i == num_steps - 1:
            use_corrector = False
            
        t_prev_0 = t_prev_list[-1]
        model_prev_0 = model_prev_list[-1]
        
        # 準備 UniPC 需要的係數: h 和 h_phi
        h = torch.log(t_prev_0 / t_next)
        hh = -h
        h_phi_1 = torch.expm1(hh) # 等同於 (t_next - t_prev_0) / t_prev_0
        
        if variant == 'bh1':
            B_h = hh
        elif variant == 'bh2':
            B_h = h_phi_1
        else:
            raise ValueError(f"不支援的 UniPC 變體: {variant}，請使用 'bh1' 或 'bh2'")
            
        rks = []
        D1s = []
        for j in range(1, step_order):
            t_prev_j = t_prev_list[-(j + 1)]
            model_prev_j = model_prev_list[-(j + 1)]
            rk = torch.log(t_prev_0 / t_prev_j) / h
            rks.append(rk)
            D1s.append((model_prev_j - model_prev_0) / rk)
            
        rks.append(torch.tensor(1.0, dtype=dtype, device=x.device))
        rks = torch.stack(rks)
        
        R = []
        b = []
        h_phi_k = h_phi_1 / hh - 1
        factorial_j = 1
        
        for j in range(1, step_order + 1):
            R.append(torch.pow(rks, j - 1))
            b.append(h_phi_k * factorial_j / B_h)
            factorial_j *= (j + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_j
            
        R = torch.stack(R)
        b = torch.stack(b)
        
        # 計算預測器 (Predictor) 和校正器 (Corrector) 的係數 rhos
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1) # Shape: (Batch, K, Channels, Height, Width)
            if step_order == 2:
                rhos_p = torch.tensor([0.5], dtype=dtype, device=x.device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None
            
        if use_corrector:
            if step_order == 1:
                rhos_c = torch.tensor([0.5], dtype=dtype, device=x.device)
            else:
                rhos_c = torch.linalg.solve(R, b)
                
        # =================== Predictor Step ===================
        # 對於 VE ODE，此處等效為指數積分器的泰勒展開
        x_t_ = (t_next / t_prev_0) * x - h_phi_1 * model_prev_0
        if D1s is not None:
            # 使用 einsum 處理任意形狀的空間維度 (2D 或 3D 卷積皆相容)
            pred_res = torch.einsum('k,bk...->b...', rhos_p, D1s)
        else:
            pred_res = 0
            
        x_t_pred = x_t_ - B_h * pred_res
        
        # =================== Corrector Step ===================
        if use_corrector:
            # 僅在這裡進行了一次新的模型推論 (與 Euler 法的 NFE 花費相同！)
            model_t = denoise(x_t_pred, t_next)
            if D1s is not None:
                corr_res = torch.einsum('k,bk...->b...', rhos_c[:-1], D1s)
            else:
                corr_res = 0
                
            D1_t = model_t - model_prev_0
            x = x_t_ - B_h * (corr_res + rhos_c[-1] * D1_t)
            
            # 將最新狀態推進緩衝區
            model_prev_list.append(model_t)
            t_prev_list.append(t_next)
        else:
            x = x_t_pred
            
    return x


sampler = {
    'edm': edm_sampler,
    'euler': euler_sampler,
    'unipc': unipc_sampler,
    'd-euler': double_euler_sampler,
    'r-euler': random_euler_sampler,
}
#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)

def generate_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Reference network for guidance. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = edm_sampler,          # Which sampler function to use.
    data_mean           = None,                 # Data mean for normalization. Path or tensor. None = 0.
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    sampler_fn = sampler[sampler_fn] if isinstance(sampler_fn, str) else sampler_fn

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guidance network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    if data_mean is not None:
        data_mean = torch.load(data_mean, map_location=device)
    else:
        data_mean = torch.zeros(1, net.img_channels, net.img_resolution, net.img_resolution, device=device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:

                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn([len(r.seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
                    r.labels = None
                    if net.label_dim > 0:
                        r.labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[len(r.seeds)], device=device)]
                        if class_idx is not None:
                            r.labels[:, :] = 0
                            r.labels[:, class_idx] = 1

                    # Generate images.
                    latents = dnnlib.util.call_func_by_name(func_name=sampler_fn, net=net, noise=r.noise,
                        labels=r.labels, gnet=gnet, randn_like=rnd.randn_like, data_mean=data_mean, **sampler_kwargs)
                    r.images = encoder.decode(latents)

                    # Save images.
                    if outdir is not None:
                        for seed, image in zip(r.seeds, r.images.permute(0, 2, 3, 1).cpu().numpy()):
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:06d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:06d}.png'))

                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, default=None)
@click.option('--net',                      help='Network pickle filename', metavar='PATH|URL',                     type=str, default=None)
@click.option('--gnet',                     help='Reference network for guidance', metavar='PATH|URL',              type=str, default=None)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--data_mean',                help='Data mean for normalization', metavar='PATH',                     type=str, default=None)

@click.option('--sampler', 'sampler_fn',    help='Sampler function', metavar='STR',                                 type=str, default='edm')
@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=None)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                         type=float, default=1, show_default=True)
@click.option('--eps_scaler', 'eps_scaler', help='Epsion scaling', metavar='FLOAT',                                 type=float, default=1., show_default=True)
@click.option('--sg', 'sg_weight',          help='Self-guidance weight', metavar='FLOAT',                           type=float, default=0., show_default=True)

def cmdline(preset, **opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = dnnlib.EasyDict(opts)

    # Apply preset.
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts[key] is None:
                opts[key] = value

    # Validate options.
    if opts.net is None:
        raise click.ClickException('Please specify either --preset or --net')
    if opts.guidance is None or opts.guidance == 1:
        opts.guidance = 1
        opts.gnet = None
    elif opts.gnet is None:
        raise click.ClickException('Please specify --gnet when using guidance')

    # Generate.
    dist.init()
    image_iter = generate_images(**opts)
    for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
