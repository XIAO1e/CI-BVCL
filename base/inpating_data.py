import torch,os
import pandas as pd
import numpy as np
from PIL import Image
import logging
import open_clip
import pickle

import cv2
from PIL import Image
import random
import numpy as np
import torch
import logging
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
from scipy.optimize import fsolve


class DirectT:
    def __init__(self):
        pass
    def __call__(self,x,U=None):
        return x
    
class UniformBlur:
    def __init__(self,blur_kernel_size):
        self.blur_kernel_size = blur_kernel_size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        img_np = np.array(img)
        if img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # sanitize kernel size: positive odd integer
        k = int(self.blur_kernel_size)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1

        img_blur = cv2.GaussianBlur(img_np, (k, k), 0)
        img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_blur)
    
class FoveaBlur:
    def __init__(self, h, w, blur_kernel_size, curve_type='exp', *args, **kwargs):
        self.blur_kernel_size = blur_kernel_size
        self.mask = np.zeros((h,w), np.float32)
        
        center = (w // 2, h // 2)
        max_distance = np.sqrt((h - center[1] - 1) ** 2 + (w - center[0] - 1) ** 2)
        c = 0.5
        center_resolution = 1-c
        edge_resolution = 0

        initial_guess = [1.0, 1.0]
        def equations(vars):
            t, r = vars
            eq1 = r * (t - np.sin(t)) - 1  # x = 1
            eq2 = -r * (1 - np.cos(t)) + 1.0  # y = 0
            return [eq1, eq2]
        solution = fsolve(equations, initial_guess)
        t_max, r_solution = solution
        self.r = r_solution

        fun_degrade = getattr(self, curve_type, None)
        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
                x0 = min(1,distance/max_distance)
                y0 = fun_degrade(x0,**kwargs)
                self.mask[i, j] = edge_resolution + (center_resolution - edge_resolution) * y0

    def alphaBlend(self, img1, img2, mask):
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
        return blended
    
    def __call__(self, img, blur_kernel_size=None): 
        if blur_kernel_size ==None:
            blur_kernel_size = self.blur_kernel_size
        img = np.array(img)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # sanitize kernel size: positive odd integer
        k = int(blur_kernel_size)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        blured = cv2.GaussianBlur(img, (k, k), 0)
        blended = self.alphaBlend(img, blured, 1- self.mask)
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        return Image.fromarray(blended)
    
    def linear(self,x,**kwargs):
        return 1-x
    
    def exp(self,x,**kwargs):
        system_g = kwargs.get('system_g', 4)
        return  np.exp(-system_g * x)
    
    def quadratic(self,x,**kwargs):
        return  1 - x**2
    
    def log(self,x,**kwargs):
        b = 1/(np.e-1)
        a = np.log(b) + 1
        return  a - np.log(x + b)
    
    def brachistochrone(self,x,**kwargs):
        
        def equation(t):
            return t - np.sin(t) - (x / self.r)

        t0 = fsolve(equation, [1.0, 1.0])[0]
        y0 = -self.r * (1 - np.cos(t0)) + 1.0
        return  y0


class CSFLowpass:
    def __init__(self, h: int = 224, w: int = 224, strength: float = 1.0, cutoff_ratio: float = 0.25):
        self.h = h
        self.w = w
        self.strength = float(max(0.0, strength))
        self.cutoff_ratio = float(min(max(cutoff_ratio, 0.05), 0.95))

        u = np.fft.fftfreq(self.h)[:, None]
        v = np.fft.fftfreq(self.w)[None, :]
        radius = np.sqrt(u * u + v * v)

        f0 = self.cutoff_ratio * 0.5
        s = 1.0 / (1.0 + (radius / (f0 + 1e-8)) ** 4)
        s = s / (s.max() + 1e-8)
        self.filter = s.astype(np.float32)

    def _apply_single(self, img: np.ndarray) -> np.ndarray:
        img_f = np.fft.fft2(img)
        img_fshift = np.fft.fftshift(img_f)
        filtered = img_fshift * self.filter
        img_ishift = np.fft.ifftshift(filtered)
        out = np.fft.ifft2(img_ishift)
        out = np.real(out)
        return out

    def __call__(self, img: Image.Image) -> Image.Image:
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        x = np.array(img).astype(np.float32) / 255.0
        if x.ndim == 2:
            y = self._apply_single(x)
        else:
            if x.shape[2] == 3:
                x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
            channels = []
            for c in range(x.shape[2]):
                channels.append(self._apply_single(x[:, :, c]))
            y = np.stack(channels, axis=2)
            y = np.clip(y, 0.0, 1.0)
            y = (y * 255.0).astype(np.uint8)
            y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
        if y.dtype != np.uint8:
            y = np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(y)