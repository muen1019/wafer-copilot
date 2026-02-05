"""
Grad-CAM (Gradient-weighted Class Activation Mapping) 實作
用於視覺化 CNN 模型的決策依據，讓工程師「看到」模型關注的晶圓區域
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os


class GradCAM:
    """
    Grad-CAM 實作類別
    
    原理：利用目標類別相對於特徵圖的梯度，計算每個通道的重要性權重，
    再對特徵圖進行加權求和，生成類別激活熱力圖。
    
    參考論文：Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks 
              via Gradient-based Localization", ICCV 2017
    """
    
    def __init__(self, model, target_layer):
        """
        初始化 Grad-CAM
        
        Args:
            model: PyTorch CNN 模型
            target_layer: 要提取特徵的目標層 (通常是最後一個卷積層)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 註冊 hooks 來捕捉前向傳播的激活值和反向傳播的梯度
        self._register_hooks()
    
    def _register_hooks(self):
        """註冊前向和反向傳播的 hooks"""
        
        def forward_hook(module, input, output):
            # 儲存目標層的輸出（激活值）
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            # 儲存目標層的梯度
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        生成 Grad-CAM 熱力圖
        
        Args:
            input_tensor: 預處理後的輸入圖片張量 [1, C, H, W]
            target_class: 目標類別索引，若為 None 則使用預測類別
            
        Returns:
            cam: 正規化的熱力圖 [H, W]，值域 [0, 1]
            pred_class: 預測的類別索引
            confidence: 預測信心度
        """
        self.model.eval()
        
        # 確保輸入需要梯度
        input_tensor = input_tensor.requires_grad_(True)
        
        # 前向傳播
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        # 如果未指定目標類別，使用預測類別
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = probs[0, target_class].item()
        
        # 清除之前的梯度
        self.model.zero_grad()
        
        # 反向傳播：對目標類別的分數求梯度
        target_score = output[0, target_class]
        target_score.backward()
        
        # 計算權重：對梯度進行全局平均池化 (Global Average Pooling)
        # gradients shape: [1, C, H, W] -> weights shape: [C]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # 加權求和：將權重與激活值相乘後求和
        # activations shape: [1, C, H, W]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # ReLU：只保留正向影響的區域
        cam = F.relu(cam)
        
        # 正規化到 [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class, confidence


def apply_colormap(cam, original_image, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    將 Grad-CAM 熱力圖疊加到原始圖片上
    
    Args:
        cam: Grad-CAM 熱力圖 [H, W]，值域 [0, 1]
        original_image: PIL Image 或 numpy array
        alpha: 熱力圖透明度
        colormap: OpenCV 色彩映射
        
    Returns:
        overlayed: 疊加後的圖片 (PIL Image)
    """
    # 將原始圖片轉換為 numpy array
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # 確保是 RGB 格式
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 4:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
    
    # 調整 CAM 大小以匹配原始圖片
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    
    # 將 CAM 轉換為 0-255 的 uint8
    cam_uint8 = np.uint8(255 * cam_resized)
    
    # 應用色彩映射
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 疊加熱力圖和原始圖片
    overlayed = np.uint8(alpha * heatmap + (1 - alpha) * original_image)
    
    return Image.fromarray(overlayed)


def create_comparison_image(original, heatmap, cam_overlay):
    """
    創建三合一比較圖：原圖 | 熱力圖 | 疊加圖
    
    Args:
        original: 原始圖片 (PIL Image)
        heatmap: 純熱力圖 (PIL Image)
        cam_overlay: 疊加圖 (PIL Image)
        
    Returns:
        combined: 合併後的圖片 (PIL Image)
    """
    # 統一大小
    size = (224, 224)
    original = original.resize(size)
    heatmap = heatmap.resize(size)
    cam_overlay = cam_overlay.resize(size)
    
    # 創建合併圖片
    combined = Image.new('RGB', (size[0] * 3 + 20, size[1]), (255, 255, 255))
    combined.paste(original, (0, 0))
    combined.paste(heatmap, (size[0] + 10, 0))
    combined.paste(cam_overlay, (size[0] * 2 + 20, 0))
    
    return combined


def generate_heatmap_only(cam, size=(224, 224), colormap=cv2.COLORMAP_JET):
    """
    生成純熱力圖（不疊加原圖）
    
    Args:
        cam: Grad-CAM 熱力圖 [H, W]
        size: 輸出大小
        colormap: 色彩映射
        
    Returns:
        heatmap: PIL Image
    """
    cam_resized = cv2.resize(cam, size)
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap)
