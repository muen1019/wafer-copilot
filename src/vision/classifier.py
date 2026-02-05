import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import uuid

from src.vision.gradcam import GradCAM, apply_colormap, generate_heatmap_only, create_comparison_image

# WM-811K 的標準 8 大類別
CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']

class WaferClassifier:
    def __init__(self, model_path="models/resnet_wm811k.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = CLASSES
        
        # 建立模型架構
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(CLASSES))
        self.model = self.model.to(self.device)
        
        # 嘗試載入權重，如果失敗則進入模擬模式
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.mock_mode = False
            print(f"✅ 模型已載入: {model_path}")
        else:
            self.mock_mode = True
            print("⚠️ 未找到模型權重，啟用模擬模式 (Mock Mode)。")
        
        # 初始化 Grad-CAM (使用 ResNet18 的最後一個卷積層: layer4)
        self.gradcam = GradCAM(self.model, self.model.layer4[-1])
        
        # 標準化預處理（用於所有推論，必須與訓練時一致）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, generate_cam=False):
        """
        預測晶圓瑕疵類別
        
        Args:
            image_path: 圖片路徑
            generate_cam: 是否生成 Grad-CAM 熱力圖
            
        Returns:
            dict: 包含 label, confidence, 以及可選的 cam_path
        """
        if self.mock_mode:
            # 模擬回傳一個結果
            result = {"label": "Edge-Ring", "confidence": 0.95}
            if generate_cam:
                result["cam_path"] = None
                result["cam_overlay_path"] = None
            return result

        try:
            # 載入原始圖片
            original_image = Image.open(image_path).convert('RGB')
            
            if generate_cam:
                # 使用 Grad-CAM 進行預測（需要梯度）
                image_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
                
                # 生成 Grad-CAM
                cam, pred_class, confidence = self.gradcam.generate(image_tensor)
                label = self.classes[pred_class]
                
                # 生成視覺化圖片
                cam_overlay = apply_colormap(cam, original_image, alpha=0.5)
                cam_heatmap = generate_heatmap_only(cam)
                
                # 儲存熱力圖
                output_dir = "data/gradcam_outputs"
                os.makedirs(output_dir, exist_ok=True)
                
                unique_id = uuid.uuid4().hex[:8]
                cam_path = os.path.join(output_dir, f"heatmap_{unique_id}.png")
                overlay_path = os.path.join(output_dir, f"overlay_{unique_id}.png")
                comparison_path = os.path.join(output_dir, f"comparison_{unique_id}.png")
                
                cam_heatmap.save(cam_path)
                cam_overlay.save(overlay_path)
                
                # 創建三合一比較圖
                comparison = create_comparison_image(original_image, cam_heatmap, cam_overlay)
                comparison.save(comparison_path)
                
                return {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "cam_path": cam_path,
                    "cam_overlay_path": overlay_path,
                    "comparison_path": comparison_path
                }
            else:
                # 不需要 Grad-CAM，使用標準推論（注意：必須使用與訓練相同的預處理）
                image_tensor = self.transform(original_image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    top_prob, top_class = probs.topk(1, dim=1)

                return {
                    "label": self.classes[top_class.item()],
                    "confidence": round(top_prob.item(), 4)
                }
        except Exception as e:
            return {"error": str(e)}
