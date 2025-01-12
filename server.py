# server.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import io
import base64
import numpy as np
from flask_cors import CORS  # 添加CORS支持

import os

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)


# 添加路由来服务主页
@app.route('/')
def home():
    print("Hello, World!")
    return render_template('index.html')

# 首先定义模型架构
class ImprovedTransformer(nn.Module):
    def __init__(self, patch_size=4, emb_dim=256, num_heads=8, num_classes=10, depth=6, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (28 // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(1, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.pos_dropout(x)
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])

# 定义预测器类
class ModelPredictor:
    def __init__(self, model_path='best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ImprovedTransformer().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def predict(self, image):
        # 确保图像是灰度图
        if isinstance(image, str):
            image = Image.open(image).convert('L')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8')).convert('L')
            
        # 反转颜色（如果需要 - 因为画布是黑底白字）
        image = Image.fromarray(255 - np.array(image))
        
        # 预处理
        input_tensor = self.transform(image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = prob[0][predicted_class].item()
        
        return predicted_class, confidence


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取Base64编码的图像数据
        image_data = request.json['image']
        # 移除Base64头部
        image_data = image_data.split(',')[1]
        # 解码Base64数据
        image_bytes = base64.b64decode(image_data)
        # 转换为图像
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # 进行预测
        predicted_class, confidence = predictor.predict(image)
        
        return jsonify({
            'prediction': int(predicted_class),
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 确保目录结构存在
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # 初始化预测器
    predictor = ModelPredictor('best_model.pth')
    
    app.run(debug=True, port=5000)