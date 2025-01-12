import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 改进1: 增强数据预处理
transform = transforms.Compose([
    transforms.RandomRotation(10),  # 添加随机旋转
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 添加随机平移
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的真实均值和标准差
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=test_transform, download=True)

# 改进2: 调整batch size
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

class ImprovedTransformer(nn.Module):
    def __init__(self, patch_size=4, emb_dim=256, num_heads=8, num_classes=10, depth=6, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (28 // patch_size) ** 2
        
        # 改进3: 使用卷积进行patch embedding
        self.patch_embed = nn.Conv2d(1, emb_dim, kernel_size=patch_size, stride=patch_size)
        
        # 改进4: 添加可学习的类别嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))
        
        # 改进5: 添加dropout
        self.pos_dropout = nn.Dropout(dropout)
        
        # 改进6: 使用更深的transformer和更好的参数
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,  # 增加FFN维度
            dropout=dropout,
            activation='gelu',  # 使用GELU激活函数
            batch_first=True,
            norm_first=True  # 使用Pre-LN结构
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 改进7: 更复杂的分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        
        # 使用卷积进行patch embedding
        x = self.patch_embed(x)  # [B, emb_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, emb_dim]
        
        # 添加分类token和位置编码
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.pos_dropout(x)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 分类
        return self.mlp_head(x[:, 0])

# 改进8: 更好的权重初始化
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedTransformer().to(device)
model.apply(initialize_weights)

# 改进9: 使用学习率调度器和更好的优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 改进10: 添加早停机制
best_acc = 0
patience = 5
no_improve = 0

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        # 改进11: 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    scheduler.step()
    
    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # 早停检查
    if accuracy > best_acc:
        best_acc = accuracy
        no_improve = 0
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping triggered")
            break