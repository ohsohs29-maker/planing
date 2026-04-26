"""
9장 실습: Prithvi 기반 홍수 탐지 파인튜닝

이 스크립트는 NASA-IBM Prithvi 파운데이션 모델을 활용하여
홍수 탐지 세그멘테이션 모델을 LoRA로 파인튜닝한다.

실행 방법:
    python 9-5-prithvi-flood-detection.py --data_dir ../data --epochs 20

필요 환경:
    pip install -r requirements.txt
    GPU: V100 32GB 이상 권장
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

# 선택적 import
try:
    from transformers import AutoModel, AutoImageProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers 라이브러리가 필요합니다.")

try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("Warning: peft 라이브러리가 없어 LoRA를 사용할 수 없습니다.")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# =============================================================================
# 데이터셋 클래스
# =============================================================================

class FloodDataset(Dataset):
    """홍수 탐지 데이터셋 (HLS 형식)"""

    def __init__(self, root_dir, split='train', num_bands=6, image_size=224):
        self.root_dir = Path(root_dir)
        self.num_bands = num_bands
        self.image_size = image_size

        # 타일 목록 로드
        split_file = self.root_dir / 'splits' / f'{split}.txt'

        if split_file.exists():
            with open(split_file) as f:
                self.tile_ids = [line.strip() for line in f if line.strip()]
            self.demo_mode = False
        else:
            print(f"  [Demo] Split file not found: {split_file}")
            self.tile_ids = [f'demo_{i:03d}' for i in range(50 if split == 'train' else 10)]
            self.demo_mode = True

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]

        if self.demo_mode:
            # 데모: 가상 HLS 데이터 생성
            image = np.random.rand(self.num_bands, self.image_size, self.image_size).astype(np.float32)
            # 홍수 마스크 (일부 영역만 홍수)
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            cx, cy = np.random.randint(50, 174, 2)
            r = np.random.randint(20, 50)
            y, x = np.ogrid[:self.image_size, :self.image_size]
            mask_region = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
            mask[mask_region] = 1.0
        else:
            # 실제 데이터 로드
            img_path = self.root_dir / 'images' / f'{tile_id}.tif'
            mask_path = self.root_dir / 'masks' / f'{tile_id}.tif'

            if HAS_RASTERIO:
                with rasterio.open(img_path) as src:
                    image = src.read().astype(np.float32)
                image = np.clip(image / 10000.0, 0, 1)

                with rasterio.open(mask_path) as src:
                    mask = src.read(1).astype(np.float32)
            else:
                image = np.load(str(img_path).replace('.tif', '.npy')).astype(np.float32)
                mask = np.load(str(mask_path).replace('.tif', '.npy')).astype(np.float32)

        return torch.from_numpy(image), torch.from_numpy(mask)


# =============================================================================
# 세그멘테이션 헤드
# =============================================================================

class FloodSegmentationHead(nn.Module):
    """Prithvi 출력을 받아 홍수 마스크 예측"""

    def __init__(self, embed_dim=1024, hidden_dim=256, output_size=224):
        super().__init__()
        self.output_size = output_size

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, features):
        # features: (B, num_patches, embed_dim)
        B, N, C = features.shape

        # 패치를 2D 그리드로 재구성
        H = W = int(N ** 0.5)
        features = features.permute(0, 2, 1).reshape(B, C, H, W)

        # 디코딩
        x = self.decoder(features)

        # 원본 해상도로 업샘플링
        x = F.interpolate(x, size=(self.output_size, self.output_size),
                          mode='bilinear', align_corners=False)

        return x.squeeze(1)


# =============================================================================
# 간단한 백본 (Prithvi 없을 때 사용)
# =============================================================================

class SimpleViTBackbone(nn.Module):
    """Prithvi 대신 사용할 간단한 ViT 백본"""

    def __init__(self, in_channels=6, embed_dim=256, patch_size=16, image_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 패치 임베딩
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # 위치 임베딩
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer 블록
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # 패치 임베딩
        x = self.patch_embed(x)  # (B, embed_dim, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # 위치 임베딩 추가
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        return type('obj', (object,), {'last_hidden_state': x})()


# =============================================================================
# 학습 함수
# =============================================================================

def calculate_iou(preds, masks):
    """IoU 계산"""
    preds = preds > 0.5
    masks = masks > 0.5

    intersection = (preds & masks).sum().item()
    union = (preds | masks).sum().item()

    return intersection / (union + 1e-6)


def train_model(backbone, seg_head, train_loader, val_loader,
                epochs, device, lr_backbone=1e-5, lr_head=1e-3):
    """모델 학습"""

    backbone = backbone.to(device)
    seg_head = seg_head.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW([
        {'params': backbone.parameters(), 'lr': lr_backbone},
        {'params': seg_head.parameters(), 'lr': lr_head},
    ], weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_iou = 0.0

    for epoch in range(epochs):
        # 학습
        backbone.train()
        seg_head.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # 특징 추출
            outputs = backbone(images)
            features = outputs.last_hidden_state

            # CLS 토큰 제외 (있는 경우)
            if features.shape[1] == 197:  # 196 patches + 1 CLS
                features = features[:, 1:, :]

            # 세그멘테이션
            preds = seg_head(features)
            loss = criterion(preds, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = train_loss / len(train_loader)

        # 검증
        backbone.eval()
        seg_head.eval()
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = backbone(images)
                features = outputs.last_hidden_state
                if features.shape[1] == 197:
                    features = features[:, 1:, :]

                preds = torch.sigmoid(seg_head(features))
                val_iou += calculate_iou(preds, masks)

        val_iou /= len(val_loader)

        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Val IoU={val_iou:.4f}')

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'backbone': backbone.state_dict(),
                'seg_head': seg_head.state_dict(),
            }, 'best_flood_model.pth')
            print(f'  => Best model saved! IoU={val_iou:.4f}')

        scheduler.step()

    return best_iou


# =============================================================================
# 메인 함수
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prithvi 홍수 탐지 파인튜닝')
    parser.add_argument('--data_dir', type=str, default='../data', help='데이터 디렉토리')
    parser.add_argument('--epochs', type=int, default=20, help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='백본 학습률')
    parser.add_argument('--lr_head', type=float, default=1e-3, help='헤드 학습률')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--use_prithvi', action='store_true', help='Prithvi 모델 사용')

    args = parser.parse_args()

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 데이터셋
    print('\n데이터셋 로드 중...')
    train_dataset = FloodDataset(args.data_dir, split='train')
    val_dataset = FloodDataset(args.data_dir, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f'  Train samples: {len(train_dataset)}')
    print(f'  Val samples: {len(val_dataset)}')

    # 모델 생성
    print('\n모델 생성 중...')

    if args.use_prithvi and HAS_TRANSFORMERS:
        print('  Prithvi-EO-2.0 로드 중...')
        model_name = "ibm-nasa-geospatial/Prithvi-EO-2.0-600M"

        try:
            backbone = AutoModel.from_pretrained(model_name)
            embed_dim = 1024

            # LoRA 적용
            if HAS_PEFT:
                print(f'  LoRA 적용 중 (r={args.lora_r})...')
                lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_r * 2,
                    target_modules=["query", "value"],
                    lora_dropout=0.1,
                )
                backbone = get_peft_model(backbone, lora_config)

                trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
                total = sum(p.numel() for p in backbone.parameters())
                print(f'  학습 파라미터: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')

        except Exception as e:
            print(f'  Prithvi 로드 실패: {e}')
            print('  SimpleViT 백본 사용')
            backbone = SimpleViTBackbone(in_channels=6, embed_dim=256)
            embed_dim = 256
    else:
        print('  SimpleViT 백본 사용 (데모 모드)')
        backbone = SimpleViTBackbone(in_channels=6, embed_dim=256)
        embed_dim = 256

    # 세그멘테이션 헤드
    seg_head = FloodSegmentationHead(embed_dim=embed_dim)

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in backbone.parameters())
    total_params += sum(p.numel() for p in seg_head.parameters())
    print(f'  전체 파라미터: {total_params:,}')

    # 학습
    print('\n학습 시작...')
    best_iou = train_model(
        backbone, seg_head,
        train_loader, val_loader,
        epochs=args.epochs,
        device=device,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head
    )

    # 결과 출력
    print('\n' + '=' * 60)
    print('9장 실습 완료: Prithvi 기반 홍수 탐지')
    print('=' * 60)
    print(f'  파인튜닝 전략: {"LoRA" if HAS_PEFT and args.use_prithvi else "Full"}')
    print(f'  Best Val IoU: {best_iou:.4f}')
    print(f'  저장된 모델: best_flood_model.pth')
    print('=' * 60)


if __name__ == '__main__':
    main()
