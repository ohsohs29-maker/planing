"""
6주차 보충자료: U-Net 기반 토지피복 세그멘테이션

이 스크립트는 Sentinel-2 다중 스펙트럼 영상을 사용하여
토지피복 세그멘테이션 파이프라인을 구현한다.

실행 방법:
    python 6-supplement-unet-landcover-segmentation.py --data_dir ../data --epochs 50

필요 환경:
    pip install -r requirements.txt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import cv2

# 선택적 import
try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False
    print("Warning: segmentation_models_pytorch not installed. Using simple U-Net.")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import albumentations as A
    HAS_ALBUMENTATION = True
except ImportError:
    HAS_ALBUMENTATION = False

try:
    import geopandas as gpd
    from shapely.geometry import shape
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


# =============================================================================
# 데이터셋 클래스
# =============================================================================

class LandCoverDataset(Dataset):
    """토지피복 세그멘테이션 데이터셋"""

    def __init__(self, root_dir, split='train', transform=None, num_bands=13, num_classes=10):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.num_bands = num_bands
        self.num_classes = num_classes

        # 타일 목록 로드
        split_file = self.root_dir / 'splits' / f'{split}.txt'

        if split_file.exists():
            with open(split_file) as f:
                self.tile_ids = [line.strip() for line in f if line.strip()]
        else:
            # 데모 모드: 가상 데이터 생성
            print(f"  [Demo] Split file not found: {split_file}")
            self.tile_ids = [f'demo_{i:03d}' for i in range(100 if split == 'train' else 20)]
            self.demo_mode = True

        self.demo_mode = not split_file.exists()

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]

        if self.demo_mode:
            # 데모 모드: 가상 데이터 생성
            image = np.random.rand(self.num_bands, 256, 256).astype(np.float32)
            mask = np.random.randint(0, self.num_classes, (256, 256)).astype(np.int64)
        else:
            # 실제 데이터 로드
            img_path = self.root_dir / 'images' / f'{tile_id}.tif'
            mask_path = self.root_dir / 'masks' / f'{tile_id}.tif'

            if HAS_RASTERIO:
                with rasterio.open(img_path) as src:
                    image = src.read().astype(np.float32)
                image = np.clip(image / 10000.0, 0, 1)

                with rasterio.open(mask_path) as src:
                    mask = src.read(1).astype(np.int64)
            else:
                # rasterio 없으면 numpy로 로드 시도
                image = np.load(str(img_path).replace('.tif', '.npy')).astype(np.float32)
                mask = np.load(str(mask_path).replace('.tif', '.npy')).astype(np.int64)

        # 데이터 증강
        if self.transform and HAS_ALBUMENTATION:
            image_hwc = image.transpose(1, 2, 0)
            transformed = self.transform(image=image_hwc, mask=mask)
            image = transformed['image'].transpose(2, 0, 1)
            mask = transformed['mask']

        return torch.from_numpy(image), torch.from_numpy(mask)


# =============================================================================
# 간단한 U-Net 구현 (smp 없을 때 사용)
# =============================================================================

class DoubleConv(nn.Module):
    """U-Net의 기본 블록: Conv-BN-ReLU x 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    """간단한 U-Net 구현"""

    def __init__(self, in_channels=13, num_classes=10):
        super().__init__()

        # 인코더
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # 바틀넥
        self.bottleneck = DoubleConv(512, 1024)

        # 디코더
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # 출력
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # 인코더
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 바틀넥
        b = self.bottleneck(self.pool(e4))

        # 디코더 + 스킵 연결
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


# =============================================================================
# 손실함수
# =============================================================================

class CombinedLoss(nn.Module):
    """Focal Loss + Dice Loss 조합"""

    def __init__(self, num_classes, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # Cross Entropy
        ce_loss = self.ce(pred, target)

        # Focal Loss 가중치
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        # Dice Loss
        pred_softmax = torch.softmax(pred, dim=1)
        target_onehot = torch.nn.functional.one_hot(target, self.num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        intersection = (pred_softmax * target_onehot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return 0.5 * focal_loss + 0.5 * dice_loss


# =============================================================================
# 평가 함수
# =============================================================================

def calculate_iou(pred, target, num_classes):
    """클래스별 IoU 계산"""
    ious = []

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()

        if union == 0:
            iou = float('nan')
        else:
            iou = float(intersection) / float(union)

        ious.append(iou)

    # mIoU: 유효한 클래스만 평균
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else 0.0

    return ious, miou


# =============================================================================
# 학습 함수
# =============================================================================

def train_model(model, train_loader, val_loader, epochs, device, lr=1e-4):
    """세그멘테이션 모델 학습"""

    model = model.to(device)
    criterion = CombinedLoss(num_classes=10)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_miou = 0.0

    for epoch in range(epochs):
        # 학습 단계
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # 검증 단계
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()

                all_preds.append(preds)
                all_targets.append(masks.numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        ious, miou = calculate_iou(
            all_preds.flatten(),
            all_targets.flatten(),
            num_classes=10
        )

        print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val mIoU={miou:.4f}')

        # 모델 저장
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), 'best_segmentation_model.pth')
            print(f'  => Best model saved! mIoU={miou:.4f}')

        scheduler.step()

    return best_miou


# =============================================================================
# 대규모 영상 추론 (타일링)
# =============================================================================

def predict_large_image(image_path, model, tile_size=256, overlap=64, device='cuda'):
    """타일링 기반 대규모 영상 추론"""

    if not HAS_RASTERIO:
        print("rasterio가 필요합니다.")
        return None

    model.eval()
    stride = tile_size - overlap

    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)
        image = np.clip(image / 10000.0, 0, 1)

        _, height, width = image.shape
        num_classes = 10

        # 결과 저장
        output = np.zeros((num_classes, height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)

        # 타일 순회
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)

                tile = image[:, y_start:y_end, x_start:x_end]

                # 패딩 (필요시)
                if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                    padded = np.zeros((tile.shape[0], tile_size, tile_size), dtype=np.float32)
                    padded[:, :tile.shape[1], :tile.shape[2]] = tile
                    tile = padded

                # 추론
                with torch.no_grad():
                    tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(device)
                    pred = model(tile_tensor)
                    pred = torch.softmax(pred, dim=1).cpu().numpy()[0]

                # 결과 누적
                h, w = y_end - y_start, x_end - x_start
                output[:, y_start:y_end, x_start:x_end] += pred[:, :h, :w]
                count[y_start:y_end, x_start:x_end] += 1

        # 평균화
        count = np.maximum(count, 1)
        output /= count[None, :, :]

        # 최종 예측
        prediction = output.argmax(axis=0).astype(np.uint8)

        return prediction


# =============================================================================
# 후처리 및 벡터화
# =============================================================================

def postprocess_and_vectorize(mask, transform, crs, min_area=100):
    """세그멘테이션 결과 후처리 및 벡터화"""

    if not HAS_GEOPANDAS or not HAS_RASTERIO:
        print("geopandas와 rasterio가 필요합니다.")
        return None

    from rasterio.features import shapes

    # 모폴로지 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # 벡터화
    geometries = []
    for geom, value in shapes(mask_clean, mask=(mask_clean > 0), transform=transform):
        poly = shape(geom)
        if poly.area >= min_area:
            geometries.append({'geometry': poly, 'class': int(value)})

    if not geometries:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(geometries, crs=crs)

    # 단순화
    gdf['geometry'] = gdf.geometry.simplify(tolerance=10.0)

    # 속성 계산
    gdf['area_m2'] = gdf.geometry.area
    gdf['perimeter_m'] = gdf.geometry.length

    return gdf


# =============================================================================
# 메인 함수
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='U-Net 토지피복 세그멘테이션')
    parser.add_argument('--data_dir', type=str, default='../data', help='데이터 디렉토리')
    parser.add_argument('--epochs', type=int, default=50, help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--tile_size', type=int, default=256, help='타일 크기')
    parser.add_argument('--num_bands', type=int, default=13, help='입력 밴드 수')
    parser.add_argument('--num_classes', type=int, default=10, help='클래스 수')

    args = parser.parse_args()

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 데이터 증강
    if HAS_ALBUMENTATION:
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
    else:
        train_transform = None

    # 데이터셋 및 데이터로더
    print('\n데이터셋 로드 중...')
    train_dataset = LandCoverDataset(
        args.data_dir, split='train',
        transform=train_transform,
        num_bands=args.num_bands,
        num_classes=args.num_classes
    )
    val_dataset = LandCoverDataset(
        args.data_dir, split='val',
        transform=None,
        num_bands=args.num_bands,
        num_classes=args.num_classes
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    print(f'  Train samples: {len(train_dataset)}')
    print(f'  Val samples: {len(val_dataset)}')

    # 모델 생성
    print('\n모델 생성 중...')
    if HAS_SMP:
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=args.num_bands,
            classes=args.num_classes,
        )
        print('  Using segmentation_models.pytorch U-Net')
    else:
        model = SimpleUNet(
            in_channels=args.num_bands,
            num_classes=args.num_classes
        )
        print('  Using SimpleUNet (smp not available)')

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total parameters: {total_params:,}')
    print(f'  Trainable parameters: {trainable_params:,}')

    # 학습
    print('\n학습 시작...')
    best_miou = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, device=device, lr=args.lr
    )

    print(f'\n학습 완료! Best mIoU: {best_miou:.4f}')

    # 결과 출력
    print('\n' + '=' * 60)
    print('6주차 보충자료 완료: U-Net 기반 토지피복 세그멘테이션')
    print('=' * 60)
    print(f'  모델: U-Net + ResNet-50 백본')
    print(f'  입력: {args.num_bands}채널 다중 스펙트럼 영상')
    print(f'  출력: {args.num_classes}클래스 세그멘테이션 마스크')
    print(f'  Best mIoU: {best_miou:.4f}')
    print(f'  저장된 모델: best_segmentation_model.pth')
    print('=' * 60)


if __name__ == '__main__':
    main()
