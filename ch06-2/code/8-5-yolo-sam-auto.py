#!/usr/bin/env python3
"""
8.5 실습: 자동화된 프롬프트 생성 (그리드 샘플링 + YOLO 연계)

이 스크립트는 두 가지 자동화된 프롬프트 생성 방법을 보여준다:
1. 그리드 샘플링: 이미지에 균등 분포 포인트를 배치
2. YOLO 연계: 탐지된 박스를 프롬프트로 사용

실행 예시:
  # 그리드 샘플링 모드
  python practice/chapter08/code/8-5-yolo-sam-auto.py \
    --image practice/chapter08/data/o41078a5.tif \
    --checkpoint practice/chapter08/models/sam_vit_b_01ec64.pth \
    --mode grid --grid-size 16 \
    --out practice/chapter08/data/output

  # YOLO 연계 모드
  python practice/chapter08/code/8-5-yolo-sam-auto.py \
    --image practice/chapter08/data/o41078a5.tif \
    --checkpoint practice/chapter08/models/sam_vit_b_01ec64.pth \
    --mode yolo \
    --out practice/chapter08/data/output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="자동화된 프롬프트 생성")
    p.add_argument("--image", type=str, required=True, help="입력 GeoTIFF")
    p.add_argument("--checkpoint", type=str, required=True, help="SAM 체크포인트")
    p.add_argument("--model-type", type=str, default="vit_b")
    p.add_argument(
        "--mode",
        type=str,
        default="grid",
        choices=["grid", "yolo", "amg"],
        help="프롬프트 생성 모드",
    )
    p.add_argument("--grid-size", type=int, default=16, help="그리드 크기 (grid 모드)")
    p.add_argument("--min-area", type=int, default=500, help="최소 마스크 픽셀 수")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="practice/chapter08/data/output")
    return p.parse_args()


def read_image_rgb(path: Path) -> tuple[np.ndarray, rasterio.DatasetReader]:
    """GeoTIFF를 RGB uint8로 읽기"""
    src = rasterio.open(path)
    arr = src.read()

    img = np.transpose(arr, (1, 2, 0))
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    img = img.astype(np.float32)
    lo, hi = np.nanpercentile(img, [2, 98])
    img = np.clip((img - lo) / max(1e-6, hi - lo), 0, 1)
    img_u8 = (img * 255).astype(np.uint8)

    return img_u8, src


def generate_grid_points(width: int, height: int, points_per_side: int) -> np.ndarray:
    """균등 분포 그리드 포인트 생성"""
    x = np.linspace(0, width - 1, points_per_side).astype(int)
    y = np.linspace(0, height - 1, points_per_side).astype(int)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    return points


def run_grid_sampling(
    image: np.ndarray,
    checkpoint: Path,
    model_type: str,
    device: str,
    grid_size: int,
    min_area: int,
) -> list[np.ndarray]:
    """그리드 포인트 기반 SAM 실행"""
    from segment_anything import SamPredictor, sam_model_registry

    h, w = image.shape[:2]
    points = generate_grid_points(w, h, grid_size)

    print(f"[그리드] {grid_size}x{grid_size} = {len(points)}개 포인트")

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    masks = []
    for i, (px, py) in enumerate(points):
        point_coords = np.array([[px, py]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int64)

        mask_outputs, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        best_mask = mask_outputs[best_idx]

        # 최소 면적 필터링
        if best_mask.sum() >= min_area:
            masks.append(best_mask)

        if (i + 1) % 50 == 0:
            print(f"  - {i + 1}/{len(points)} 처리 완료")

    return masks


def run_amg(
    image: np.ndarray,
    checkpoint: Path,
    model_type: str,
    device: str,
    min_area: int,
) -> list[np.ndarray]:
    """Automatic Mask Generation (AMG) 모드"""
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    print("[AMG] 자동 마스크 생성 모드")

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=min_area,
    )

    print("[AMG] 마스크 생성 중 (시간이 걸릴 수 있습니다)...")
    results = mask_generator.generate(image)

    masks = [r["segmentation"] for r in results]
    print(f"[AMG] 생성된 마스크: {len(masks)}개")

    return masks


def run_yolo_mode(
    image: np.ndarray,
    checkpoint: Path,
    model_type: str,
    device: str,
    min_area: int,
) -> list[np.ndarray]:
    """YOLO 탐지 → SAM 마스크 모드"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics가 설치되지 않았습니다.")
        print("설치: pip install ultralytics")
        return []

    from segment_anything import SamPredictor, sam_model_registry

    print("[YOLO] 객체 탐지 수행 중...")
    yolo = YOLO("yolov8n.pt")
    results = yolo(image, conf=0.25, device=device, verbose=False)

    boxes = []
    for result in results:
        if result.boxes is not None:
            for i in range(len(result.boxes)):
                xyxy = result.boxes.xyxy[i].cpu().numpy()
                boxes.append(xyxy)

    print(f"[YOLO] 탐지된 박스: {len(boxes)}개")

    if not boxes:
        return []

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    masks = []
    print("[SAM] 박스 프롬프트로 마스크 생성 중...")
    for box in boxes:
        mask_outputs, scores, _ = predictor.predict(
            box=box.astype(np.float32),
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        best_mask = mask_outputs[best_idx]

        if best_mask.sum() >= min_area:
            masks.append(best_mask)

    return masks


def merge_and_save(
    masks: list[np.ndarray],
    src: rasterio.DatasetReader,
    out_dir: Path,
    mode: str,
) -> None:
    """마스크 병합 및 저장"""
    if not masks:
        print("저장할 마스크가 없습니다.")
        return

    h, w = masks[0].shape

    # 인스턴스 마스크 (각 마스크에 고유 ID)
    instance_mask = np.zeros((h, w), dtype=np.uint16)
    for i, m in enumerate(masks, start=1):
        instance_mask[m.astype(bool)] = i

    # 바이너리 마스크 (합집합)
    binary_mask = np.any(np.stack(masks, axis=0), axis=0).astype(np.uint8)

    # 저장
    profile = src.profile.copy()

    out_instance = out_dir / f"sam_{mode}_instances.tif"
    profile.update(count=1, dtype="uint16", compress="deflate")
    with rasterio.open(out_instance, "w", **profile) as dst:
        dst.write(instance_mask, 1)
    print(f"[저장] 인스턴스 마스크: {out_instance}")

    out_binary = out_dir / f"sam_{mode}_binary.tif"
    profile.update(count=1, dtype="uint8", compress="deflate")
    with rasterio.open(out_binary, "w", **profile) as dst:
        dst.write(binary_mask, 1)
    print(f"[저장] 바이너리 마스크: {out_binary}")


def main() -> None:
    args = parse_args()
    img_path = Path(args.image)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[입력] 이미지: {img_path}")
    print(f"[모드] {args.mode}")

    image_rgb, src = read_image_rgb(img_path)
    print(f"- 크기: {image_rgb.shape}")

    if args.mode == "grid":
        masks = run_grid_sampling(
            image_rgb,
            ckpt_path,
            args.model_type,
            args.device,
            args.grid_size,
            args.min_area,
        )
    elif args.mode == "yolo":
        masks = run_yolo_mode(
            image_rgb,
            ckpt_path,
            args.model_type,
            args.device,
            args.min_area,
        )
    elif args.mode == "amg":
        masks = run_amg(
            image_rgb,
            ckpt_path,
            args.model_type,
            args.device,
            args.min_area,
        )
    else:
        raise ValueError(f"지원하지 않는 모드: {args.mode}")

    print(f"\n[결과] 유효 마스크 수: {len(masks)}")

    if masks:
        total_pixels = sum(m.sum() for m in masks)
        print(f"- 총 마스크 픽셀: {total_pixels:,}")

    merge_and_save(masks, src, out_dir, args.mode)

    src.close()
    print("\n[완료]")


if __name__ == "__main__":
    main()
