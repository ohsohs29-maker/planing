#!/usr/bin/env python3
"""
8.2 실습: samgeo 기본 사용법

segment-geospatial(samgeo) 라이브러리를 사용하여
GeoTIFF 영상에 SAM을 적용하고 결과를 벡터로 변환한다.

주의:
- samgeo는 segment-anything 위에 구축된 지리공간 특화 래퍼다.
- 입력/출력 모두 CRS와 transform이 보존된다.

실행 예시:
  python practice/chapter08/code/8-2-samgeo-basic.py \
    --image practice/chapter08/data/o41078a5.tif \
    --checkpoint practice/chapter08/models/sam_vit_b_01ec64.pth \
    --out practice/chapter08/data/output
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="samgeo 기본 사용법")
    p.add_argument("--image", type=str, required=True, help="입력 GeoTIFF 경로")
    p.add_argument("--checkpoint", type=str, required=True, help="SAM 체크포인트(.pth)")
    p.add_argument("--model-type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--device", type=str, default="cpu", help="cpu/cuda")
    p.add_argument("--out", type=str, default="practice/chapter08/data/output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    img_path = Path(args.image)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # samgeo 임포트 (선택적 의존성)
    try:
        from samgeo import SamGeo
    except ImportError:
        print("samgeo가 설치되지 않았습니다.")
        print("설치: pip install segment-geospatial")
        print("")
        print("samgeo 없이 기본 SAM 사용법을 보여드립니다:")
        demo_without_samgeo(img_path, ckpt_path, args.model_type, args.device, out_dir)
        return

    print("[samgeo] 초기화 중...")
    sam = SamGeo(
        model_type=args.model_type,
        checkpoint=str(ckpt_path),
        device=args.device,
    )

    out_mask = out_dir / "samgeo_mask.tif"
    out_vector = out_dir / "samgeo_mask.shp"

    print(f"[samgeo] 자동 세그멘테이션 실행: {img_path}")
    sam.generate(
        source=str(img_path),
        output=str(out_mask),
        batch=True,
        foreground=True,
    )
    print(f"- 마스크 저장: {out_mask}")

    print("[samgeo] 벡터 변환 중...")
    sam.tiff_to_vector(str(out_mask), str(out_vector))
    print(f"- 벡터 저장: {out_vector}")

    print("\n[완료] samgeo 기본 실행 완료")


def demo_without_samgeo(
    img_path: Path,
    ckpt_path: Path,
    model_type: str,
    device: str,
    out_dir: Path,
) -> None:
    """samgeo 없이 기본 SAM + rasterio로 동일한 작업 수행 (데모)"""
    import numpy as np
    import rasterio

    print("[데모] samgeo 없이 SAM 직접 사용")

    # 이미지 읽기
    with rasterio.open(img_path) as src:
        arr = src.read()
        profile = src.profile.copy()

    # (C, H, W) -> (H, W, C)
    img = np.transpose(arr, (1, 2, 0))
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    # 정규화
    img = img.astype(np.float32)
    lo, hi = np.nanpercentile(img, [2, 98])
    img = np.clip((img - lo) / max(1e-6, hi - lo), 0, 1)
    img_u8 = (img * 255).astype(np.uint8)

    print(f"- 이미지 크기: {img_u8.shape}")

    # SAM 로드 및 AMG 실행
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=str(ckpt_path))
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=500,
    )

    print("[데모] 자동 마스크 생성 중 (시간이 걸릴 수 있습니다)...")
    masks = mask_generator.generate(img_u8)
    print(f"- 생성된 마스크 수: {len(masks)}")

    # 마스크 병합 (인스턴스 ID로)
    h, w = img_u8.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint16)
    for i, m in enumerate(masks, start=1):
        combined[m["segmentation"]] = i

    # GeoTIFF로 저장
    out_mask = out_dir / "sam_amg_mask.tif"
    profile.update(count=1, dtype="uint16", compress="deflate")
    with rasterio.open(out_mask, "w", **profile) as dst:
        dst.write(combined, 1)
    print(f"- 마스크 저장: {out_mask}")

    # 통계
    total_pixels = int((combined > 0).sum())
    print(f"- 총 마스크 픽셀: {total_pixels:,}")


if __name__ == "__main__":
    main()
