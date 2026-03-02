#!/usr/bin/env python3
"""
8.5 실습: SAM을 활용한 지리공간 영상 세그멘테이션 (Box/Point 프롬프트)

이 스크립트는 GeoTIFF(위성/항공영상)를 입력으로 받아,
SAM(Segment Anything Model)으로 프롬프트 기반 세그멘테이션을 수행하고
마스크를 GeoTIFF로 저장한다.

중요:
- 본 실습은 "자동 추출"이 아니라 프롬프트 기반 세그멘테이션을 다룬다.
- 결과 마스크는 운영용 정답 지도가 아니다(라벨/검증이 별도로 필요).

실행 예시:
  python practice/chapter08/code/8-0-download-assets.py
  python practice/chapter08/code/8-5-sam-geospatial-segmentation.py \
    --image practice/chapter08/data/o41078a5.tif \
    --checkpoint practice/chapter08/models/sam_vit_b_01ec64.pth \
    --prompt box --box 130 110 310 290 \
    --out practice/chapter08/data/output
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio


@dataclass(frozen=True)
class Prompt:
    kind: str
    box: tuple[int, int, int, int] | None = None
    points: list[tuple[int, int, int]] | None = None  # (x, y, label)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM 지리공간 세그멘테이션(GeoTIFF)")
    p.add_argument("--image", type=str, required=True, help="입력 GeoTIFF 경로(RGB 또는 1밴드)")
    p.add_argument("--checkpoint", type=str, required=True, help="SAM 체크포인트(.pth) 경로")
    p.add_argument("--model-type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--device", type=str, default="cpu", help="cpu/cuda/mps (가용한 것 사용)")
    p.add_argument("--out", type=str, default="practice/chapter08/data/output", help="출력 디렉토리")

    p.add_argument("--prompt", type=str, default="box", choices=["box", "point"], help="프롬프트 종류")
    p.add_argument("--box", type=int, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument(
        "--point",
        type=int,
        nargs=3,
        action="append",
        default=None,
        metavar=("X", "Y", "LABEL"),
        help="포인트 프롬프트(여러 번 지정 가능). LABEL: 1(전경), 0(배경)",
    )
    p.add_argument("--multimask", action="store_true", help="다중 마스크 생성(최고 score 선택)")
    return p.parse_args()


def read_rgb_u8(path: Path) -> tuple[np.ndarray, rasterio.DatasetReader]:
    src = rasterio.open(path)
    arr = src.read()
    if arr.ndim != 3:
        raise ValueError("입력 래스터 차원이 예상과 다릅니다.")
    # (bands, H, W) -> (H, W, bands)
    img = np.transpose(arr, (1, 2, 0))

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    img = img.astype(np.float32)
    lo = np.nanpercentile(img, 2)
    hi = np.nanpercentile(img, 98)
    img = np.clip((img - lo) / max(1e-6, (hi - lo)), 0, 1)
    img_u8 = (img * 255).astype(np.uint8)

    # OpenCV는 BGR을 기대하지만, SAM은 RGB를 사용하므로 그대로 유지한다.
    return img_u8, src


def build_prompt(args: argparse.Namespace) -> Prompt:
    if args.prompt == "box":
        if not args.box:
            raise ValueError("--prompt box에는 --box X0 Y0 X1 Y1가 필요합니다.")
        x0, y0, x1, y1 = args.box
        return Prompt(kind="box", box=(x0, y0, x1, y1))

    if args.prompt == "point":
        if not args.point:
            raise ValueError("--prompt point에는 --point X Y LABEL을 1개 이상 지정해야 합니다.")
        pts = [(int(x), int(y), int(lbl)) for x, y, lbl in args.point]
        return Prompt(kind="point", points=pts)

    raise ValueError(f"지원하지 않는 프롬프트: {args.prompt}")


def run_sam(
    image_rgb: np.ndarray,
    checkpoint: Path,
    model_type: str,
    device: str,
    prompt: Prompt,
    multimask: bool,
) -> tuple[np.ndarray, float]:
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    if prompt.kind == "box":
        x0, y0, x1, y1 = prompt.box or (0, 0, 0, 0)
        box = np.array([x0, y0, x1, y1], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=box, multimask_output=multimask)
    else:
        pts = prompt.points or []
        point_coords = np.array([[x, y] for x, y, _ in pts], dtype=np.float32)
        point_labels = np.array([lbl for _, _, lbl in pts], dtype=np.int64)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask,
        )

    best = int(np.argmax(scores))
    mask = masks[best].astype(np.uint8)  # (H, W) 0/1
    return mask, float(scores[best])


def write_mask_geotiff(out_path: Path, mask: np.ndarray, src: rasterio.DatasetReader) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = src.meta.copy()
    meta.update(count=1, dtype="uint8", compress="deflate")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask.astype(np.uint8), 1)


def summarize(mask: np.ndarray, src: rasterio.DatasetReader) -> dict[str, float]:
    n = int(mask.sum())
    pixel_area = float(abs(src.transform.a * src.transform.e))
    area = n * pixel_area
    return {"mask_pixels": float(n), "pixel_area": pixel_area, "mask_area": area}


def main() -> None:
    args = parse_args()
    img_path = Path(args.image)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(args)
    image_rgb, src = read_rgb_u8(img_path)

    mask, score = run_sam(
        image_rgb=image_rgb,
        checkpoint=ckpt_path,
        model_type=args.model_type,
        device=args.device,
        prompt=prompt,
        multimask=args.multimask,
    )

    out_mask = out_dir / f"sam_mask_{prompt.kind}.tif"
    write_mask_geotiff(out_mask, mask, src)

    stats = summarize(mask, src)
    print("\n[입력]")
    print(f"- image: {img_path}")
    print(f"- size: {src.width}x{src.height}, bands={src.count}")
    print(f"- crs: {src.crs}")

    print("\n[프롬프트]")
    if prompt.kind == "box":
        print(f"- box: {prompt.box}")
    else:
        print(f"- points: {prompt.points}")

    print("\n[결과]")
    print(f"- score: {score:.4f}")
    print(f"- mask_pixels: {int(stats['mask_pixels']):,}")
    print(f"- pixel_area: {stats['pixel_area']:.6f} (CRS 단위^2)")
    print(f"- mask_area: {stats['mask_area']:.3f} (CRS 단위^2)")
    print(f"- saved: {out_mask}")

    # 간단한 시각화 PNG(비지리공간)도 저장: 마스크 오버레이
    overlay = image_rgb.copy()
    overlay[mask.astype(bool)] = (0.5 * overlay[mask.astype(bool)] + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    out_png = out_dir / f"sam_overlay_{prompt.kind}.png"
    cv2.imwrite(str(out_png), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"- saved: {out_png}")

    src.close()


if __name__ == "__main__":
    main()

