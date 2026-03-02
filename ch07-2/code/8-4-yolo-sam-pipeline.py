#!/usr/bin/env python3
"""
8.4 실습: YOLO → SAM 파이프라인

객체 탐지 모델(YOLO)로 바운딩 박스를 얻고,
SAM의 Box 프롬프트로 정밀한 마스크를 생성하는 파이프라인.

이 접근법의 장점:
- 탐지 모델이 "어디에 무엇이 있는지" 결정
- SAM이 "정확한 경계"를 결정
- 클래스 정보가 마스크에 자동 부여됨

실행 예시:
  python practice/chapter08/code/8-4-yolo-sam-pipeline.py \
    --image practice/chapter08/data/o41078a5.tif \
    --sam-checkpoint practice/chapter08/models/sam_vit_b_01ec64.pth \
    --out practice/chapter08/data/output

주의:
- YOLO는 자연 사진에 학습되어 있어 위성영상에서 성능이 낮을 수 있음
- 위성영상 전용 탐지 모델로 교체하면 성능 향상 가능
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rasterio


class Detection(NamedTuple):
    """YOLO 탐지 결과"""

    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO → SAM 파이프라인")
    p.add_argument("--image", type=str, required=True, help="입력 GeoTIFF")
    p.add_argument("--sam-checkpoint", type=str, required=True, help="SAM 체크포인트")
    p.add_argument("--sam-model-type", type=str, default="vit_b")
    p.add_argument("--yolo-model", type=str, default="yolov8n.pt", help="YOLO 모델")
    p.add_argument("--conf-threshold", type=float, default=0.25, help="탐지 신뢰도 임계값")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="practice/chapter08/data/output")
    return p.parse_args()


def read_image_rgb(path: Path) -> tuple[np.ndarray, rasterio.DatasetReader]:
    """GeoTIFF를 RGB uint8로 읽기"""
    src = rasterio.open(path)
    arr = src.read()

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

    return img_u8, src


def run_yolo_detection(
    image: np.ndarray,
    model_path: str,
    conf_threshold: float,
    device: str,
) -> list[Detection]:
    """YOLO로 객체 탐지 수행"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics가 설치되지 않았습니다.")
        print("설치: pip install ultralytics")
        return []

    print(f"[YOLO] 모델 로드: {model_path}")
    model = YOLO(model_path)

    print("[YOLO] 탐지 수행 중...")
    results = model(image, conf=conf_threshold, device=device, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            cls_id = int(boxes.cls[i].cpu().numpy())
            conf = float(boxes.conf[i].cpu().numpy())
            cls_name = result.names.get(cls_id, f"class_{cls_id}")

            detections.append(
                Detection(
                    box=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                )
            )

    return detections


def run_sam_with_boxes(
    image: np.ndarray,
    detections: list[Detection],
    checkpoint: Path,
    model_type: str,
    device: str,
) -> list[dict]:
    """SAM으로 각 박스에 대해 마스크 생성"""
    from segment_anything import SamPredictor, sam_model_registry

    print(f"[SAM] 모델 로드: {checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    results = []
    print(f"[SAM] {len(detections)}개 박스에 대해 마스크 생성 중...")

    for i, det in enumerate(detections):
        box = np.array(det.box, dtype=np.float32)
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)

        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx]
        best_score = float(scores[best_idx])

        results.append(
            {
                "detection": det,
                "mask": best_mask,
                "sam_score": best_score,
                "mask_pixels": int(best_mask.sum()),
            }
        )

        if (i + 1) % 10 == 0:
            print(f"  - {i + 1}/{len(detections)} 완료")

    return results


def save_results(
    results: list[dict],
    src: rasterio.DatasetReader,
    out_dir: Path,
) -> None:
    """결과 저장 (마스크 GeoTIFF + 메타데이터)"""
    import json

    if not results:
        print("저장할 결과가 없습니다.")
        return

    h, w = results[0]["mask"].shape

    # 인스턴스 마스크 생성 (각 객체에 고유 ID)
    instance_mask = np.zeros((h, w), dtype=np.uint16)
    for i, r in enumerate(results, start=1):
        instance_mask[r["mask"]] = i

    # GeoTIFF 저장
    out_mask = out_dir / "yolo_sam_instances.tif"
    profile = src.profile.copy()
    profile.update(count=1, dtype="uint16", compress="deflate")
    with rasterio.open(out_mask, "w", **profile) as dst:
        dst.write(instance_mask, 1)
    print(f"[저장] 인스턴스 마스크: {out_mask}")

    # 메타데이터 JSON
    metadata = []
    for i, r in enumerate(results, start=1):
        det = r["detection"]
        metadata.append(
            {
                "instance_id": i,
                "class_id": det.class_id,
                "class_name": det.class_name,
                "yolo_confidence": det.confidence,
                "sam_score": r["sam_score"],
                "mask_pixels": r["mask_pixels"],
                "box": list(det.box),
            }
        )

    out_meta = out_dir / "yolo_sam_metadata.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[저장] 메타데이터: {out_meta}")


def main() -> None:
    args = parse_args()
    img_path = Path(args.image)
    sam_ckpt = Path(args.sam_checkpoint)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[입력] 이미지: {img_path}")
    image_rgb, src = read_image_rgb(img_path)
    print(f"- 크기: {image_rgb.shape}")

    # YOLO 탐지
    detections = run_yolo_detection(
        image_rgb,
        args.yolo_model,
        args.conf_threshold,
        args.device,
    )
    print(f"\n[YOLO 결과] 탐지된 객체: {len(detections)}개")

    if not detections:
        print("탐지된 객체가 없습니다. 종료합니다.")
        print("참고: YOLO는 자연 사진에 학습되어 위성영상에서 성능이 낮을 수 있습니다.")
        src.close()
        return

    # 클래스별 통계
    class_counts: dict[str, int] = {}
    for det in detections:
        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  - {cls_name}: {count}개")

    # SAM 마스크 생성
    results = run_sam_with_boxes(
        image_rgb,
        detections,
        sam_ckpt,
        args.sam_model_type,
        args.device,
    )

    # 결과 저장
    save_results(results, src, out_dir)

    # 통계
    total_pixels = sum(r["mask_pixels"] for r in results)
    avg_sam_score = np.mean([r["sam_score"] for r in results])
    print(f"\n[통계]")
    print(f"- 총 마스크 픽셀: {total_pixels:,}")
    print(f"- 평균 SAM 점수: {avg_sam_score:.4f}")

    src.close()
    print("\n[완료]")


if __name__ == "__main__":
    main()
