"""
13.5 실습: Google Earth Engine에서 NDVI 시계열 분석

이 스크립트는 Google Earth Engine Python API를 사용하여
관심 지역(서울)의 NDVI 시계열을 분석한다.

실행 전 준비사항:
1. Earth Engine 계정 등록: https://earthengine.google.com/
2. 인증: earthengine authenticate
3. 의존성 설치: pip install -r requirements.txt

사용법:
    python 13-5-gee-ndvi-timeseries.py

출력:
    - practice/chapter13/data/output/seoul_ndvi_monthly_2024.csv
    - practice/chapter13/data/output/seoul_ndvi_timeseries.png
    - 콘솔에 월별 NDVI 통계 출력
"""

import ee
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ========================================
# 1. 환경 설정 및 Earth Engine 초기화
# ========================================

def initialize_ee():
    """Earth Engine 초기화"""
    try:
        # 프로젝트 ID 설정 (사용자 프로젝트로 변경 필요)
        # ee.Initialize(project='your-project-id')
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        print("[INFO] Earth Engine 초기화 완료")
    except Exception as e:
        print(f"[WARNING] 초기화 실패, 인증 시도: {e}")
        ee.Authenticate()
        try:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        except Exception:
            # 프로젝트 지정 없이 기본 초기화 시도
            ee.Initialize()
        print("[INFO] 인증 및 초기화 완료")

# ========================================
# 2. 관심 지역(AOI) 및 기간 설정
# ========================================

# 서울 중심부 (종로구~강남구 일대)
AOI_BOUNDS = [126.90, 37.48, 127.10, 37.58]  # [min_lon, min_lat, max_lon, max_lat]
START_DATE = "2024-01-01"
END_DATE = "2024-12-01"
CLOUD_THRESHOLD = 20  # 구름 비율 20% 이하 영상만 사용

# 출력 경로
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========================================
# 3. NDVI 계산 함수
# ========================================

def add_ndvi(image):
    """
    Sentinel-2 영상에 NDVI 밴드를 추가한다.

    NDVI = (NIR - Red) / (NIR + Red)
         = (B8 - B4) / (B8 + B4)

    Args:
        image: ee.Image - Sentinel-2 영상

    Returns:
        ee.Image - NDVI 밴드가 추가된 영상
    """
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)


def mask_clouds(image):
    """
    SCL 밴드를 사용하여 구름을 마스킹한다.

    SCL 값:
        0: No data
        1: Saturated or defective
        2: Dark area pixels
        3: Cloud shadows
        4: Vegetation
        5: Bare soils
        6: Water
        7: Unclassified
        8: Cloud medium probability
        9: Cloud high probability
        10: Thin cirrus
        11: Snow/Ice

    Args:
        image: ee.Image - Sentinel-2 L2A 영상

    Returns:
        ee.Image - 구름이 마스킹된 영상
    """
    scl = image.select('SCL')
    # 구름(8,9), 구름그림자(3), 눈/얼음(11) 제외
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(11))
    return image.updateMask(mask)

# ========================================
# 4. 데이터 수집 및 처리
# ========================================

def get_sentinel2_collection(aoi, start_date, end_date, cloud_threshold):
    """
    조건에 맞는 Sentinel-2 L2A 컬렉션을 반환한다.

    Args:
        aoi: ee.Geometry - 관심 지역
        start_date: str - 시작 날짜 (YYYY-MM-DD)
        end_date: str - 종료 날짜 (YYYY-MM-DD)
        cloud_threshold: int - 최대 구름 비율 (%)

    Returns:
        ee.ImageCollection - 필터링된 Sentinel-2 컬렉션
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .map(mask_clouds)
        .map(add_ndvi)
    )
    return collection


def compute_monthly_ndvi(collection, aoi, year=2024):
    """
    월별 NDVI 평균을 계산한다.

    Args:
        collection: ee.ImageCollection - NDVI가 포함된 컬렉션
        aoi: ee.Geometry - 관심 지역
        year: int - 분석 연도

    Returns:
        list - 월별 NDVI 통계 딕셔너리 리스트
    """
    results = []

    for month in range(1, 13):
        # 해당 월의 영상 필터링
        start = f"{year}-{month:02d}-01"
        if month == 12:
            end = f"{year + 1}-01-01"
        else:
            end = f"{year}-{month + 1:02d}-01"

        monthly = collection.filterDate(start, end)
        count = monthly.size().getInfo()

        if count > 0:
            # 월별 합성 (중앙값)
            composite = monthly.select('NDVI').median()

            # 통계 계산
            stats = composite.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.stdDev(), sharedInputs=True
                ).combine(
                    ee.Reducer.minMax(), sharedInputs=True
                ),
                geometry=aoi,
                scale=10,
                maxPixels=1e9
            ).getInfo()

            results.append({
                'year': year,
                'month': month,
                'date': f"{year}-{month:02d}-15",
                'ndvi_mean': stats.get('NDVI_mean'),
                'ndvi_std': stats.get('NDVI_stdDev'),
                'ndvi_min': stats.get('NDVI_min'),
                'ndvi_max': stats.get('NDVI_max'),
                'image_count': count
            })
            print(f"  {year}-{month:02d}: NDVI 평균={stats.get('NDVI_mean'):.4f}, "
                  f"영상 {count}개")
        else:
            results.append({
                'year': year,
                'month': month,
                'date': f"{year}-{month:02d}-15",
                'ndvi_mean': None,
                'ndvi_std': None,
                'ndvi_min': None,
                'ndvi_max': None,
                'image_count': 0
            })
            print(f"  {year}-{month:02d}: 영상 없음")

    return results

# ========================================
# 5. 시각화
# ========================================

def plot_ndvi_timeseries(df, output_path):
    """
    NDVI 시계열 그래프를 생성한다.

    Args:
        df: pd.DataFrame - 월별 NDVI 데이터
        output_path: Path - 출력 파일 경로
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # 유효한 데이터만 필터링
    valid = df.dropna(subset=['ndvi_mean'])

    if len(valid) == 0:
        print("[WARNING] 유효한 NDVI 데이터가 없습니다.")
        return

    # 날짜 변환
    dates = pd.to_datetime(valid['date'])

    # NDVI 평균 플롯
    ax.plot(dates, valid['ndvi_mean'], 'g-o', linewidth=2,
            markersize=8, label='NDVI Mean')

    # 표준편차 범위 표시
    if 'ndvi_std' in valid.columns:
        ax.fill_between(
            dates,
            valid['ndvi_mean'] - valid['ndvi_std'],
            valid['ndvi_mean'] + valid['ndvi_std'],
            alpha=0.2, color='green', label='±1 Std Dev'
        )

    # 그래프 설정
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('NDVI', fontsize=12)
    ax.set_title('Seoul NDVI Time Series (2024)\nSentinel-2 L2A', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # x축 포맷
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 시계열 그래프 저장: {output_path}")

# ========================================
# 6. 메인 실행
# ========================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("13.5 실습: Google Earth Engine NDVI 시계열 분석")
    print("=" * 60)

    # 1. Earth Engine 초기화
    print("\n[1/4] Earth Engine 초기화...")
    initialize_ee()

    # 2. AOI 설정
    print("\n[2/4] 관심 지역 설정...")
    aoi = ee.Geometry.Rectangle(AOI_BOUNDS)
    print(f"  AOI: {AOI_BOUNDS}")
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print(f"  구름 임계값: {CLOUD_THRESHOLD}%")

    # 3. 데이터 수집 및 처리
    print("\n[3/4] Sentinel-2 데이터 수집 및 NDVI 계산...")
    collection = get_sentinel2_collection(aoi, START_DATE, END_DATE, CLOUD_THRESHOLD)
    total_images = collection.size().getInfo()
    print(f"  총 영상 수: {total_images}개")

    print("\n  월별 NDVI 계산:")
    monthly_stats = compute_monthly_ndvi(collection, aoi)

    # 4. 결과 저장
    print("\n[4/4] 결과 저장...")
    df = pd.DataFrame(monthly_stats)

    # CSV 저장
    csv_path = OUTPUT_DIR / "seoul_ndvi_monthly_2024.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  CSV 저장: {csv_path}")

    # 시각화
    png_path = OUTPUT_DIR / "seoul_ndvi_timeseries.png"
    plot_ndvi_timeseries(df, png_path)

    # 요약 출력
    print("\n" + "=" * 60)
    print("분석 결과 요약")
    print("=" * 60)

    valid = df.dropna(subset=['ndvi_mean'])
    if len(valid) > 0:
        print(f"  분석 기간: {START_DATE} ~ {END_DATE}")
        print(f"  유효 월 수: {len(valid)}개월")
        print(f"  총 영상 수: {df['image_count'].sum()}개")
        print(f"  연평균 NDVI: {valid['ndvi_mean'].mean():.4f}")
        print(f"  최대 NDVI: {valid['ndvi_mean'].max():.4f} ({valid.loc[valid['ndvi_mean'].idxmax(), 'date']})")
        print(f"  최소 NDVI: {valid['ndvi_mean'].min():.4f} ({valid.loc[valid['ndvi_mean'].idxmin(), 'date']})")

    print("\n산출물:")
    print(f"  - {csv_path}")
    print(f"  - {png_path}")
    print("\n완료!")


if __name__ == "__main__":
    main()
