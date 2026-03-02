#!/usr/bin/env python3
"""
10장 실습: LLM API를 활용한 자연어 기반 공간 쿼리 시스템
==========================================================

이 스크립트는 OpenAI GPT-4와 LangChain을 활용하여
자연어로 공간 분석을 수행하는 대화형 시스템을 구현합니다.

사용법:
    python 10-5-natural-language-gis.py

환경변수:
    OPENAI_API_KEY: OpenAI API 키

데이터:
    ../data/seoul_subway.geojson: 서울시 지하철역
    ../data/seoul_restaurants.geojson: 서울시 음식점
    ../data/seoul_districts.geojson: 서울시 행정구역
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import geopandas as gpd
import folium
from shapely.geometry import Point

# LangChain
from langchain_openai import ChatOpenAI
from langchain.agents import tool, create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 환경 설정
load_dotenv()
DATA_DIR = Path(__file__).parent.parent / "data"


# ============================================================
# 데이터 로드
# ============================================================

def load_data():
    """서울시 공간 데이터 로드"""
    global subway_stations, restaurants, districts

    # 데모 데이터 생성 (실제 파일이 없는 경우)
    if not (DATA_DIR / "seoul_subway.geojson").exists():
        print("데모 데이터 생성 중...")
        create_demo_data()

    subway_stations = gpd.read_file(DATA_DIR / "seoul_subway.geojson")
    restaurants = gpd.read_file(DATA_DIR / "seoul_restaurants.geojson")
    districts = gpd.read_file(DATA_DIR / "seoul_districts.geojson")

    # 좌표계 통일 (EPSG:5179 - Korea 2000)
    subway_stations = subway_stations.to_crs(epsg=5179)
    restaurants = restaurants.to_crs(epsg=5179)
    districts = districts.to_crs(epsg=5179)

    print(f"데이터 로드 완료:")
    print(f"  - 지하철역: {len(subway_stations)}개")
    print(f"  - 음식점: {len(restaurants)}개")
    print(f"  - 행정구: {len(districts)}개")


def create_demo_data():
    """데모용 가상 데이터 생성"""
    import numpy as np

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 서울시 주요 지하철역 (일부)
    stations_data = {
        'name': ['강남역', '신촌역', '홍대입구역', '서울역', '종로3가역',
                 '잠실역', '건대입구역', '신림역', '사당역', '왕십리역'],
        'line': ['2호선', '2호선', '2호선', '1호선', '1호선',
                 '2호선', '2호선', '2호선', '2호선', '2호선'],
        'geometry': [
            Point(127.0276, 37.4979), Point(126.9368, 37.5550),
            Point(126.9236, 37.5571), Point(126.9725, 37.5547),
            Point(126.9920, 37.5710), Point(127.1001, 37.5133),
            Point(127.0703, 37.5403), Point(126.9296, 37.4849),
            Point(126.9816, 37.4766), Point(127.0376, 37.5614)
        ]
    }
    subway_gdf = gpd.GeoDataFrame(stations_data, crs="EPSG:4326")
    subway_gdf.to_file(DATA_DIR / "seoul_subway.geojson", driver="GeoJSON")

    # 음식점 데이터 (가상)
    np.random.seed(42)
    n_restaurants = 500
    center_lat, center_lon = 37.5, 127.0
    restaurants_data = {
        'name': [f'음식점_{i}' for i in range(n_restaurants)],
        'type': np.random.choice(['한식', '중식', '일식', '양식', '카페'], n_restaurants),
        'geometry': [
            Point(center_lon + np.random.normal(0, 0.05),
                  center_lat + np.random.normal(0, 0.03))
            for _ in range(n_restaurants)
        ]
    }
    restaurants_gdf = gpd.GeoDataFrame(restaurants_data, crs="EPSG:4326")
    restaurants_gdf.to_file(DATA_DIR / "seoul_restaurants.geojson", driver="GeoJSON")

    # 행정구 데이터 (가상 - 단순화된 경계)
    from shapely.geometry import box
    districts_data = {
        'name': ['강남구', '서초구', '송파구', '강동구', '마포구'],
        'geometry': [
            box(127.0, 37.48, 127.1, 37.53),
            box(126.95, 37.46, 127.05, 37.51),
            box(127.08, 37.49, 127.15, 37.54),
            box(127.12, 37.52, 127.18, 37.57),
            box(126.88, 37.54, 126.96, 37.58)
        ]
    }
    districts_gdf = gpd.GeoDataFrame(districts_data, crs="EPSG:4326")
    districts_gdf.to_file(DATA_DIR / "seoul_districts.geojson", driver="GeoJSON")

    print("데모 데이터 생성 완료")


# ============================================================
# 공간 분석 도구 정의
# ============================================================

@tool
def find_nearby_pois(station_name: str, radius_meters: float, poi_type: str = "restaurant") -> str:
    """특정 지하철역 주변의 POI를 검색합니다.

    Args:
        station_name: 지하철역 이름 (예: '강남역')
        radius_meters: 검색 반경 (미터)
        poi_type: POI 유형 (기본값: 'restaurant')

    Returns:
        검색 결과 문자열
    """
    # 역 찾기
    station = subway_stations[subway_stations['name'].str.contains(station_name)]
    if len(station) == 0:
        return f"'{station_name}'을(를) 찾을 수 없습니다. 사용 가능한 역: {', '.join(subway_stations['name'].tolist()[:5])}..."

    # 버퍼 생성
    station_point = station.geometry.iloc[0]
    buffer = station_point.buffer(radius_meters)

    # POI 검색
    pois = restaurants[restaurants.geometry.within(buffer)]

    return f"{station_name} 반경 {radius_meters}m 내 음식점: {len(pois)}개"


@tool
def get_district_statistics(district_name: str) -> str:
    """특정 행정구의 통계 정보를 반환합니다.

    Args:
        district_name: 행정구 이름 (예: '강남구')

    Returns:
        행정구 통계 정보
    """
    district = districts[districts['name'].str.contains(district_name)]
    if len(district) == 0:
        return f"'{district_name}'을(를) 찾을 수 없습니다. 사용 가능한 구: {', '.join(districts['name'].tolist())}"

    area_km2 = district.geometry.area.iloc[0] / 1_000_000

    # 구 내 지하철역 수
    stations_in_district = gpd.sjoin(
        subway_stations,
        district[['geometry']],
        how='inner',
        predicate='within'
    )

    # 구 내 음식점 수
    restaurants_in_district = gpd.sjoin(
        restaurants,
        district[['geometry']],
        how='inner',
        predicate='within'
    )

    return (f"{district_name}: 면적 {area_km2:.2f}km², "
            f"지하철역 {len(stations_in_district)}개, "
            f"음식점 {len(restaurants_in_district)}개")


@tool
def calculate_distance(from_station: str, to_station: str) -> str:
    """두 지하철역 간 직선 거리를 계산합니다.

    Args:
        from_station: 출발역 이름
        to_station: 도착역 이름

    Returns:
        거리 정보
    """
    from_st = subway_stations[subway_stations['name'].str.contains(from_station)]
    to_st = subway_stations[subway_stations['name'].str.contains(to_station)]

    if len(from_st) == 0:
        return f"'{from_station}'을(를) 찾을 수 없습니다."
    if len(to_st) == 0:
        return f"'{to_station}'을(를) 찾을 수 없습니다."

    distance = from_st.geometry.iloc[0].distance(to_st.geometry.iloc[0])
    return f"{from_station} ↔ {to_station}: {distance:,.0f}m (직선거리)"


@tool
def list_stations() -> str:
    """사용 가능한 지하철역 목록을 반환합니다.

    Returns:
        지하철역 목록
    """
    stations = subway_stations['name'].tolist()
    return f"사용 가능한 지하철역 ({len(stations)}개): {', '.join(stations)}"


# ============================================================
# 에이전트 설정
# ============================================================

def create_gis_agent():
    """GIS 분석 에이전트 생성"""

    # 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 서울시 공간 데이터 분석 전문가입니다.
사용자의 질문에 답하기 위해 제공된 도구를 활용하세요.

사용 가능한 데이터:
- 서울시 지하철역 위치
- 서울시 음식점 위치
- 서울시 행정구역 경계

답변은 한국어로 제공하고, 수치는 명확하게 표시하세요.
도구 사용 결과를 그대로 전달하지 말고, 자연스러운 문장으로 답변하세요."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # LLM 설정
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("경고: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("데모 모드로 실행합니다 (도구 직접 호출)")
        return None

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # 도구 목록
    tools = [find_nearby_pois, get_district_statistics, calculate_distance, list_stations]

    # 에이전트 생성
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


# ============================================================
# 시각화
# ============================================================

def visualize_buffer_result(station_name: str, radius_meters: float) -> folium.Map:
    """버퍼 분석 결과를 지도로 시각화"""

    station = subway_stations[subway_stations['name'].str.contains(station_name)]
    if len(station) == 0:
        print(f"'{station_name}'을 찾을 수 없습니다.")
        return None

    # WGS84로 변환하여 지도 생성
    station_4326 = station.to_crs(epsg=4326)
    center = station_4326.geometry.iloc[0]

    m = folium.Map(location=[center.y, center.x], zoom_start=15)

    # 버퍼 영역
    folium.Circle(
        location=[center.y, center.x],
        radius=radius_meters,
        color='blue',
        fill=True,
        fill_opacity=0.2,
        popup=f"{station_name} 반경 {radius_meters}m"
    ).add_to(m)

    # 역 마커
    folium.Marker(
        location=[center.y, center.x],
        popup=station_name,
        icon=folium.Icon(color='red', icon='subway', prefix='fa')
    ).add_to(m)

    # 버퍼 내 음식점 마커
    station_point = station.geometry.iloc[0]
    buffer = station_point.buffer(radius_meters)
    pois_in_buffer = restaurants[restaurants.geometry.within(buffer)]
    pois_4326 = pois_in_buffer.to_crs(epsg=4326)

    for _, poi in pois_4326.iterrows():
        folium.CircleMarker(
            location=[poi.geometry.y, poi.geometry.x],
            radius=3,
            color='green',
            fill=True,
            popup=poi.get('name', '음식점')
        ).add_to(m)

    return m


# ============================================================
# 메인 함수
# ============================================================

def chat_with_gis(agent_executor):
    """대화형 GIS 쿼리 시스템"""
    print("\n" + "=" * 50)
    print("서울시 공간 분석 챗봇")
    print("=" * 50)
    print("\n질문 예시:")
    print("  - '강남역 500m 내 음식점 수는?'")
    print("  - '강남구 통계 알려줘'")
    print("  - '강남역에서 신촌역까지 거리는?'")
    print("  - '지하철역 목록'")
    print("\n종료하려면 'quit' 입력")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n질문: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("종료합니다.")
                break

            if agent_executor:
                response = agent_executor.invoke({"input": user_input})
                print(f"\n답변: {response['output']}")
            else:
                # 데모 모드: 직접 도구 호출
                demo_response(user_input)

        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"\n오류 발생: {e}")


def demo_response(user_input: str):
    """API 없이 데모 응답 생성"""
    user_input = user_input.lower()

    if '목록' in user_input or '역' in user_input:
        print(f"\n답변: {list_stations.invoke({})}")
    elif '거리' in user_input:
        print(f"\n답변: {calculate_distance.invoke({'from_station': '강남역', 'to_station': '신촌역'})}")
    elif '통계' in user_input or '구' in user_input:
        print(f"\n답변: {get_district_statistics.invoke({'district_name': '강남구'})}")
    elif '음식점' in user_input or '주변' in user_input:
        print(f"\n답변: {find_nearby_pois.invoke({'station_name': '강남역', 'radius_meters': 500.0, 'poi_type': 'restaurant'})}")
    else:
        print("\n답변: 질문을 이해하지 못했습니다. 다시 시도해 주세요.")


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("10장 실습: 자연어 기반 공간 쿼리 시스템")
    print("=" * 50)

    # 데이터 로드
    load_data()

    # 에이전트 생성
    agent_executor = create_gis_agent()

    # 대화형 인터페이스 실행
    chat_with_gis(agent_executor)


if __name__ == "__main__":
    main()
