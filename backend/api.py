"""
api.py — FastAPI 백엔드

실행 방법:
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000

엔드포인트:
    POST /recommend  → 추천 장소 + 경로 반환
    GET  /health     → 서버 상태 확인
"""

import os
import logging
import requests
import urllib.parse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from inference import TravelRecommender, build_google_maps_url
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="도쿄 여행 추천 API")

# ─────────────────────────────────────────────
# CORS 설정 (프론트엔드 도메인 허용)
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # 배포 시 실제 프론트엔드 URL로 변경
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# 서버 시작 시 1회 로드 (요청마다 재로드 방지)
# ─────────────────────────────────────────────
recommender = TravelRecommender()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
DIRECTIONS_API_KEY = os.environ.get("DIRECTIONS_API_KEY", "")

# ─────────────────────────────────────────────
# 요청/응답 스키마
# ─────────────────────────────────────────────
class UserInput(BaseModel):
    category:       str = Field(..., example="관광",             description="관광 | 미식 | 쇼핑")
    companion_type: str = Field(..., example="연인과 떠나는 여행", description="혼자|연인과|친구와|부모님과 떠나는 여행")
    age:            int = Field(..., example=32,                 ge=1, le=100)
    n_places:       int = Field(..., example=3,                  ge=1, le=5)


# ─────────────────────────────────────────────
# 유틸 — Directions API 호출
# ─────────────────────────────────────────────
def call_directions_api(places: list[dict]) -> dict | None:
    """
    Google Maps Directions API를 호출해 경로 정보를 반환합니다.
    API 키가 없거나 호출 실패 시 None을 반환합니다.
    """
    if not GOOGLE_API_KEY:
        logger.warning("[Directions] GOOGLE_API_KEY 없음 → 폴백")
        return None
    
    if not DIRECTIONS_API_KEY:
        logger.warning("[Directions] DIRECTIONS_API_KEY -> 풀백")
        return None

    if len(places) < 2:
        logger.info("[Directions] 장소 1개 → 경로 없음")
        return None

    def coord(p: dict) -> str:
        return f"{p['latitude']},{p['longitude']}"

    waypts = "|".join(coord(p) for p in places[1:-1])

    params = {
        "origin":      coord(places[0]),
        "destination": coord(places[-1]),
        "key":         DIRECTIONS_API_KEY,
        "language":    "ko",
    }
    if waypts:
        params["waypoints"] = waypts

    try:
        res = requests.get(
            "https://maps.googleapis.com/maps/api/directions/json",
            params=params,
            timeout=5,
        )
        data = res.json()
    except Exception as e:
        logger.error(f"[Directions] 네트워크 오류: {e}")
        return None

    status = data.get("status")
    if status != "OK":
        # ★ 진단: 서버 로그에서 실제 실패 원인 확인 가능
        logger.error(f"[Directions] status={status} | error_message={data.get('error_message', '')}")
        return None

    route = data["routes"][0]
    legs  = route["legs"]
    polyline = route["overview_polyline"]["points"]
    logger.info(f"[Directions] 성공 | polyline 길이={len(polyline)}")

    return {
        "overview_polyline": polyline,
        "legs": [
            {
                "from":     leg["start_address"],
                "to":       leg["end_address"],
                "distance": leg["distance"]["text"],
                "duration": leg["duration"]["text"],
            }
            for leg in legs
        ],
        "total_distance": f"{sum(l['distance']['value'] for l in legs) / 1000:.1f} km",
        "total_duration": f"{sum(l['duration']['value'] for l in legs) // 60}분",
    }


def build_straight_line_path(places: list[dict]) -> str:
    """
    Directions API 없이 좌표만으로 직선 경로 파라미터를 생성합니다.
    Static Maps API의 &path= 형식으로 반환합니다.
    """
    coords = "|".join(f"{p['latitude']},{p['longitude']}" for p in places)
    return f"color:0x234F3B|weight:4|{coords}"


def build_static_map_url(places: list[dict], polyline: str | None, api_key: str) -> str:
    """
    Static Maps URL을 조립합니다.
    - polyline이 있으면 실제 도로 경로 사용
    - 없으면 좌표 직선 연결 폴백
    URL 길이가 8192자를 초과하면 polyline을 제거하고 직선으로 대체합니다.
    """
    base = "https://maps.googleapis.com/maps/api/staticmap?size=800x800&scale=2"

    # 마커
    markers = "".join(
        f"&markers=color:0x234F3B|label:{i+1}|{p['latitude']},{p['longitude']}"
        for i, p in enumerate(places)
    )

    # 경로 — polyline 우선, 없으면 직선
    # 흰색 테두리(굵게) + 진한 초록 실선을 겹쳐 가독성 확보
    if polyline:
        safe_poly = urllib.parse.quote(polyline)
        path  = f"&path=color:0xFFFFFFCC|weight:8|enc:{safe_poly}"   # 흰 외곽선
        path += f"&path=color:0x1A6B48FF|weight:5|enc:{safe_poly}"   # 진한 초록 실선
    else:
        straight      = build_straight_line_path(places)
        safe_straight = urllib.parse.quote(straight)
        path  = f"&path=color:0xFFFFFFCC|weight:8|{safe_straight}"
        path += f"&path=color:0x1A6B48FF|weight:5|{safe_straight}"

    key_param = f"&key={api_key}"
    url = base + markers + path + key_param

    # URL 길이 초과 시 polyline → 직선으로 강제 대체
    if len(url) > 8192 and polyline:
        logger.warning(f"[StaticMap] URL 길이 {len(url)} 초과 → 직선 경로로 대체")
        straight      = build_straight_line_path(places)
        safe_straight = urllib.parse.quote(straight)
        path  = f"&path=color:0xFFFFFFCC|weight:8|{safe_straight}"
        path += f"&path=color:0x1A6B48FF|weight:5|{safe_straight}"
        url = base + markers + path + key_param

    logger.info(f"[StaticMap] 최종 URL 길이={len(url)}, polyline 사용={bool(polyline)}")
    return url


# ─────────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(body: UserInput):
    # ① 모델 추론
    # predict()는 latitude, longitude 포함 DataFrame 반환
    try:
        result = recommender.predict(
            body.category, body.companion_type, body.n_places, body.age
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ② places 페이로드 조립
    places_payload = [
        {
            "rank":                 i + 1,
            "name":                 row["place_name"],
            "district":             row["district"],
            "mood":                 row["mood"],
            "google_rating":        float(row["google_rating"]),
            "stay_time_minutes":    int(row["stay_time_minutes"]),
            "predicted_score":      round(float(row["predicted_score"]), 2),
            "latitude":             float(row["latitude"]),
            "longitude":            float(row["longitude"]),
            "nearby_restaurants":   (
                recommender.get_nearby_restaurants(
                    float(row["latitude"]),
                    float(row["longitude"]),
                )
                if body.category in ["관광", "쇼핑"]
                else []
            ),
        }
        for i, (_, row) in enumerate(result.iterrows())
    ]

    # ③ Directions API 호출 시도
    route_payload = call_directions_api(places_payload)

    # ④ 좌표 기반 Google Maps 딥링크 URL (항상 생성)
    maps_url = build_google_maps_url(places_payload, api_key='')

    if route_payload is None:
        route_payload = {
            "overview_polyline": None,
            "legs":              [],
            "total_distance":    None,
            "total_duration":    None,
        }

    route_payload["maps_url"] = maps_url

    # ⑤ Static Maps URL 조립
    # - Directions API 성공 → 실제 도로 polyline 사용
    # - 실패 또는 URL 초과 → 좌표 직선 연결 폴백 (경로선은 항상 표시)
    static_map_url = build_static_map_url(
        places=places_payload,
        polyline=route_payload.get("overview_polyline"),
        api_key=GOOGLE_API_KEY,
    )

    return {
        "input":          body.model_dump(),
        "places":         places_payload,
        "route":          route_payload,
        "static_map_url": static_map_url,
        "maps_url":       maps_url,
    }