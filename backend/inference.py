import math
import torch
import pandas as pd
import numpy as np
import urllib.parse
from dataset import load_preprocessors
from model import AdvancedRecommender
import os

# ─────────────────────────────────────────────
# 0. 상수 정의
# ─────────────────────────────────────────────

VALID_CATEGORIES = ['관광', '미식', '쇼핑']
VALID_COMPANION_TYPES = [
    '혼자 떠나는 여행',
    '연인과 떠나는 여행',
    '친구와 떠나는 여행',
    '부모님과 떠나는 여행',
]
MAX_PLACES  = 5
DEFAULT_AGE = 22

# 거리 페널티 가중치: 1km 멀어질 때마다 예측 점수에서 차감할 값
# 예측 점수 범위(~3점)와 도쿄 장소 간 거리 범위(~6km)를 고려해 0.5로 설정
ALPHA = 0.5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')


# ─────────────────────────────────────────────
# 1. 유틸 함수
# ─────────────────────────────────────────────

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 좌표 사이의 거리(km)를 반환합니다."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def build_google_maps_url(places: list[dict], api_key: str = '') -> str:
    """
    추천된 장소 순서대로 Google Maps Directions URL을 생성합니다.
    장소명 대신 위경도 좌표를 사용해 검색 오류를 방지합니다.

    Parameters
    ----------
    places  : [{'name': str, 'latitude': float, 'longitude': float}, ...]
    api_key : Google Maps API 키 (없으면 브라우저용 공개 URL 반환)

    Returns
    -------
    str : Google Maps URL
    """
    if len(places) < 2:
        raise ValueError("경로 생성을 위해 장소가 2개 이상 필요합니다.")

    def coord(p: dict) -> str:
        return urllib.parse.quote(f"{p['latitude']},{p['longitude']}")

    if api_key:
        base        = 'https://maps.googleapis.com/maps/api/directions/json'
        origin      = coord(places[0])
        destination = coord(places[-1])
        waypoints   = '|'.join(coord(p) for p in places[1:-1])
        url = f"{base}?origin={origin}&destination={destination}"
        if waypoints:
            url += f"&waypoints={waypoints}"
        url += f"&key={api_key}&language=ko"
    else:
        base = 'https://www.google.com/maps/dir/'
        url  = base + '/'.join(coord(p) for p in places)

    return url


# ─────────────────────────────────────────────
# 2. 추천 엔진 클래스
# ─────────────────────────────────────────────

class TravelRecommender:
    """
    학습된 모델과 전처리 객체를 로드하고
    사용자 입력에 따라 장소 추천 결과를 반환합니다.

    추론 방식:
        1. 선택한 카테고리의 전체 장소에 모델 예측 점수 부여
        2. Greedy 방식으로 동선을 고려해 N개 장소 선택
           - 1순위: 예측 점수 최고 장소
           - 이후: 조정 점수(예측 점수 - alpha × 직전 장소까지 거리) 최고 장소
    """

    def __init__(
        self,
        model_path:        str = 'best_recommender.pth',
        preprocessor_path: str = 'preprocessors.pkl',
        places_path:       str = os.path.join(DATA_DIR, 'places.csv'),
    ):
        pre = load_preprocessors(preprocessor_path)

        self.label_encoders = pre['label_encoders']
        self.scaler         = pre['scaler']
        self.target_scaler  = pre['target_scaler']
        self.cat_features   = pre['cat_features']
        self.num_features   = pre['num_features']

        self.places = pd.read_csv(places_path, encoding='utf-8-sig')

        vocab_sizes = {
            col: len(self.label_encoders[col].classes_)
            for col in self.cat_features
        }
        self.model = AdvancedRecommender(vocab_sizes, len(self.num_features))
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

        print("✅ 모델 및 전처리 객체 로드 완료")

    # ── 입력 검증 ──────────────────────────────

    @staticmethod
    def validate_inputs(category: str, companion_type: str, n_places: int):
        if category not in VALID_CATEGORIES:
            raise ValueError(f"category는 {VALID_CATEGORIES} 중 하나여야 합니다.")
        if companion_type not in VALID_COMPANION_TYPES:
            raise ValueError(f"companion_type은 {VALID_COMPANION_TYPES} 중 하나여야 합니다.")
        if not (1 <= n_places <= MAX_PLACES):
            raise ValueError(f"n_places는 1~{MAX_PLACES} 사이여야 합니다.")

    # ── 점수 산출 ──────────────────────────────

    def _score_candidates(
        self,
        candidates:     pd.DataFrame,
        category:       str,
        companion_type: str,
        age:            int,
    ) -> np.ndarray:
        """후보 장소 전체에 대해 모델 예측 점수(원본 스케일)를 반환합니다."""

        input_df = pd.DataFrame({
            'category':            category,
            'companion_type':      companion_type,
            'place_name':          candidates['place_name'],
            'mood':                candidates['mood'],
            'district':            candidates['district'],
            'crowdedness':         candidates['crowdedness'],
            'mobility_preference': candidates['mobility_preference'],
            'revisit':             'Y',
            'age':                 age,
            'google_rating':       candidates['google_rating'],
            'stay_time_minutes':   candidates['stay_time_minutes'],
        })

        for col in self.cat_features:
            input_df[col] = self.label_encoders[col].transform(
                input_df[col].astype(str)
            )

        input_df[self.num_features] = self.scaler.transform(
            input_df[self.num_features]
        )

        x_cat = torch.tensor(input_df[self.cat_features].values, dtype=torch.long)
        x_num = torch.tensor(input_df[self.num_features].values, dtype=torch.float32)

        with torch.no_grad():
            scores_norm = self.model(x_cat, x_num).numpy()

        return self.target_scaler.inverse_transform(
            scores_norm.reshape(-1, 1)
        ).flatten()

    # ── 핵심 추론 로직 ─────────────────────────

    def predict(
        self,
        category:       str,
        companion_type: str,
        n_places:       int,
        age:            int = DEFAULT_AGE,
        alpha:          float = ALPHA,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        category       : '관광' | '미식' | '쇼핑'
        companion_type : '혼자 떠나는 여행' 등 4종
        n_places       : 추천 장소 수 (1~5)
        age            : 사용자 나이 (미입력 시 기본값 22세 사용)
        alpha          : 거리 페널티 가중치 (1km당 차감 점수, 기본 0.5)

        Returns
        -------
        DataFrame 컬럼: place_name, district, mood,
                        google_rating, stay_time_minutes,
                        predicted_score, latitude, longitude
        """
        self.validate_inputs(category, companion_type, n_places)

        # 해당 카테고리 장소 후보 선정
        candidates = (
            self.places[self.places['category'] == category]
            .copy()
            .reset_index(drop=True)
        )

        if candidates.empty:
            raise ValueError(f"'{category}' 카테고리에 해당하는 장소가 없습니다.")

        # 전체 후보 점수 산출
        candidates['predicted_score'] = np.round(
            self._score_candidates(candidates, category, companion_type, age), 2
        )

        # ── Greedy 동선 최적화 ──────────────────
        selected  = []
        remaining = candidates.copy()

        for step in range(n_places):
            if remaining.empty:
                break

            if step == 0:
                # 1순위: 거리 고려 없이 순수 예측 점수 최고 장소
                best_idx = remaining['predicted_score'].idxmax()
            else:
                # 이후: 직전 장소 기준 거리 페널티 적용
                prev_lat = selected[-1]['latitude']
                prev_lng = selected[-1]['longitude']

                distances = remaining.apply(
                    lambda row: haversine_distance(
                        prev_lat, prev_lng, row['latitude'], row['longitude']
                    ),
                    axis=1,
                )
                adjusted_scores = remaining['predicted_score'] - (alpha * distances)
                best_idx = adjusted_scores.idxmax()

            selected.append(remaining.loc[best_idx])
            remaining = remaining.drop(best_idx)
        # ────────────────────────────────────────

        result = pd.DataFrame(selected).reset_index(drop=True)
        return result[[
            'place_name', 'district', 'mood',
            'google_rating', 'stay_time_minutes',
            'predicted_score', 'latitude', 'longitude',
        ]]

    # ── 주변 식당 탐색 ─────────────────────────

    def get_nearby_restaurants(
        self,
        latitude:  float,
        longitude: float,
        radius_m:  float = 2000.0,
        top_n:     int   = 3,
    ) -> list[dict]:
        """
        특정 좌표 기준 반경 내 식당(category='미식')을 거리순으로 반환합니다.

        Parameters
        ----------
        latitude  : 기준 장소의 위도
        longitude : 기준 장소의 경도
        radius_m  : 탐색 반경 (단위: 미터, 기본값 500m)
        top_n     : 반환할 최대 식당 수 (기본값 3)

        Returns
        -------
        list[dict] : [
            {
                'name':          str,
                'district':      str,
                'mood':          str,
                'google_rating': float,
                'distance_m':    int,
            },
            ...
        ]
        """
        # ① 미식 카테고리 필터링
        restaurants = self.places[self.places['category'] == '미식'].copy()

        if restaurants.empty:
            return []

        # ② 각 식당까지의 거리 계산 (km → m 변환)
        restaurants['distance_m'] = restaurants.apply(
            lambda row: haversine_distance(
                latitude, longitude,
                row['latitude'], row['longitude']
            ) * 1000,
            axis=1,
        ).astype(int)

        # ③ 반경 필터링 + 거리 오름차순 정렬 + 상위 N개 선택
        nearby = (
            restaurants[restaurants['distance_m'] <= radius_m]
            .sort_values('distance_m')
            .head(top_n)
        )

        if nearby.empty:
            return []

        # ④ 반환 페이로드 조립
        return [
            {
                'name':          row['place_name'],
                'district':      row['district'],
                'mood':          row['mood'],
                'google_rating': float(row['google_rating']),
                'distance_m':    int(row['distance_m']),
            }
            for _, row in nearby.iterrows()
        ]


# ─────────────────────────────────────────────
# 3. 메인 실행 (CLI 테스트용)
# ─────────────────────────────────────────────

def get_user_input() -> dict:
    """CLI에서 사용자 입력을 받아 딕셔너리로 반환합니다."""

    print("\n" + "=" * 50)
    print("        🗾 도쿄 여행 일정 추천 시스템")
    print("=" * 50)

    print("\n[여행 테마]")
    for i, c in enumerate(VALID_CATEGORIES, 1):
        print(f"  {i}. {c}")
    while True:
        try:
            category = VALID_CATEGORIES[int(input("선택 (숫자 입력): ")) - 1]
            break
        except (ValueError, IndexError):
            print("  ⚠ 올바른 번호를 입력해 주세요.")

    print("\n[동행 유형]")
    for i, c in enumerate(VALID_COMPANION_TYPES, 1):
        print(f"  {i}. {c}")
    while True:
        try:
            companion_type = VALID_COMPANION_TYPES[int(input("선택 (숫자 입력): ")) - 1]
            break
        except (ValueError, IndexError):
            print("  ⚠ 올바른 번호를 입력해 주세요.")

    while True:
        try:
            n_places = int(input(f"\n[방문 장소 수] 1~{MAX_PLACES} 중 선택: "))
            if 1 <= n_places <= MAX_PLACES:
                break
            print(f"  ⚠ 1~{MAX_PLACES} 사이로 입력해 주세요.")
        except ValueError:
            print("  ⚠ 숫자를 입력해 주세요.")

    age_input = input(f"\n[나이] 입력 (Enter 시 기본값 {DEFAULT_AGE}세 적용): ").strip()
    try:
        age = int(age_input) if age_input else DEFAULT_AGE
    except ValueError:
        print(f"  ⚠ 숫자가 아니어서 기본값 {DEFAULT_AGE}세를 사용합니다.")
        age = DEFAULT_AGE

    return {
        'category':       category,
        'companion_type': companion_type,
        'n_places':       n_places,
        'age':            age,
    }


def _print_results(results: pd.DataFrame, user_input: dict):
    """추천 결과를 콘솔에 출력합니다."""

    print("\n" + "=" * 50)
    print("✨ 추천 여행 일정")
    print("=" * 50)
    print(
        f"  테마: {user_input['category']} | "
        f"동행: {user_input['companion_type']} | "
        f"나이: {user_input['age']}세\n"
    )

    for rank, (_, row) in enumerate(results.iterrows(), 1):
        print(f"  {rank}. {row['place_name']}")
        print(f"     📍 {row['district']}  |  분위기: {row['mood']}")
        print(
            f"     ⭐ Google 평점: {row['google_rating']}  |  "
            f"예상 체류: {int(row['stay_time_minutes'])}분  |  "
            f"예측 만족도: {row['predicted_score']:.1f}점"
        )
        print()

    if len(results) >= 2:
        places_for_url = results[['place_name', 'latitude', 'longitude']].rename(
            columns={'place_name': 'name'}
        ).to_dict('records')
        maps_url = build_google_maps_url(places_for_url)
        print("🗺  Google Maps 경로 URL:")
        print(f"  {maps_url}\n")
    else:
        print("  (경로 URL은 장소 2개 이상 선택 시 생성됩니다)\n")


def main():
    recommender = TravelRecommender(
        model_path        = 'best_recommender.pth',
        preprocessor_path = 'preprocessors.pkl',
        places_path       = os.path.join(DATA_DIR, 'places.csv'),
    )

    # 하드코딩 테스트
    print("\n" + "★" * 50)
    print("   [테스트] 하드코딩된 예시 입력을 실행합니다.")
    print("★" * 50)

    hardcoded_input = {
        'category':       '쇼핑',
        'companion_type': '친구와 떠나는 여행',
        'n_places':       4,
        'age':            28,
    }
    print(f"\n⏳ 테마: {hardcoded_input['category']} | "
          f"동행: {hardcoded_input['companion_type']} | "
          f"장소 수: {hardcoded_input['n_places']} | "
          f"나이: {hardcoded_input['age']}세")

    _print_results(recommender.predict(**hardcoded_input), hardcoded_input)
    print("★" * 50 + "\n")

    # CLI 직접 입력
    user_input = get_user_input()
    print("\n⏳ 추천 장소를 계산하는 중...")
    _print_results(recommender.predict(**user_input), user_input)


if __name__ == '__main__':
    main()
