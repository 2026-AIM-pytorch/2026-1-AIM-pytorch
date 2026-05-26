// ─────────────────────────────────────────────────────────
// result.js  — 새 디자인용 (타임라인 + 인터랙티브 지도)
// ─────────────────────────────────────────────────────────

// 카테고리 → 한글 태그 매핑
const CATEGORY_LABEL = {
    '관광':  'SIGHTSEEING',
    '미식':  'DINING',
    '쇼핑':  'SHOPPING',
};

// 동반자 유형 → 제목 문구 매핑
const COMPANION_TITLE = {
    '혼자 떠나는 여행':   'Solo Journey',
    '친구와 떠나는 여행': 'Friends\' Journey',
    '연인과 떠나는 여행': 'Romantic Journey',
    '부모님과 떠나는 여행':'Family Journey',
};

// 체류 시간 기준 방문 시각 자동 계산 (09:00 시작)
function calcTimes(places) {
    const times = [];
    let minutes = 9 * 60; // 09:00
    for (const p of places) {
        const h = String(Math.floor(minutes / 60)).padStart(2, '0');
        const m = String(minutes % 60).padStart(2, '0');
        times.push(`${h}:${m}`);
        minutes += p.stay_time_minutes + 30; // 이동 여유 30분
    }
    return times;
}

document.addEventListener('DOMContentLoaded', function () {

    // ── 1. 데이터 로드 ──────────────────────────────────────
    const storedData = sessionStorage.getItem('travelResult');
    if (!storedData) {
        alert('저장된 여정 데이터가 없습니다. 메인 페이지로 돌아갑니다.');
        window.location.href = 'login.html';
        return;
    }

    const data        = JSON.parse(storedData);
    const places      = data.places  || [];
    const route       = data.route   || {};
    const input       = data.input   || {};
    const staticMapUrl = data.static_map_url;
    const mapsUrl      = data.maps_url || route.maps_url;

    // ── 2. 헤딩 텍스트 ──────────────────────────────────────
    const titleEl    = document.getElementById('journey-title');
    const subtitleEl = document.getElementById('journey-subtitle');

    const companionTitle = COMPANION_TITLE[input.companion_type] || '당신을 위한 여정';
    titleEl.textContent  = companionTitle;

    const categoryKo = input.category || '';
    subtitleEl.textContent =
        `${categoryKo} 테마로 AI가 선별한 도쿄 최적 코스입니다. ` +
        `매칭 점수와 교통 동선을 함께 분석한 결과입니다.`;

    // ── 3. 경로 요약 바 — 삭제됨 ───────────────────────────

    // ── 4. 지도 이미지 ──────────────────────────────────────
    const skeleton = document.getElementById('map-skeleton');
    const mapImg   = document.getElementById('map-img');
    const mapsLink = document.getElementById('maps-link');
    const openMapsBtn = document.getElementById('open-maps-btn');

    if (staticMapUrl) {
        mapImg.onload = () => {
            skeleton.style.display = 'none';
            mapImg.style.display   = 'block';
        };
        mapImg.onerror = () => {
            skeleton.style.display = 'none';
            mapImg.style.display   = 'none';
        };
        mapImg.src = staticMapUrl;
    } else {
        skeleton.style.display = 'none';
    }

    if (mapsUrl) {
        mapsLink.style.display = 'flex';
        mapsLink.addEventListener('click', () => window.open(mapsUrl, '_blank'));
        openMapsBtn.href          = mapsUrl;
        openMapsBtn.style.display = 'inline-flex';
    }

    // ── 5. 타임라인 카드 ──────────────────────────────────────
    const timeline = document.getElementById('timeline');
    if (!places.length) {
        timeline.innerHTML = `<p style="color:#9a9b92; padding:40px 0;">추천된 장소가 없습니다.</p>`;
        return;
    }

    const times = calcTimes(places);

    places.forEach((place, i) => {
        const tagLabel = CATEGORY_LABEL[place.mood?.includes('쇼핑') ? '쇼핑' : input.category] || 'TRAVEL';
        const card = document.createElement('div');
        card.className = 'timeline-item';
        card.style.animationDelay = `${i * 0.08}s`;

        card.innerHTML = `
            <div class="rank-badge">${place.rank}</div>

            <div style="flex:1; min-width:0;">
                <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
                    <span style="font-size:13px; font-weight:600; color:#717973; letter-spacing:0.04em;">
                        ${times[i]}
                    </span>
                    <span class="category-tag">${tagLabel}</span>
                </div>

                <div class="place-name">${place.name}</div>

                <div class="place-meta">
                    📍 ${place.district} &nbsp;·&nbsp; ⏱️ 권장 체류 ${place.stay_time_minutes}분
                </div>
            </div>
        `;

        timeline.appendChild(card);
    });

});