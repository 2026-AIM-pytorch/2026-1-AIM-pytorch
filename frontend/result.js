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
    subtitleEl.innerHTML =
        `AI가 ${categoryKo} 테마에 맞춰 선별한 도쿄 최적 코스입니다.<br> ` +
        `매칭 점수와 교통 동선을 종합적으로 분석하여 반영했습니다.`;

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

    places.forEach((place, i) => {
        const card = document.createElement('div');
        card.className = 'timeline-item';
        card.style.animationDelay = `${i * 0.08}s`;

        // 주변 식당 HTML 조립 (관광·쇼핑 테마이고 데이터가 있을 때만)
        const restaurants = place.nearby_restaurants || [];
        const restaurantHTML = restaurants.length
            ? `<div style="margin-top:10px; padding:10px 12px; background:#f7f6ee; border-radius:8px;">
                <div style="font-size:11px; font-weight:700; letter-spacing:0.08em; color:#717973; margin-bottom:6px;">🍽️ 주변 식당</div>
                ${restaurants.map(r => `
                    <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px; color:#414943; padding:3px 0;">
                        <span>${r.name} <span style="color:#9a9b92; font-size:11px;">${r.mood}</span></span>
                        <span style="color:#717973; white-space:nowrap; margin-left:12px;">⭐ ${r.google_rating} &nbsp;·&nbsp; ${r.distance_m}m</span>
                    </div>
                `).join('')}
               </div>`
            : '';

        card.innerHTML = `
            <div class="rank-badge">${place.rank}</div>

            <div style="flex:1; min-width:0;">

                <div class="place-name">
                    ${place.name}
                    <span style="font-size:15px; font-weight:400; color: #717973; margin-left: 8px;">
                        ${place.mood}
                    </span>
                </div>

                <div class="place-meta">
                    📍 ${place.district} &nbsp;·&nbsp; ⏱️ 권장 체류 ${place.stay_time_minutes}분 &nbsp;·&nbsp; 🎯 매칭 점수 ${place.predicted_score}점
                </div>

                ${restaurantHTML}
            </div>
        `;

        timeline.appendChild(card);
    });

});