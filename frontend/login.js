document.addEventListener('DOMContentLoaded', function() {
    
    const submitBtn = document.getElementById('submitBtn');
    
    if (submitBtn) {
        submitBtn.addEventListener('click', async function(event) {
            event.preventDefault(); 

            // 1. 각 select 엘리먼트에서 유저가 선택한 값 가져오기
            const category = document.getElementById('category').value;
            const companion_type = document.getElementById('companion_type').value;
            const age = document.getElementById('age').value;
            const n_places = document.getElementById('n_places').value;

            // 2. 입력값 검증 (하나라도 선택되지 않아 빈 스트링이면 안내창 출력)
            if (!category || !companion_type || !age || !n_places) {
                alert("모든 항목을 선택해주세요!");
                return; 
            }

            // 3. 백엔드(api.py) 스키마 포맷에 맞춰 데이터 포장
            const requestData = {
                category: category,
                companion_type: companion_type,
                age: parseInt(age, 10),       
                n_places: parseInt(n_places, 10)
            };

            // 4. 로딩 텍스트 상태 변경 및 중복 클릭 차단
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.innerHTML = 'AI가 일정을 생성 중입니다... ⏳';
            submitBtn.disabled = true;
            submitBtn.classList.add('opacity-50', 'cursor-not-allowed'); 

            // 5. 비동기 통신 요청
            try {
                const response = await fetch("http://localhost:8000/recommend", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(requestData)
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || "서버 요청 중 오류가 발생했습니다.");
                }

                const responseData = await response.json();

                // 6. 응답 본문을 로컬 스토리지에 세이브 후 결과화면 이동
                sessionStorage.setItem("travelResult", JSON.stringify(responseData));
                window.location.href = "result.html";

            } catch (error) {
                console.error("통신 에러 발생:", error);
                alert("일정을 설계하지 못했습니다: " + error.message);
                
                // 에러 시 버튼 원상복귀
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
                submitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
        });
    }
});