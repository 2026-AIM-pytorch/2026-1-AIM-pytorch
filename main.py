"""
main.py — 통합 실행 진입점

실행 방법:
    python main.py

접속:
    http://localhost:8000 → login.html 자동 이동
"""

import os
import sys
import time

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# backend/ 폴더를 모듈 탐색 경로에 추가
# → api.py 내부의 `from inference import ...` 등이 정상 동작
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

# ─────────────────────────────────────────────
# 환경변수 로드 (최상위에서 1회 처리)
# ─────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# FastAPI 앱 import (sys.path 등록 이후에 수행)
# ─────────────────────────────────────────────
from api import app

# ─────────────────────────────────────────────
# 루트 경로 → login.html 리다이렉트
# StaticFiles mount 이전에 등록해야 '/' 라우트가 우선 적용됨
# ─────────────────────────────────────────────
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

@app.get("/")
def root():
    return RedirectResponse(url="/login.html")

# ─────────────────────────────────────────────
# frontend/ 폴더 정적 파일 마운트
# ─────────────────────────────────────────────
app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend",
)

# ─────────────────────────────────────────────
# 서버 실행
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    import threading
    import webbrowser

    def open_browser():
        time.sleep(1)
        webbrowser.open("http://localhost:8000")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[BASE_DIR],
    )