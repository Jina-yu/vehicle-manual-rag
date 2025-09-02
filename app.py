# app.py (Render 배포용)
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from gradio import mount_gradio_app
import gradio as gr
import os
import sys
import time
from typing import List, Tuple
from dotenv import load_dotenv

# src 폴더를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.embeddings import VehicleManualEmbeddings
from src.rag_chain import VehicleManualRAG

load_dotenv()

# FastAPI 앱 생성
app = FastAPI(
    title="🚗 팰리세이드 매뉴얼 AI 어시스턴트",
    description="현대 팰리세이드 2026 차량 매뉴얼 Q&A 시스템",
    version="1.0.0"
)

# CORS 설정 (필요시)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수로 RAG 시스템 저장
rag_system = None
chat_history = []


def initialize_system():
    """RAG 시스템 초기화"""
    global rag_system

    if rag_system is not None:
        return "✅ 시스템이 이미 초기화되어 있습니다."

    try:
        # 경로 설정
        project_root = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(project_root, "data", "faiss_index")

        # 벡터 저장소 로드
        print("벡터 인덱스 로딩 중...")
        embedder = VehicleManualEmbeddings()
        vector_store = embedder.load_index()

        # RAG 시스템 초기화 (OpenAI 사용)
        print("RAG 시스템 초기화 중...")
        rag_system = VehicleManualRAG(vector_store, use_ollama=False)

        return "✅ 시스템 초기화 완료! 질문을 입력해주세요."

    except Exception as e:
        return f"❌ 초기화 실패: {str(e)}"


def answer_question(question: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """질문에 답변하고 채팅 기록 업데이트"""
    global rag_system

    if rag_system is None:
        return "먼저 '시스템 초기화' 버튼을 클릭해주세요.", history

    if not question.strip():
        return "", history

    try:
        # RAG 시스템으로 답변 생성
        result = rag_system.answer_question(question)

        # 답변 포맷팅
        answer = result['answer']
        source_pages = result.get('source_pages', [])
        response_time = result.get('response_time', 0)

        # 출처 정보 추가 (선택사항)
        if source_pages:
            answer += f"\n\n📄 출처: 매뉴얼 {', '.join(map(str, source_pages[:3]))} 페이지"

        # 채팅 기록 업데이트
        history.append((question, answer))
        return "", history

    except Exception as e:
        error_msg = f"오류 발생: {str(e)}"
        history.append((question, error_msg))
        return "", history


def clear_chat():
    """채팅 기록 초기화"""
    return [], []


def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    with gr.Blocks(title="🚗 팰리세이드 매뉴얼 AI", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🚗 팰리세이드 2026 매뉴얼 AI 어시스턴트

        현대 팰리세이드 차량 매뉴얼에 대한 질문을 자연어로 입력하세요.
        """)

        # 초기화 섹션
        with gr.Row():
            init_btn = gr.Button("🚀 시스템 초기화", variant="primary")
            init_status = gr.Textbox(label="상태", interactive=False)

        # 채팅 인터페이스
        chatbot = gr.Chatbot(
            label="대화 내역",
            height=400,
            type="tuples"
        )

        with gr.Row():
            msg = gr.Textbox(
                label="질문 입력",
                placeholder="예: 엔진오일 교체 주기는?",
                lines=2,
                scale=4
            )
            send_btn = gr.Button("📤 전송", variant="primary", scale=1)

        # 예시 질문 버튼들
        gr.Markdown("### 💡 예시 질문")
        with gr.Row():
            gr.Button("엔진오일 교체 주기", size="sm").click(
                lambda: ("엔진오일 교체 주기는 얼마나 되나요?", []),
                outputs=[msg, chatbot]
            )
            gr.Button("타이어 적정 공기압", size="sm").click(
                lambda: ("타이어 적정 공기압은 얼마인가요?", []),
                outputs=[msg, chatbot]
            )

        # 대화 초기화
        clear_btn = gr.Button("🗑️ 대화 초기화")

        # 이벤트 연결
        init_btn.click(fn=initialize_system, outputs=init_status)
        msg.submit(answer_question, [msg, chatbot], [msg, chatbot])
        send_btn.click(answer_question, [msg, chatbot], [msg, chatbot])
        clear_btn.click(clear_chat, outputs=[chatbot, msg])

        # 자동 초기화
        interface.load(fn=initialize_system, outputs=init_status)

    return interface


# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Vehicle Manual RAG",
        "rag_initialized": rag_system is not None
    }


# API 엔드포인트 (선택사항)
@app.get("/api/info")
async def get_info():
    return {
        "title": "팰리세이드 2026 매뉴얼 AI",
        "total_pages": 590,
        "total_chunks": 6354,
        "model": "GPT-3.5-turbo"
    }


# Gradio 인터페이스 생성 및 마운트
gradio_interface = create_gradio_interface()
app = mount_gradio_app(app, gradio_interface, path="/")


# Startup 이벤트
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
    print("=" * 60)
    print("🚗 팰리세이드 매뉴얼 AI 어시스턴트 시작")
    print("=" * 60)

    # 환경변수 체크
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
    else:
        print("✅ OpenAI API Key detected")


# 메인 실행
if __name__ == "__main__":
    # Render는 PORT 환경변수를 제공합니다
    port = int(os.environ.get("PORT", 10000))

    # Uvicorn 서버 실행
    uvicorn.run(
        "app:app",  # app.py의 app 객체
        host="0.0.0.0",
        port=port,
        reload=False  # Production에서는 False
    )