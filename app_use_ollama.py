import gradio as gr
import os
import sys
import time
from typing import List, Tuple
from dotenv import load_dotenv

# src 폴더를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 전역 변수 정의 (반드시 import 후, 함수 정의 전에!)
rag_system = None
chat_history = []

load_dotenv()

from src.embeddings import VehicleManualEmbeddings
from src.rag_chain import VehicleManualRAG


def initialize_system():
    """RAG 시스템 초기화"""
    global rag_system  # 전역 변수 사용 선언

    if rag_system is not None:
        return "시스템이 이미 초기화되어 있습니다."

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
        rag_system = VehicleManualRAG(vector_store, use_ollama=True)

        return "시스템 초기화 완료! 질문을 입력해주세요."

    except Exception as e:
        return f"초기화 실패: {str(e)}"


def answer_question(question: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """질문에 답변하고 채팅 기록 업데이트"""
    global rag_system  # 전역 변수 사용 선언

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

        # 출처 정보 추가
        if source_pages:
            answer += f"\n\n📄 **출처**: 매뉴얼 {', '.join(map(str, source_pages[:3]))} 페이지"
        answer += f"\n⏱️ 응답시간: {response_time:.2f}초"

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


def create_demo():
    """Gradio 데모 인터페이스 생성"""

    with gr.Blocks(title="🚗 팰리세이드 매뉴얼 AI 어시스턴트", theme=gr.themes.Soft()) as demo:
        # 헤더
        gr.Markdown("""
        # 🚗 팰리세이드 2026 매뉴얼 AI 어시스턴트

        현대 팰리세이드 차량 매뉴얼에 대한 질문을 자연어로 입력하세요.
        AI가 매뉴얼을 검색하여 정확한 답변을 제공합니다.

        **사용 방법:**
        1. 먼저 '시스템 초기화' 버튼을 클릭하세요
        2. 차량 관련 질문을 입력하세요
        3. Enter를 누르거나 '전송' 버튼을 클릭하세요
        """)

        # 초기화 섹션
        with gr.Row():
            init_btn = gr.Button("🚀 시스템 초기화", variant="primary")
            init_status = gr.Textbox(label="상태", interactive=False)

        # 채팅 인터페이스
        chatbot = gr.Chatbot(
            label="대화 내역",
            height=400,
            type="tuples"  # 타입 명시
        )

        with gr.Row():
            msg = gr.Textbox(
                label="질문 입력",
                placeholder="예: 엔진오일 교체 주기는? / 타이어 공기압은? / 경고등이 켜졌어요",
                lines=2,
                scale=4
            )
            send_btn = gr.Button("📤 전송", variant="primary", scale=1)

        # 예시 질문 버튼들
        gr.Markdown("### 💡 예시 질문")
        with gr.Row():
            example_btns = [
                gr.Button("엔진오일 교체 주기", size="sm"),
                gr.Button("타이어 적정 공기압", size="sm"),
                gr.Button("와이퍼 교체 방법", size="sm"),
                gr.Button("경고등 대처법", size="sm")
            ]

        # 컨트롤 버튼
        with gr.Row():
            clear_btn = gr.Button("🗑️ 대화 초기화")

        # 추가 정보
        with gr.Accordion("📊 시스템 정보", open=False):
            gr.Markdown("""
            - **PDF**: LX3_2026_ko_KR.pdf (590페이지)
            - **청크 수**: 6,354개
            - **임베딩 모델**: paraphrase-multilingual-MiniLM-L12-v2
            - **벡터 DB**: FAISS
            - **LLM**: GPT-3.5-turbo
            - **평균 응답시간**: < 2초
            """)

        # 이벤트 연결
        init_btn.click(
            fn=initialize_system,
            outputs=init_status
        )

        # 메시지 전송
        msg.submit(answer_question, [msg, chatbot], [msg, chatbot])
        send_btn.click(answer_question, [msg, chatbot], [msg, chatbot])

        # 예시 질문 버튼들
        example_questions = [
            "엔진오일 교체 주기는 얼마나 되나요?",
            "타이어 적정 공기압은 얼마인가요?",
            "와이퍼를 어떻게 교체하나요?",
            "경고등이 켜졌을 때 어떻게 해야 하나요?"
        ]

        for btn, question in zip(example_btns, example_questions):
            btn.click(
                lambda q=question: (q, []),
                outputs=[msg, chatbot]
            )

        # 대화 초기화
        clear_btn.click(clear_chat, outputs=[chatbot, msg])

        # 자동 초기화
        demo.load(fn=initialize_system, outputs=init_status)

    return demo


# 메인 실행
if __name__ == "__main__":
    print("=" * 60)
    print("팰리세이드 매뉴얼 AI 어시스턴트 시작")
    print("=" * 60)

    # 환경변수 체크
    if not os.getenv("OPENAI_API_KEY"):
        print("\n경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("설정 방법:")
        print("Windows: set OPENAI_API_KEY=sk-...")
        print("Mac/Linux: export OPENAI_API_KEY=sk-...")

        # 직접 입력 받기
        api_key = input("\nOpenAI API Key를 입력하세요 (sk-...): ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print("API Key 설정 완료")
        else:
            print("API Key 없이는 고급 답변 기능이 제한됩니다.")

    # Gradio 데모 실행
    demo = create_demo()

    print("\n웹 인터페이스 시작 중...")
    print("브라우저에서 자동으로 열립니다.")
    print("수동 접속: http://localhost:7860")
    print("\n종료: Ctrl+C")

    # 데모 실행
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )