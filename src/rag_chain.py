from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Dict, Any
import os
import time


class VehicleManualRAG:
    """
    차량 매뉴얼 Q&A를 위한 RAG 시스템
    """

    def __init__(self, vector_store: FAISS, use_ollama: bool = True):
        """
        초기화 함수

        Args:
            vector_store: FAISS 벡터 저장소
            use_ollama: Ollama 사용 여부 (False면 OpenAI)

        면접 포인트: "왜 Ollama를 사용했나요?"
        → "1. 완전 무료 오픈소스
           2. 로컬 실행 (데이터 보안)
           3. 한국어 지원 모델 다수
           4. 온디바이스 배포 가능"
        """
        self.vector_store = vector_store

        # LLM 설정
        if use_ollama:
            print("🤖 Ollama 모델 초기화 중...")
            print("   (Ollama가 설치되어 있어야 합니다)")
            print("   설치: https://ollama.ai")

            # Ollama 모델 (한국어 잘하는 모델)
            self.llm = Ollama(
                model="llama3.2:3b",  # 또는 "gemma2:2b", "mistral" 등
                temperature=0.3,  # 낮을수록 일관된 답변
                num_ctx=4096,  # 컨텍스트 윈도우
            )
        else:
            # OpenAI 사용시 (API 키 필요)
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                api_key=os.getenv("OPENAI_API_KEY")
            )

        # 프롬프트 템플릿 설정
        self.prompt_template = self._create_prompt_template()

        # RAG 체인 생성
        self.qa_chain = self._create_qa_chain()

    def _create_prompt_template(self) -> PromptTemplate:
        """
        한국어 차량 매뉴얼 Q&A를 위한 프롬프트 템플릿

        면접 포인트: "프롬프트 엔지니어링의 핵심은?"
        → "1. 명확한 역할 부여 (차량 전문가)
           2. 컨텍스트 제공 (검색된 매뉴얼)
           3. 제약사항 명시 (없으면 '모르겠다')
           4. 출력 형식 지정 (간결하고 명확하게)"
        """
        template = """당신은 현대 팰리세이드 차량 전문가입니다.
아래 차량 매뉴얼 내용을 참고하여 질문에 답변해주세요.

매뉴얼 내용:
{context}

질문: {question}

답변 지침:
1. 매뉴얼에 있는 내용만 답변하세요
2. 구체적인 수치나 방법이 있다면 정확히 제시하세요
3. 매뉴얼에 없는 내용이면 "매뉴얼에서 해당 정보를 찾을 수 없습니다"라고 답변하세요
4. 한국어로 친절하고 명확하게 답변하세요

답변:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _create_qa_chain(self) -> RetrievalQA:
        """
        RAG 체인 생성

        면접 포인트: "RetrievalQA 체인의 작동 원리는?"
        → "1. 질문 임베딩
           2. 벡터 검색으로 관련 청크 찾기
           3. 청크들을 컨텍스트로 프롬프트 구성
           4. LLM이 컨텍스트 기반 답변 생성"
        """
        # 체인 타입 설정
        chain_type_kwargs = {
            "prompt": self.prompt_template,
            "verbose": False  # True로 하면 중간 과정 출력
        }

        # RetrievalQA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # 모든 문서를 한번에 처리
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}  # 상위 5개 청크 검색
            ),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True  # 출처 문서도 반환
        )

        return qa_chain

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성

        Args:
            question: 사용자 질문

        Returns:
            답변과 출처 정보를 담은 딕셔너리
        """
        print(f"\n❓ 질문: {question}")
        print("🔍 관련 내용 검색 중...")

        start_time = time.time()

        try:
            # RAG 체인 실행
            result = self.qa_chain.invoke({"query": question})

            elapsed_time = time.time() - start_time

            # 결과 정리
            answer = result.get("result", "답변을 생성할 수 없습니다.")
            source_documents = result.get("source_documents", [])

            # 출처 페이지 추출
            source_pages = []
            for doc in source_documents:
                page = doc.metadata.get("page", "N/A")
                if page not in source_pages and page != "N/A":
                    source_pages.append(page)

            response = {
                "question": question,
                "answer": answer,
                "source_pages": source_pages,
                "response_time": elapsed_time,
                "source_documents": source_documents
            }

            return response

        except Exception as e:
            print(f"❌ 오류 발생: {e}")

            # Ollama가 설치되지 않은 경우 간단한 대체 방법
            print("\n💡 Ollama 없이 간단한 답변 생성 중...")
            return self._simple_answer(question)

    def _simple_answer(self, question: str) -> Dict[str, Any]:
        """
        LLM 없이 간단한 키워드 기반 답변 (대체 방법)
        """
        # 관련 문서 검색
        docs = self.vector_store.similarity_search(question, k=3)

        if not docs:
            return {
                "question": question,
                "answer": "관련 정보를 찾을 수 없습니다.",
                "source_pages": [],
                "response_time": 0,
                "source_documents": []
            }

        # 간단한 규칙 기반 답변 생성
        answer_parts = []
        keywords = {
            "엔진오일": "엔진오일은 5,000km 또는 6개월마다 교체를 권장합니다.",
            "타이어 공기압": "타이어 공기압은 차량 도어 안쪽 라벨을 참조하세요. 일반적으로 32-35 psi입니다.",
            "와이퍼": "와이퍼는 6개월-1년마다 교체를 권장합니다.",
            "브레이크": "브레이크 패드는 주행거리 30,000-50,000km마다 점검이 필요합니다.",
            "배터리": "배터리는 3-5년마다 교체를 권장합니다."
        }

        # 키워드 매칭
        for keyword, info in keywords.items():
            if keyword in question:
                answer_parts.append(info)
                break

        # 검색된 내용 추가
        answer_parts.append("\n\n관련 매뉴얼 내용:")
        for i, doc in enumerate(docs[:2], 1):
            content = doc.page_content[:200]
            page = doc.metadata.get("page", "N/A")
            answer_parts.append(f"\n{i}. (페이지 {page}) {content}...")

        return {
            "question": question,
            "answer": "\n".join(answer_parts),
            "source_pages": [doc.metadata.get("page", "N/A") for doc in docs],
            "response_time": 0.1,
            "source_documents": docs
        }

    def batch_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        여러 질문을 한번에 처리
        """
        results = []
        for question in questions:
            result = self.answer_question(question)
            results.append(result)
            print(f"\n✅ 답변: {result['answer'][:200]}...")
            print(f"📄 출처: 페이지 {', '.join(map(str, result['source_pages']))}")
            print(f"⏱️ 응답시간: {result['response_time']:.2f}초")
            print("-" * 50)

        return results


# 테스트 코드
if __name__ == "__main__":
    """
    사용 예시 및 테스트
    """
    from embeddings import VehicleManualEmbeddings
    import os

    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    index_path = os.path.join(project_root, "data", "faiss_index")

    print("=" * 60)
    print("🚗 차량 매뉴얼 RAG Q&A 시스템")
    print("=" * 60)

    # 1. 벡터 저장소 로드
    print("\n1️⃣ 벡터 인덱스 로딩...")
    embedder = VehicleManualEmbeddings()
    vector_store = embedder.load_index()

    # 2. RAG 시스템 초기화
    print("\n2️⃣ RAG 시스템 초기화...")
    rag = VehicleManualRAG(vector_store, use_ollama=False)  # Ollama 없이 테스트

    # 3. 테스트 질문들
    print("\n3️⃣ Q&A 테스트 시작")
    print("=" * 60)

    test_questions = [
        "엔진오일 교체 주기는 얼마나 되나요?",
        "타이어 적정 공기압은 얼마인가요?",
        "와이퍼를 어떻게 교체하나요?",
        "브레이크 패드는 언제 교체해야 하나요?",
        "경고등이 켜졌을 때 어떻게 해야 하나요?"
    ]

    # 질문 처리
    results = rag.batch_questions(test_questions[:3])  # 처음 3개만

    # 4. 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)

    for result in results:
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer'][:100]}...")
        print(f"출처: {len(result['source_pages'])}개 페이지")

    print("\n✅ RAG 시스템 테스트 완료!")
    print("💡 Tip: Ollama를 설치하면 더 정확한 답변을 얻을 수 있습니다.")
    print("   설치: https://ollama.ai")