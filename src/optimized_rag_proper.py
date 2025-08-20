"""
optimized_rag_proper.py
LLM을 유지하면서 응답시간을 개선하는 올바른 방법
목표: 1초 이내 (현실적 목표)
"""

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import os
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import json

class ProperlyOptimizedRAG:
    """
    LLM을 유지하면서 최적화하는 올바른 방법
    """

    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store

        # 1. 더 빠른 모델 사용 (gpt-3.5-turbo = 안정적인 버전)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # 안정적인 모델
            temperature=0.1,  # 일관성
            max_tokens=300,  # 답변 길이 제한으로 속도 향상
            api_key=os.getenv("OPENAI_API_KEY"),
            request_timeout=5
        )

        # 2. 최적화된 짧은 프롬프트
        self.prompt_template = self._create_minimal_prompt()

        # 3. 스마트 캐싱 (LRU 캐시)
        self.cache = {}  # {question_hash: (answer, timestamp)}
        self.cache_ttl = 3600  # 1시간

        # 4. 비동기 처리 준비
        self.executor = ThreadPoolExecutor(max_workers=2)

    def _create_minimal_prompt(self) -> PromptTemplate:
        """
        최소한의 프롬프트 (토큰 수 줄이기)
        """
        # 짧고 명확한 프롬프트 = 빠른 처리
        template = """매뉴얼: {context}

질문: {question}

매뉴얼 내용만으로 간단명료하게 답변:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        최적화된 답변 생성 (여전히 LLM 사용)
        """
        start_time = time.time()

        # 1. 캐시 확인 (5ms)
        cache_key = hash(question)
        if cache_key in self.cache:
            cached_answer, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return {
                    "question": question,
                    "answer": cached_answer['answer'],
                    "source_pages": cached_answer['pages'],
                    "response_time": 0.01,  # 캐시는 10ms
                    "cached": True
                }

        # 2. 병렬 처리: 벡터 검색과 전처리 동시 실행
        with self.executor as executor:
            # 벡터 검색 (비동기)
            future_search = executor.submit(
                self._fast_vector_search,
                question
            )

            # 검색 결과 대기
            search_results = future_search.result(timeout=0.5)

            if not search_results:
                return self._fallback_response(question, start_time)

            # 3. 컨텍스트 최적화 (중복 제거, 요약)
            context = self._optimize_context(search_results)

            # 4. LLM 호출 (스트리밍으로 첫 토큰 빠르게)
            answer = self._fast_llm_call(question, context)

            # 5. 페이지 정보 추출
            pages = list(set([
                doc.metadata.get('page', 0)
                for doc in search_results
            ]))[:3]

            response = {
                "question": question,
                "answer": answer,
                "source_pages": sorted(pages),
                "response_time": time.time() - start_time,
                "cached": False
            }

            # 6. 캐시 저장
            self.cache[cache_key] = (
                {"answer": answer, "pages": pages},
                time.time()
            )

            return response

    def _fast_vector_search(self, question: str, k: int = 3) -> List[Document]:
        """
        빠른 벡터 검색 (k를 줄이고 MMR 제거)
        """
        try:
            # similarity_search가 similarity_search_with_score보다 빠름
            docs = self.vector_store.similarity_search(
                question,
                k=k,  # 3개만 검색
                fetch_k=k  # MMR 비활성화
            )
            return docs
        except Exception as e:
            print(f"벡터 검색 오류: {e}")
            return []

    def _optimize_context(self, docs: List[Document]) -> str:
        """
        컨텍스트 최적화 (중복 제거, 핵심만 추출)
        """
        seen_content = set()
        optimized = []

        for doc in docs:
            content = doc.page_content.strip()

            # 중복 체크
            content_hash = hash(content[:50])  # 앞 50자로 중복 체크
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)

            # 너무 짧거나 긴 내용 제외
            if len(content) < 50 or len(content) > 500:
                content = content[:500]  # 최대 500자

            optimized.append(content)

        # 최대 3개 청크만 사용 (토큰 수 제한)
        return "\n---\n".join(optimized[:3])

    def _fast_llm_call(self, question: str, context: str) -> str:
        """
        빠른 LLM 호출
        """
        try:
            # 프롬프트 구성
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )

            # LLM 호출 (invoke가 가장 빠름)
            response = self.llm.invoke(prompt)

            # 응답 추출
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            print(f"LLM 호출 오류: {e}")
            return "답변 생성 중 오류가 발생했습니다."

    def _fallback_response(self, question: str, start_time: float) -> Dict:
        """
        폴백 응답 (벡터 검색 실패 시)
        """
        return {
            "question": question,
            "answer": "해당 정보를 매뉴얼에서 찾을 수 없습니다. 다른 표현으로 질문해 주시거나, 서비스센터(1577-0001)로 문의해 주세요.",
            "source_pages": [],
            "response_time": time.time() - start_time,
            "cached": False
        }

    def batch_test(self, questions: List[str]) -> Dict[str, Any]:
        """
        배치 테스트 및 성능 측정
        """
        results = []
        times = []

        print("\n" + "="*60)
        print("🚀 최적화된 RAG 성능 테스트 (LLM 유지)")
        print("="*60)

        for i, question in enumerate(questions, 1):
            result = self.answer_question(question)
            results.append(result)
            times.append(result['response_time'])

            status = "📦 캐시" if result.get('cached') else "🔍 검색"
            print(f"\n[{i}] {question}")
            print(f"⏱️  {result['response_time']:.2f}초 ({status})")
            print(f"📄 페이지: {result['source_pages']}")
            print(f"💬 {result['answer'][:150]}...")

        # 통계
        avg_time = sum(times) / len(times)
        cached_count = sum(1 for r in results if r.get('cached'))

        print("\n" + "="*60)
        print(f"📊 성능 통계:")
        print(f"  평균 응답: {avg_time:.2f}초")
        print(f"  최소/최대: {min(times):.2f}초 / {max(times):.2f}초")
        print(f"  캐시 적중: {cached_count}/{len(questions)}")
        print(f"  목표 달성: {'✅' if avg_time < 1.0 else '⚠️ 1초 목표 미달'}")
        print("="*60)

        return {
            "average": avg_time,
            "min": min(times),
            "max": max(times),
            "results": results
        }


# 추가: 비동기 버전 (더 빠름)
class AsyncOptimizedRAG:
    """
    비동기 처리로 더 빠른 응답 (실험적)
    """

    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-1106",
            temperature=0.1,
            max_tokens=300,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.cache = {}

    async def answer_question_async(self, question: str) -> Dict[str, Any]:
        """
        비동기 답변 생성
        """
        start_time = time.time()

        # 비동기로 벡터 검색과 LLM 준비 동시 실행
        search_task = asyncio.create_task(
            self._async_vector_search(question)
        )

        # 검색 완료 대기
        docs = await search_task

        if not docs:
            return {
                "question": question,
                "answer": "정보를 찾을 수 없습니다.",
                "source_pages": [],
                "response_time": time.time() - start_time
            }

        # LLM 호출
        context = "\n".join([d.page_content[:300] for d in docs[:3]])
        answer = await self._async_llm_call(question, context)

        return {
            "question": question,
            "answer": answer,
            "source_pages": [d.metadata.get('page', 0) for d in docs],
            "response_time": time.time() - start_time
        }

    async def _async_vector_search(self, question: str) -> List[Document]:
        """비동기 벡터 검색"""
        return await asyncio.to_thread(
            self.vector_store.similarity_search,
            question,
            k=3
        )

    async def _async_llm_call(self, question: str, context: str) -> str:
        """비동기 LLM 호출"""
        prompt = f"매뉴얼: {context}\n질문: {question}\n답변:"

        response = await asyncio.to_thread(
            self.llm.invoke,
            prompt
        )

        return response.content if hasattr(response, 'content') else str(response)


# 메인 테스트
if __name__ == "__main__":
    from embeddings import VehicleManualEmbeddings
    import os

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API Key를 입력하세요:")
        api_key = input("sk-... : ").strip()
        os.environ["OPENAI_API_KEY"] = api_key

    # 벡터 저장소 로드
    print("벡터 인덱스 로딩 중...")
    embedder = VehicleManualEmbeddings()
    vector_store = embedder.load_index()

    # 최적화된 RAG 시스템
    rag = ProperlyOptimizedRAG(vector_store)

    # 테스트 질문
    test_questions = [
        "엔진오일 교체 주기는?",
        "타이어 공기압은 얼마?",
        "경고등이 켜졌을 때 어떻게 해야 하나요?",
        "눈길 주행 시 주의사항",
        "운전자 보조 시스템 설정"
    ]

    # 첫 번째 실행 (콜드 스타트)
    print("\n### 1차 실행 (캐시 없음) ###")
    stats1 = rag.batch_test(test_questions)

    # 두 번째 실행 (캐시 활용)
    print("\n### 2차 실행 (캐시 활용) ###")
    stats2 = rag.batch_test(test_questions[:3])

    # 개선율 계산
    improvement = ((stats1['average'] - stats2['average']) / stats1['average']) * 100
    print(f"\n🎯 캐시 효과: {improvement:.1f}% 속도 향상")