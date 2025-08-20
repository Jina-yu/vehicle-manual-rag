from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Tuple
import numpy as np
import pickle
import os
from tqdm import tqdm
import time


class VehicleManualEmbeddings:


    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):

        print(f" 임베딩 모델 로딩 중: {model_name}")
        print("   (첫 실행시 모델 다운로드로 시간이 걸릴 수 있습니다)")

        # HuggingFaceEmbeddings는 LangChain과 통합이 쉬움
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # GPU 없어도 OK
            encode_kwargs={'normalize_embeddings': True}  # 코사인 유사도 계산 최적화
        )

        self.vector_store = None
        self.index_path = "data/faiss_index"  # 인덱스 저장 경로

        print("임베딩 모델 로드 완료")

    def create_vector_store(self, chunks: List[Document], save: bool = True) -> FAISS:
        print(f" {len(chunks)}개 청크를 벡터로 변환 중...")
        print("   (6000개 기준 약 2-5분 소요)")

        start_time = time.time()

        # 배치 처리로 속도 향상
        batch_size = 100
        all_texts = [chunk.page_content for chunk in chunks]
        all_metadatas = [chunk.metadata for chunk in chunks]


        self.vector_store = FAISS.from_documents(
            documents=chunks[:batch_size],  # 첫 배치로 초기화
            embedding=self.embeddings
        )

        # 나머지 배치 추가
        for i in tqdm(range(batch_size, len(chunks), batch_size), desc="벡터화 진행"):
            batch_chunks = chunks[i:i + batch_size]
            if batch_chunks:  # 빈 배치가 아닌 경우만
                self.vector_store.add_documents(batch_chunks)

        elapsed_time = time.time() - start_time
        print(f"벡터화 완료! (소요시간: {elapsed_time:.1f}초)")

        # 인덱스 저장
        if save:
            self.save_index()

        # 통계 출력
        self._print_statistics()

        return self.vector_store

    def save_index(self):

        if not self.vector_store:
            raise ValueError("먼저 create_vector_store()를 실행하세요")

        print(f"인덱스 저장 중: {self.index_path}")

        # 디렉토리 생성
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # FAISS 인덱스 저장
        self.vector_store.save_local(self.index_path)

        print("인덱스 저장 완료")

    def load_index(self) -> FAISS:
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"인덱스 파일이 없습니다: {self.index_path}")

        print(f"인덱스 로딩 중: {self.index_path}")
        self.vector_store = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True  # 로컬 파일이므로 안전
        )
        print("인덱스 로드 완료")

        return self.vector_store

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        if not self.vector_store:
            raise ValueError("먼저 create_vector_store() 또는 load_index()를 실행하세요")

        # 벡터 검색 수행
        results = self.vector_store.similarity_search(query, k=k)

        return results

    def similarity_search_with_score(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        if not self.vector_store:
            raise ValueError("먼저 create_vector_store() 또는 load_index()를 실행하세요")

        results = self.vector_store.similarity_search_with_score(query, k=k)

        return results

    def _print_statistics(self):
        if not self.vector_store:
            return

        # FAISS 인덱스 정보
        print("\n 벡터 저장소 통계:")
        print(f"   - 저장된 벡터 수: {self.vector_store.index.ntotal:,}개")
        print(f"   - 벡터 차원: {self.vector_store.index.d}차원")
        print(f"   - 인덱스 타입: {type(self.vector_store.index).__name__}")


# 테스트 코드
if __name__ == "__main__":
    from document_loader import VehicleManualLoader
    from text_splitter import VehicleManualTextSplitter
    import os

    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    pdf_path = os.path.join(project_root, "data", "LX3_2026_ko_KR.pdf")
    index_path = os.path.join(project_root, "data", "faiss_index")

    print("=" * 60)
    print("차량 매뉴얼 임베딩 및 벡터 검색 테스트")
    print("=" * 60)

    # 임베딩 시스템 초기화
    embedder = VehicleManualEmbeddings()

    # 기존 인덱스가 있으면 로드, 없으면 새로 생성
    if os.path.exists(index_path):
        print("\n기존 인덱스 발견! 로드합니다...")
        vector_store = embedder.load_index()
    else:
        print("\n새로운 인덱스를 생성합니다...")

        # 1. PDF 로드
        print("\n1. PDF 로딩...")
        loader = VehicleManualLoader(pdf_path)
        documents = loader.load_pdf()

        # 2. 텍스트 분할
        print("\n2️. 텍스트 분할...")
        splitter = VehicleManualTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        # 3. 벡터화 및 인덱스 생성
        print("\n3️. 벡터화 시작...")
        vector_store = embedder.create_vector_store(chunks, save=True)

    # 4. 검색 테스트
    print("\n4️. 검색 테스트")
    print("-" * 50)

    test_queries = [
        "엔진 오일 교체 주기는?",
        "타이어 공기압은 얼마가 적정한가요?",
        "와이퍼 교체 방법",
        "경고등이 켜졌을 때 대처법",
        "브레이크 패드 점검"
    ]

    for query in test_queries[:3]:  # 처음 3개만 테스트
        print(f"\n 질문: {query}")

        # 유사도 점수와 함께 검색
        results = embedder.similarity_search_with_score(query, k=2)

        for i, (doc, score) in enumerate(results):
            print(f"\n   [{i + 1}] 유사도: {score:.3f}")
            print(f"   페이지: {doc.metadata.get('page', 'N/A')}")
            print(f"   섹션: {doc.metadata.get('section', 'N/A')}")
            print(f"   내용: {doc.page_content[:150]}...")

    print("\n" + "=" * 60)
    print("모든 테스트 완료!")
    print("=" * 60)