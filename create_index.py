import os
import sys

sys.path.append('src')

from src.document_loader import VehicleManualLoader
from src.text_splitter import VehicleManualTextSplitter
from src.embeddings import VehicleManualEmbeddings


def create_faiss_index():
    print("=" * 60)
    print("FAISS 인덱스 생성 시작")
    print("=" * 60)

    # 1. PDF 경로 설정
    pdf_path = "data/LX3_2026_ko_KR.pdf"

    if not os.path.exists(pdf_path):
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        print("data 폴더에 LX3_2026_ko_KR.pdf 파일을 넣어주세요.")
        return False

    try:
        # 2. PDF 로드
        print("\n1. PDF 로딩 중... (590페이지)")
        loader = VehicleManualLoader(pdf_path)
        documents = loader.load_pdf()
        print(f"{len(documents)}페이지 로드 완료")

        # 3. 텍스트 분할
        print("\n2. 텍스트 분할 중...")
        splitter = VehicleManualTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)
        print(f"{len(chunks)}개 청크 생성 완료")

        # 4. 임베딩 및 인덱스 생성
        print("\n3. 벡터화 및 인덱스 생성 중...")
        print("약 3-5분 소요됩니다. 잠시만 기다려주세요...")

        embedder = VehicleManualEmbeddings()
        vector_store = embedder.create_vector_store(chunks, save=True)

        print("\nFAISS 인덱스 생성 완료!")
        print(f"저장 위치: data/faiss_index/")

        # 5. 테스트 검색
        print("\n4. 테스트 검색 수행 중...")
        test_query = "엔진오일 교체 주기"
        results = embedder.similarity_search(test_query, k=2)

        if results:
            print(f"테스트 성공! '{test_query}' 검색 결과:")
            for i, doc in enumerate(results, 1):
                print(f"  [{i}] {doc.page_content[:100]}...")

        return True

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 현재 디렉토리 확인
    print(f"현재 디렉토리: {os.getcwd()}")
    print(f"파일 목록: {os.listdir('.')}")

    # 인덱스 생성
    success = create_faiss_index()

    if success:
        print("\n" + "=" * 60)
        print("인덱스 생성 성공!")
        print("이제 app.py를 실행하세요: python app.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("인덱스 생성 실패")
        print("위의 오류 메시지를 확인하세요")
        print("=" * 60)