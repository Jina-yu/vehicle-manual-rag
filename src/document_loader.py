from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List
import os

class VehicleManualLoader:
    def __init__(self, file_path: str):

        self.file_path = file_path
        self.documents = []

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {file_path}")

    def load_pdf(self) -> List[Document]:
        print(f"PDF 로딩 중: {self.file_path}")

        loader = PyPDFLoader(self.file_path)
        self.documents = loader.load()

        print(f"총 {len(self.documents)} 페이지 로드 완료")

        self._preview_documents()
        return self.documents

    def _preview_documents(self, num_pages: int = 2):
        print("\n 문서 미리 보기: ")
        print("-" * 50)

        for i, doc in enumerate(self.documents[:num_pages]):
            print(f"\n[페이지 {i+1}]")
            print(f"메타데이터: {doc.metadata}")

            content_preview = doc.page_content[:500]
            print(f"내용 미리보기: {content_preview} ...")
            print("-" * 50)

    def get_page(self, page_num: int) -> Document:
        if not self.documents:
            raise ValueError("먼저 load_pdf()를 실행해주세요.")

        if page_num < 1 or page_num > len(self.documents):
            raise ValueError(f"페이지 번호는 1~{len(self.documents)} 사이여야 합니다")

        return self.documents[page_num - 1]

    def search_keyword(self ,keyword: str) -> List[tuple]:
        if not self.documents:
            raise ValueError("먼저 load_pdf()를 실행해주세요.")

        results = []

        for i, doc in enumerate(self.documents):
            if keyword.lower() in doc.page_content.lower():
                sentences = doc.page_content.split(".")
                for sentence in sentences:
                    if keyword.lower() in sentence.lower():
                        results.append((i+1, sentence.strip()))

        return results


if __name__ == "__main__":
    pdf_path = r"C:\Users\Admin\Desktop\vehicle-manual-rag\data\LX3_2026_ko_KR.pdf"

    try:
        loader = VehicleManualLoader(pdf_path)

        documents = loader.load_pdf()

        print(f"\n 문서 통계:")
        print(f"- 총 페이지 수: {len(documents)}")

        total_chars = sum(len(doc.page_content) for doc in documents)
        print(f" - 총 문자 수: {total_chars}")
        print(f" - 평균 페이지당 문자: {total_chars // len(documents):,}")

        print("\n '엔진 오일' 키워드 검색 결과: ")
        results = loader.search_keyword("엔진 오일")
        for page_num, sentence in results[:3]:
            print(f" - {page_num}페이지: {sentence[:100]}")

    except FileNotFoundError as e:
        print(f"에러: {e}")
        print("PDF 파일을 'data' 폴더에 넣어주세요")