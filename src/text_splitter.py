from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import re

class VehicleManualTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = len,
            separators=["\n\n", "\n", ".", "。", "!", "?", ",", " ", ""],
            is_separator_regex=False
        )

        self.chunks = []

    def split_documents(self, documents: List[Document]) -> List[Document]:

        print(f"{len(documents)}개 페이지를 청크로 분할 중...")

        # 분할 전 전처리
        processed_docs = self._preprocess_documents(documents)

        # LangChain의 split_documents 메서드 사용
        self.chunks = self.text_splitter.split_documents(processed_docs)

        # 청크에 추가 메타데이터 부여
        self._add_chunk_metadata()

        print(f"총 {len(self.chunks)}개 청크 생성 완료")
        print(f"평균 청크 크기: {self._get_avg_chunk_size():.0f}자")

        return self.chunks

    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:

        processed = []

        for doc in documents:
            # 연속된 공백을 하나로
            content = re.sub(r'\s+', ' ', doc.page_content)

            # 페이지 번호 패턴 제거 (예: "- 123 -", "Page 123")
            content = re.sub(r'-\s*\d+\s*-', '', content)
            content = re.sub(r'Page\s*\d+', '', content, flags=re.IGNORECASE)

            # 너무 짧은 페이지는 스킵 (목차, 빈 페이지 등)
            if len(content.strip()) < 50:
                continue

            processed.append(Document(
                page_content=content,
                metadata=doc.metadata
            ))

        return processed

    def _add_chunk_metadata(self):
        for i, chunk in enumerate(self.chunks):
            chunk.metadata['chunk_id'] = f"chunk_{i:04d}"
            chunk.metadata['chunk_index'] = i

            # 청크가 어떤 섹션에 속하는지 추론 (제목 기반)
            section = self._infer_section(chunk.page_content)
            if section:
                chunk.metadata['section'] = section

    def _infer_section(self, text: str) -> str:
        text_lower = text.lower()

        # 주요 섹션 키워드 매핑
        section_keywords = {
            '엔진': ['엔진', '시동', '출력', '연료'],
            '브레이크': ['브레이크', '제동', '페달'],
            '타이어': ['타이어', '휠', '공기압'],
            '전기장치': ['배터리', '퓨즈', '램프', '조명'],
            '안전': ['에어백', '안전벨트', '경고등'],
            '정비': ['점검', '교체', '정비', '오일'],
            '운전': ['주행', '운전', '기어', '변속'],
            'ADAS': ['크루즈', '차선', '충돌', '자동']
        }

        for section, keywords in section_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return section

        return '일반'

    def _get_avg_chunk_size(self) -> float:
        """평균 청크 크기 계산"""
        if not self.chunks:
            return 0
        return sum(len(c.page_content) for c in self.chunks) / len(self.chunks)

    def get_chunk_statistics(self) -> dict:
        if not self.chunks:
            return {}

        lengths = [len(c.page_content) for c in self.chunks]

        return {
            'total_chunks': len(self.chunks),
            'avg_size': sum(lengths) / len(lengths),
            'min_size': min(lengths),
            'max_size': max(lengths),
            'total_chars': sum(lengths),
            'sections': self._count_sections()
        }

    def _count_sections(self) -> dict:
        sections = {}
        for chunk in self.chunks:
            section = chunk.metadata.get('section', '일반')
            sections[section] = sections.get(section, 0) + 1
        return sections

    def search_chunks(self, keyword: str, limit: int = 5) -> List[Document]:
        results = []
        for chunk in self.chunks:
            if keyword.lower() in chunk.page_content.lower():
                results.append(chunk)
                if len(results) >= limit:
                    break
        return results


# 테스트 코드
if __name__ == "__main__":

    from document_loader import VehicleManualLoader
    import os

    # 현재 디렉토리 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    pdf_path = os.path.join(project_root, "data", "LX3_2026_ko_KR.pdf")

    print("=" * 60)
    print("차량 매뉴얼 텍스트 분할 테스트")
    print("=" * 60)

    try:
        # 1. PDF 로드
        print("\n1. PDF 로딩...")
        loader = VehicleManualLoader(pdf_path)
        documents = loader.load_pdf()

        # 2. 텍스트 분할
        print("\n2. 텍스트 분할 중...")
        splitter = VehicleManualTextSplitter(
            chunk_size=500,  # 한국어 기준 약 2-3문단
            chunk_overlap=100  # 문맥 유지를 위한 중복
        )
        chunks = splitter.split_documents(documents)

        # 3. 통계 출력
        print("\n3. 청크 통계:")
        stats = splitter.get_chunk_statistics()
        print(f"   - 총 청크 수: {stats['total_chunks']:,}개")
        print(f"   - 평균 크기: {stats['avg_size']:.0f}자")
        print(f"   - 최소/최대: {stats['min_size']}자 / {stats['max_size']}자")
        print(f"   - 총 문자 수: {stats['total_chars']:,}자")

        print("\n   섹션별 분포:")
        for section, count in stats['sections'].items():
            print(f"   - {section}: {count}개")

        # 4. 샘플 청크 확인
        print("\n4️. 샘플 청크 (처음 3개):")
        print("-" * 50)
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n[청크 {i + 1}]")
            print(f"ID: {chunk.metadata.get('chunk_id')}")
            print(f"섹션: {chunk.metadata.get('section')}")
            print(f"원본 페이지: {chunk.metadata.get('page', 'N/A')}")
            print(f"내용: {chunk.page_content[:150]}...")
            print("-" * 50)

        # 5. 키워드 검색 테스트
        print("\n5️. '엔진 오일' 검색 테스트:")
        results = splitter.search_chunks("엔진 오일", limit=3)
        print(f"   찾은 청크: {len(results)}개")
        for i, chunk in enumerate(results):
            print(f"   - 청크 {chunk.metadata['chunk_id']}: {chunk.page_content[:100]}...")

    except Exception as e:
        print(f" 에러 발생: {e}")
        import traceback

        traceback.print_exc()