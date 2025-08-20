import time
import json
import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import re
from collections import Counter


sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from dotenv import load_dotenv

load_dotenv()

from src.embeddings import VehicleManualEmbeddings
from src.rag_chain import VehicleManualRAG


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


@dataclass
class EvaluationMetrics:

    semantic_similarity: float  # 의미 유사도
    answer_relevance: float  # 답변 관련성
    faithfulness: float  # 원문 충실도
    completeness: float  # 답변 완전성
    response_time: float  # 응답 시간
    consistency: float  # 일관성 점수

    @property
    def overall_score(self) -> float:

        weights = {
            'semantic_similarity': 0.20,
            'answer_relevance': 0.25,
            'faithfulness': 0.25,
            'completeness': 0.15,
            'consistency': 0.15
        }

        score = (
                self.semantic_similarity * weights['semantic_similarity'] +
                self.answer_relevance * weights['answer_relevance'] +
                self.faithfulness * weights['faithfulness'] +
                self.completeness * weights['completeness'] +
                self.consistency * weights['consistency']
        )
        return score


class RAGEvaluator:
    """
    RAG 시스템 종합 평가 클래스

    평가 지표 선정 이유:
    1. Semantic Similarity: 답변이 질문의 의도를 얼마나 잘 이해했는지 측정
    2. Answer Relevance: 답변이 질문과 얼마나 관련있는지 측정
    3. Faithfulness: 답변이 소스 문서를 얼마나 정확히 반영하는지 측정
    4. Completeness: 답변이 필요한 정보를 모두 포함하는지 측정
    5. Consistency: 같은 질문에 대한 답변의 일관성 측정
    6. Response Time: 실시간 시스템 요구사항 충족 여부 측정
    """

    def __init__(self):
        """평가 시스템 초기화"""
        print("🔧 평가 시스템 초기화 중...")

        # 의미 유사도 측정을 위한 임베딩 모델
        self.embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        # RAG 시스템 초기화
        embedder = VehicleManualEmbeddings()
        vector_store = embedder.load_index()
        self.rag_system = VehicleManualRAG(vector_store, use_ollama=False)

        print("✅ 평가 시스템 준비 완료\n")

    def evaluate_semantic_similarity(self, question: str, answer: str) -> float:
        """
        1. 의미 유사도 평가 (Semantic Similarity Score)

        이유: 답변이 질문의 의도를 제대로 이해했는지 측정
        방법: 질문과 답변의 임베딩 벡터 간 코사인 유사도 계산

        Returns:
            0~1 사이의 유사도 점수
        """
        # 질문과 답변을 임베딩
        q_embedding = self.embed_model.encode([question])
        a_embedding = self.embed_model.encode([answer])

        # 코사인 유사도 계산
        similarity = cosine_similarity(q_embedding, a_embedding)[0][0]

        # 스케일 조정 (0.3~0.8 → 0~1)
        scaled_similarity = max(0, min(1, (similarity - 0.3) / 0.5))

        return scaled_similarity

    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """
        2. 답변 관련성 평가 (Answer Relevance Score)

        이유: 답변이 질문과 직접적으로 관련있는 내용을 담고 있는지 측정
        방법: 키워드 매칭 + 의미적 관련성 체크

        Returns:
            0~1 사이의 관련성 점수
        """
        # 질문에서 핵심 키워드 추출
        question_keywords = self._extract_keywords(question)
        answer_lower = answer.lower()

        # 키워드 매칭률 계산
        matched = sum(1 for kw in question_keywords if kw in answer_lower)
        keyword_score = matched / len(question_keywords) if question_keywords else 0

        # 답변이 "모르겠다", "찾을 수 없다" 등 부정적 응답인지 체크
        negative_responses = ['찾을 수 없', '모르겠', '정보가 없', '매뉴얼에 없']
        has_negative = any(neg in answer for neg in negative_responses)

        # 부정적 응답이면 관련성 낮음
        if has_negative:
            return max(0.2, keyword_score * 0.5)

        # 답변 길이가 너무 짧으면 관련성 의심
        if len(answer) < 20:
            return keyword_score * 0.7

        return min(1.0, keyword_score + 0.3)  # 기본 점수 0.3 부여

    def evaluate_faithfulness(self, answer: str, source_docs: List[Any]) -> float:
        """
        3. 원문 충실도 평가 (Faithfulness Score)

        이유: 답변이 소스 문서의 내용을 왜곡 없이 정확히 전달하는지 측정
        방법: 답변의 핵심 정보가 소스 문서에 존재하는지 검증

        Returns:
            0~1 사이의 충실도 점수
        """
        if not source_docs:
            return 0.3  # 소스가 없으면 낮은 점수

        # 소스 문서들을 하나의 텍스트로 결합
        source_text = " ".join([doc.page_content for doc in source_docs]).lower()

        # 답변에서 숫자, 단위, 고유명사 등 팩트 추출
        facts = self._extract_facts(answer)

        if not facts:
            # 팩트가 없으면 일반적인 텍스트 유사도로 평가
            answer_embedding = self.embed_model.encode([answer])
            source_embedding = self.embed_model.encode([source_text[:1000]])  # 처음 1000자만
            similarity = cosine_similarity(answer_embedding, source_embedding)[0][0]
            return max(0.5, similarity)  # 최소 0.5점

        # 팩트가 소스에 있는지 확인
        found_facts = sum(1 for fact in facts if fact.lower() in source_text)
        faithfulness = found_facts / len(facts)

        return faithfulness

    def evaluate_completeness(self, question: str, answer: str) -> float:
        """
        4. 답변 완전성 평가 (Completeness Score)

        이유: 사용자 질문에 필요한 모든 정보가 답변에 포함되었는지 측정
        방법: 질문 유형별 필수 요소 체크

        Returns:
            0~1 사이의 완전성 점수
        """
        score = 0.5  # 기본 점수

        # 질문 유형 파악
        question_lower = question.lower()

        # "어떻게" 질문 → 단계별 설명 필요
        if any(word in question_lower for word in ['어떻게', '방법', '절차']):
            # 순서 표시자나 단계 구분이 있는지 체크
            has_steps = any(marker in answer for marker in ['1.', '2.', '첫째', '둘째', '먼저', '다음'])
            if has_steps:
                score += 0.3
            if len(answer) > 100:  # 충분한 설명
                score += 0.2

        # "언제", "주기" 질문 → 구체적 수치 필요
        elif any(word in question_lower for word in ['언제', '주기', '얼마나', '몇']):
            # 숫자와 단위가 있는지 체크
            has_numbers = bool(re.search(r'\d+', answer))
            has_units = any(unit in answer for unit in ['km', '개월', '년', '일', 'psi', '시간'])
            if has_numbers and has_units:
                score += 0.5
            elif has_numbers or has_units:
                score += 0.3

        # "무엇", "뭐" 질문 → 정의나 설명 필요
        elif any(word in question_lower for word in ['무엇', '뭐', '뭔가요']):
            if len(answer) > 50:  # 충분한 설명
                score += 0.3
            if '입니다' in answer or '있습니다' in answer:  # 명확한 정의
                score += 0.2

        # 일반적인 완전성 체크
        else:
            # 답변 길이
            if len(answer) > 100:
                score += 0.3
            elif len(answer) > 50:
                score += 0.2

            # 구체적 정보 포함 여부
            if any(char.isdigit() for char in answer):
                score += 0.2

        return min(1.0, score)

    def evaluate_consistency(self, question: str, num_trials: int = 3) -> float:
        """
        5. 일관성 평가 (Consistency Score)

        이유: 같은 질문에 대해 일관된 답변을 제공하는지 측정 (신뢰성)
        방법: 같은 질문을 여러 번 하여 답변의 유사도 측정

        Returns:
            0~1 사이의 일관성 점수
        """
        if num_trials < 2:
            return 1.0

        answers = []
        for _ in range(num_trials):
            response = self.rag_system.answer_question(question)
            answers.append(response['answer'])
            time.sleep(0.1)  # API 호출 간격

        # 모든 답변 쌍의 유사도 계산
        similarities = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                emb1 = self.embed_model.encode([answers[i]])
                emb2 = self.embed_model.encode([answers[j]])
                sim = cosine_similarity(emb1, emb2)[0][0]
                similarities.append(sim)

        # 평균 유사도가 일관성 점수
        consistency = np.mean(similarities) if similarities else 1.0

        return consistency

    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 핵심 키워드 추출"""
        # 불용어 제거
        stopwords = ['은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '도', '로', '으로', '만', '까지']

        # 명사와 주요 단어 추출 (간단한 방법)
        words = re.findall(r'[가-힣]+|[a-zA-Z]+|\d+', text.lower())
        keywords = [w for w in words if len(w) > 1 and w not in stopwords]

        return keywords[:5]  # 상위 5개 키워드

    def _extract_facts(self, text: str) -> List[str]:
        """텍스트에서 팩트(숫자, 단위, 고유명사) 추출"""
        facts = []

        # 숫자와 단위 조합 (예: 5000km, 35psi)
        number_units = re.findall(r'\d+[가-힣a-zA-Z]+', text)
        facts.extend(number_units)

        # 독립적인 숫자
        numbers = re.findall(r'\d+', text)
        facts.extend(numbers)

        # 영어 약어 (대문자로 시작)
        abbreviations = re.findall(r'\b[A-Z]{2,}\b', text)
        facts.extend(abbreviations)

        return facts

    def evaluate_single_qa(self, question: str, check_consistency: bool = False) -> Dict[str, Any]:
        """
        단일 질문-답변 쌍에 대한 종합 평가
        """
        print(f"\n📝 평가 중: {question}")
        print("-" * 50)

        # RAG 시스템으로 답변 생성
        start_time = time.time()
        response = self.rag_system.answer_question(question)
        response_time = time.time() - start_time

        answer = response['answer']
        source_docs = response.get('source_documents', [])

        # 각 지표 평가
        metrics = EvaluationMetrics(
            semantic_similarity=self.evaluate_semantic_similarity(question, answer),
            answer_relevance=self.evaluate_answer_relevance(question, answer),
            faithfulness=self.evaluate_faithfulness(answer, source_docs),
            completeness=self.evaluate_completeness(question, answer),
            response_time=response_time,
            consistency=self.evaluate_consistency(question, 2) if check_consistency else 1.0
        )

        # 결과 출력
        print(f"  📊 의미 유사도: {metrics.semantic_similarity:.2%}")
        print(f"  📊 답변 관련성: {metrics.answer_relevance:.2%}")
        print(f"  📊 원문 충실도: {metrics.faithfulness:.2%}")
        print(f"  📊 답변 완전성: {metrics.completeness:.2%}")
        if check_consistency:
            print(f"  📊 일관성: {metrics.consistency:.2%}")
        print(f"  ⏱️  응답 시간: {metrics.response_time:.2f}초")
        print(f"  ✨ 종합 점수: {metrics.overall_score:.2%}")

        return {
            'question': question,
            'answer': answer[:200] + '...' if len(answer) > 200 else answer,
            'metrics': metrics,
            'source_pages': response.get('source_pages', [])
        }

    def evaluate_test_set(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        테스트 세트 전체 평가
        """
        print("\n" + "=" * 60)
        print("🔬 RAG 시스템 종합 평가 시작")
        print("=" * 60)

        all_results = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}]", end="")
            result = self.evaluate_single_qa(
                test_case['question'],
                check_consistency=test_case.get('check_consistency', False)
            )
            result['category'] = test_case.get('category', 'general')
            result['expected'] = test_case.get('expected', '')
            all_results.append(result)

        # 전체 통계 계산
        avg_metrics = self._calculate_average_metrics(all_results)
        category_analysis = self._analyze_by_category(all_results)

        # 종합 보고서
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(test_cases),
            'average_metrics': avg_metrics,
            'category_analysis': category_analysis,
            'detailed_results': all_results,
            'system_assessment': self._generate_assessment(avg_metrics)
        }

        return report

    def _calculate_average_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """평균 지표 계산"""
        metrics_sum = {
            'semantic_similarity': 0,
            'answer_relevance': 0,
            'faithfulness': 0,
            'completeness': 0,
            'consistency': 0,
            'response_time': 0,
            'overall_score': 0
        }

        for result in results:
            m = result['metrics']
            metrics_sum['semantic_similarity'] += m.semantic_similarity
            metrics_sum['answer_relevance'] += m.answer_relevance
            metrics_sum['faithfulness'] += m.faithfulness
            metrics_sum['completeness'] += m.completeness
            metrics_sum['consistency'] += m.consistency
            metrics_sum['response_time'] += m.response_time
            metrics_sum['overall_score'] += m.overall_score

        n = len(results)
        return {k: v / n for k, v in metrics_sum.items()}

    def _analyze_by_category(self, results: List[Dict]) -> Dict[str, Dict]:
        """카테고리별 분석"""
        categories = {}

        for result in results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result['metrics'])

        category_stats = {}
        for cat, metrics_list in categories.items():
            category_stats[cat] = {
                'count': len(metrics_list),
                'avg_overall': np.mean([m.overall_score for m in metrics_list]),
                'avg_response_time': np.mean([m.response_time for m in metrics_list])
            }

        return category_stats

    def _generate_assessment(self, avg_metrics: Dict[str, float]) -> Dict[str, Any]:
        """시스템 평가 및 개선 제안 생성"""
        assessment = {
            'overall_grade': '',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }

        # 종합 등급 부여
        overall = avg_metrics['overall_score']
        if overall >= 0.9:
            assessment['overall_grade'] = 'A+ (탁월함)'
        elif overall >= 0.8:
            assessment['overall_grade'] = 'A (우수함)'
        elif overall >= 0.7:
            assessment['overall_grade'] = 'B (양호함)'
        elif overall >= 0.6:
            assessment['overall_grade'] = 'C (개선 필요)'
        else:
            assessment['overall_grade'] = 'D (심각한 개선 필요)'

        # 강점 분석
        if avg_metrics['semantic_similarity'] >= 0.8:
            assessment['strengths'].append('질문 의도 파악 우수')
        if avg_metrics['faithfulness'] >= 0.8:
            assessment['strengths'].append('원문 기반 정확한 답변')
        if avg_metrics['response_time'] <= 1.0:
            assessment['strengths'].append('빠른 응답 속도')

        # 약점 분석
        if avg_metrics['answer_relevance'] < 0.7:
            assessment['weaknesses'].append('답변 관련성 부족')
            assessment['recommendations'].append('프롬프트 엔지니어링 개선 필요')

        if avg_metrics['completeness'] < 0.7:
            assessment['weaknesses'].append('불완전한 답변')
            assessment['recommendations'].append('검색 청크 수 증가 고려')

        if avg_metrics['consistency'] < 0.8:
            assessment['weaknesses'].append('일관성 부족')
            assessment['recommendations'].append('Temperature 파라미터 낮추기 (현재 0.3 → 0.1)')

        if avg_metrics['response_time'] > 2.0:
            assessment['weaknesses'].append('느린 응답 속도')
            assessment['recommendations'].append('캐싱 전략 도입 또는 모델 경량화')

        return assessment

    def save_evaluation_report(self, report: Dict[str, Any], filename: str = 'evaluation_report.json'):
        """평가 보고서 저장"""
        # JSON 직렬화를 위해 metrics 객체를 dict로 변환
        serializable_report = json.loads(
            json.dumps(report, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)))

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, ensure_ascii=False, indent=2)

        print(f"\n💾 평가 보고서 저장 완료: {filename}")

    def print_summary_report(self, report: Dict[str, Any]):
        """평가 요약 보고서 출력"""
        print("\n" + "=" * 60)
        print("📊 RAG 시스템 평가 요약 보고서")
        print("=" * 60)

        avg = report['average_metrics']

        print("\n### 🎯 종합 평가 지표")
        print(f"  • 의미 유사도: {avg['semantic_similarity']:.2%}")
        print(f"  • 답변 관련성: {avg['answer_relevance']:.2%}")
        print(f"  • 원문 충실도: {avg['faithfulness']:.2%}")
        print(f"  • 답변 완전성: {avg['completeness']:.2%}")
        print(f"  • 일관성: {avg['consistency']:.2%}")
        print(f"  • 평균 응답시간: {avg['response_time']:.2f}초")
        print(f"\n  ⭐ 종합 점수: {avg['overall_score']:.2%}")

        print("\n### 📈 카테고리별 성능")
        for cat, stats in report['category_analysis'].items():
            print(f"  • {cat}: {stats['avg_overall']:.2%} (평균 {stats['avg_response_time']:.2f}초)")

        assessment = report['system_assessment']
        print(f"\n### 🏆 시스템 평가 등급: {assessment['overall_grade']}")

        if assessment['strengths']:
            print("\n### ✅ 강점")
            for strength in assessment['strengths']:
                print(f"  • {strength}")

        if assessment['weaknesses']:
            print("\n### ⚠️ 개선 필요 사항")
            for weakness in assessment['weaknesses']:
                print(f"  • {weakness}")

        if assessment['recommendations']:
            print("\n### 💡 개선 제안")
            for rec in assessment['recommendations']:
                print(f"  • {rec}")

        print("\n" + "=" * 60)


def main():
    """메인 실행 함수"""

    # 평가할 테스트 케이스 정의
    test_cases = [
        {
            'question': '엔진오일 교체 주기는 얼마나 되나요?',
            'category': '정비',
            'expected': '15,000km 또는 12개월',
            'check_consistency': True
        },
        {
            'question': '타이어 적정 공기압은 얼마인가요?',
            'category': '타이어',
            'expected': '35psi'
        },
        {
            'question': '경고등이 켜졌을 때 어떻게 해야 하나요?',
            'category': '안전',
            'expected': '안전한 곳에 정차'
        },
        {
            'question': 'ADAS 기능을 설정하는 방법은?',
            'category': 'ADAS',
            'expected': '인포테인먼트 시스템에서 설정'
        },
        {
            'question': '브레이크 패드 교체 시기는?',
            'category': '정비',
            'expected': '30,000km'
        },
        {
            'question': '와이퍼를 어떻게 교체하나요?',
            'category': '정비',
            'expected': '와이퍼 암을 들어올려'
        }
    ]

    # 평가 시스템 실행
    evaluator = RAGEvaluator()
    report = evaluator.evaluate_test_set(test_cases)

    # 보고서 출력 및 저장
    evaluator.print_summary_report(report)
    evaluator.save_evaluation_report(report)

    # 마크다운 생성
    print("\n### 📝 평가 결과")
    print("```")
    print(f"종합 점수: {report['average_metrics']['overall_score']:.2%}")
    print(f"평균 응답시간: {report['average_metrics']['response_time']:.2f}초")
    print(f"시스템 등급: {report['system_assessment']['overall_grade']}")
    print("```")


if __name__ == "__main__":
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY를 설정해주세요")
        api_key = input("API Key (sk-...): ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    main()