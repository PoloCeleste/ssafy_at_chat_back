from typing import Dict, List
import numpy as np
from datetime import datetime
import json, time, logging, os, requests, random
from pathlib import Path
from sentence_transformers import CrossEncoder
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_upstage import ChatUpstage
from langchain.prompts import ChatPromptTemplate
import langchain
from dotenv import load_dotenv
from pprint import pprint

from ragas.metrics import context_precision, context_recall
from ragas import evaluate
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from datasets import Dataset
from langchain_community.cache import SQLiteCache

class RAGEvaluator:
    def __init__(self, vector_stores):
        self.llm = ChatUpstage(model='solar-pro')
        self.embeddings = UpstageEmbeddings(model="embedding-query")
        self.vector_stores = vector_stores
        
    def evaluate_vector_stores(self, query_set: List[Dict]):
        results = {}
        
        for store_name, store in self.vector_stores.items():
            print(f"Evaluating {store_name}...")
            retriever = store.as_retriever(
                search_type='mmr',
                search_kwargs={"k": 3}
            )
            
            evaluation_data = {
                "question": [],
                "answer": [],
                "contexts": [],
                "ground_truth": []
            }
            
            # 각 쿼리에 대한 평가 데이터 생성
            for query in query_set:
                results = retriever.invoke(query['query'])
                contexts = [doc.page_content for doc in results]
                
                evaluation_data["question"].append(query['query'])
                evaluation_data["answer"].append("")
                evaluation_data["contexts"].append(contexts)
                evaluation_data["ground_truth"].append("")
            
            # RAGAS 평가 실행
            dataset = Dataset.from_dict(evaluation_data)
            score = evaluate(
                dataset=dataset,
                metrics=[
                    context_precision,
                    context_recall
                ],
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            results[store_name] = score
            
        return results

    def generate_random_queries(self, n_samples: int = 10) -> List[Dict]:
        """무작위 질문 샘플링"""
        query_templates = [
            "{}의 최신 트렌드는 무엇인가요?",
            "{}의 활용 사례를 설명해주세요",
            "{}가 산업에 미치는 영향은?",
            "{}의 핵심 기술은 무엇인가요?",
            "{}의 장단점을 설명해주세요"
        ]
        
        tech_keywords = [
            "인공지능", "블록체인", "클라우드", "빅데이터", 
            "IoT", "5G", "로보틱스", "메타버스",
            "양자컴퓨팅", "디지털트윈", "사이버보안", "엣지컴퓨팅"
        ]
        
        queries = []
        for _ in range(n_samples):
            template = random.choice(query_templates)
            keyword = random.choice(tech_keywords)
            queries.append({"query": template.format(keyword)})
            
        return queries



# class RAGEvaluator:
#     def __init__(self, evaluator, vector_stores):
#         self.llm = ChatUpstage(model='solar-pro')
#         # 캐시 설정
#         langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
        
#         # CrossEncoder 초기화
#         self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        
#         # 성능 측정 결과 저장소
#         self.metrics_store = {
#             'chroma': [],
#             'faiss': [],
#             'pinecone': []
#         }
        
#         self.evaluator = evaluator
#         self.vector_stores = vector_stores
#         self.best_params = None
#         self.best_score = 0
    
#     def generate_random_queries(self, n_samples: int = 10) -> List[Dict]:
#         """무작위 질문 샘플링"""
#         query_templates = [
#             "{}의 최신 트렌드는 무엇인가요?",
#             "{}의 활용 사례를 설명해주세요",
#             "{}가 산업에 미치는 영향은?",
#             "{}의 핵심 기술은 무엇인가요?",
#             "{}의 장단점을 설명해주세요"
#         ]
        
#         tech_keywords = [
#             "인공지능", "블록체인", "클라우드", "빅데이터", 
#             "IoT", "5G", "로보틱스", "메타버스", 
#             "양자컴퓨팅", "디지털트윈", "사이버보안", "엣지컴퓨팅"
#         ]
        
#         queries = []
#         for _ in range(n_samples):
#             template = random.choice(query_templates)
#             keyword = random.choice(tech_keywords)
#             queries.append({"query": template.format(keyword)})
            
#         return queries
    
#     def generate_answer(self, query: str, docs: List[Document]) -> str:
#         """문서 기반 답변 생성"""
#         try:
#             # 컨텍스트 생성
#             context = "\n\n".join([doc.page_content for doc in docs])
            
#             # 프롬프트 생성
#             messages = [
#                 {"role": "system", "content": "당신은 IT 트렌드 전문가입니다. 주어진 문서를 기반으로 최신 IT 트렌드와 기술 정보를 사용자의 수준에 맞게 설명해주세요."},
#                 {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
#             ]
#             # prompt = ChatPromptTemplate.from_messages(messages)
            
#             # 답변 생성
#             response = self.llm.generate(
#                 messages,
#                 temperature=0.7,
#                 max_tokens=500
#             )
            
#             # Groundness Check 검증
#             if self._verify_groundness(response, context):
#                 return response.generations[0][0].text
#             else:
#                 # 검증 실패 시 재시도
#                 return self.generate_answer(query, docs)
#         except Exception as e:
#             logging.error(f"answer gen error: {str(e)}")
#             return "답변 생성 중 오류가 발생했습니다."
    
#     def rerank_documents(self, query: str, docs: List[Document], top_k: int = 10) -> List[Document]:
#         """CrossEncoder를 사용한 문서 재정렬"""
#         if not docs:
#             return []
            
#         # 문서-쿼리 쌍 생성
#         pairs = [(query, doc.page_content) for doc in docs]
        
#         # 유사도 점수 계산
#         scores = self.reranker.predict(pairs)
        
#         # 점수로 정렬
#         doc_score_pairs = list(zip(docs, scores))
#         ranked_docs = [doc for doc, score in sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)]
        
#         return ranked_docs[:top_k]
    
#     def monitor_performance(self, vector_store, query: str, results: List[Document], response: str):
#         """RAG 시스템의 통합 성능 모니터링"""
#         start_time = time.time()
        
#         # RAGAS 평가 수행
#         evaluation = evaluate(
#             queries=[query],
#             contexts=[doc.page_content for doc in results],
#             answers=[response],
#             metrics=[
#                 faithfulness,
#                 answer_relevancy,
#                 context_precision,
#                 context_recall
#             ]
#         )
        
#         # 검색 성능 계산
#         latency = time.time() - start_time
#         similarity_scores = [r.metadata.get('score', 0) for r in results]
        
#         # 통합 메트릭스
#         metrics = {
#             'ragas_metrics': {
#                 'context_precision': evaluation['context_precision'],
#                 'context_recall': evaluation['context_recall'],
#                 'faithfulness': evaluation['faithfulness'],
#                 'answer_relevancy': evaluation['answer_relevancy']
#             },
#             'retrieval_metrics': {
#                 'latency': latency,
#                 'result_count': len(results),
#                 'avg_similarity_score': np.mean(similarity_scores),
#                 'max_similarity_score': max(similarity_scores, default=0)
#             },
#             'timestamp': datetime.now().isoformat()
#         }
        
#         # 목표 지표 달성 여부 확인
#         metrics['goals_achieved'] = {
#             'context_precision': metrics['ragas_metrics']['context_precision'] >= 0.7,
#             'context_recall': metrics['ragas_metrics']['context_recall'] >= 0.75,
#             'faithfulness': metrics['ragas_metrics']['faithfulness'] >= 0.8,
#             'answer_relevancy': metrics['ragas_metrics']['answer_relevancy'] >= 0.75
#         }
        
#         # 결과 로깅
#         logging.info(f"Query: {query}")
#         logging.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")
        
#         return metrics

#     def evaluate_rag_pipeline(self, vector_store, query_set: List[Dict]):
#         """RAG 파이프라인 성능 평가"""
#         from datasets import Dataset
        
#         evaluation_results = {
#             'queries': [],
#             'aggregate_metrics': {
#                 'faithfulness': [],
#                 'answer_relevancy': [],
#                 'context_precision': [],
#                 'context_recall': []
#             }
#         }
        
#         # 평가용 데이터셋 구성
#         eval_data = {
#             "question": [],
#             "answer": [],
#             "contexts": [],
#             "ground_truth": []
#         }
        
#         for query_item in query_set:
#             query = query_item['query']
            
#             # 문서 검색 및 재정렬
#             docs = vector_store.similarity_search(query, k=4)
#             reranked_docs = self.rerank_documents(query, docs)
            
#             # 답변 생성
#             response = self.generate_answer(query, reranked_docs)
            
#             # 데이터셋에 추가
#             eval_data["question"].append(query)
#             eval_data["answer"].append(response)
#             eval_data["contexts"].append([doc.page_content for doc in reranked_docs])
#             eval_data["ground_truth"].append("")
        
#         pprint(eval_data)
#         # Dataset 객체 생성
#         dataset = Dataset.from_dict(eval_data)
        
#         # RAGAS 평가 실행
#         evaluation = evaluate(
#             dataset=dataset,
#             metrics=[
#                 faithfulness,
#                 answer_relevancy,
#                 context_precision,
#                 context_recall
#             ]
#         )
        
#         return evaluation, eval_data
    
    
#     def compare_vector_stores(self, vector_stores: Dict, query_set: List[Dict]):
#         """벡터 스토어 성능 비교"""
#         results = {}
        
#         for store_name, store in vector_stores.items():
#             print(f"Evaluating {store_name}...")
#             metrics, eval_data = self.evaluate_rag_pipeline(store, query_set)
            
#             # 평가 결과를 직렬화 가능한 형태로 변환
#             serializable_metrics = {}
            
#             # metrics는 딕셔너리 형태로 반환됨
#             for metric_name in ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']:
#                 try:
#                     # 각 메트릭을 개별적으로 추출하고 평균 계산
#                     metric_values = metrics[metric_name]
#                     if hasattr(metric_values, 'mean'):
#                         serializable_metrics[metric_name] = float(metric_values.mean())
#                     else:
#                         serializable_metrics[metric_name] = float(np.mean(metric_values))
#                 except Exception as e:
#                     print(f"메트릭 처리 오류 ({metric_name}): {str(e)}")
#                     serializable_metrics[metric_name] = 0.0
            
#             results[store_name] = serializable_metrics
            
#             # 메트릭스 저장소 업데이트
#             self.metrics_store[store_name].append({
#                 'timestamp': datetime.now().isoformat(),
#                 'metrics': serializable_metrics
#             })
        
#         return results


    
#     def save_metrics(self, filepath: str = 'rag_metrics.json'):
#         """성능 측정 결과 저장"""
#         with open(filepath, 'w') as f:
#             json.dump(self.metrics_store, f)
    
#     def load_metrics(self, filepath: str = 'rag_metrics.json'):
#         """성능 측정 결과 로드"""
#         if Path(filepath).exists():
#             with open(filepath, 'r') as f:
#                 self.metrics_store = json.load(f)
    
    
    
#     def _verify_groundness(self, response: str, context: str) -> bool:
#         """Upstage Groundness Check API를 사용한 검증"""
#         try:
#             # 3번 검증 시도
#             verification_attempts = 3
#             for _ in range(verification_attempts):
#                 groundness_score = self._check_with_upstage_api(response, context)
#                 if groundness_score >= 0.8:  # 임계값 설정
#                     return True
#             return False
#         except Exception as e:
#             logging.error(f"Groundness check failed: {str(e)}")
#             return True  # API 오류 시 기본적으로 통과

#     def _check_with_upstage_api(self, response: str, context: str) -> float:
#         """Upstage API를 통한 Groundness 점수 계산"""
#         try:
#             checker = UpstageGroundnessChecker()
#             result = checker.check(response, context)
#             return result['groundness_score']
#         except Exception as e:
#             logging.error(f"Upstage API check failed: {str(e)}")
#             raise e
        
    
#     def optimize_hyperparameters(self):
#         """하이퍼파라미터 최적화"""
#         param_space = {
#             'chunk_size': [300, 500, 700],
#             'chunk_overlap': [30, 50, 70],
#             'similarity_top_k': [3, 4, 5],
#             'reranker_top_k': [8, 10, 12]
#         }
        
#         best_params = None
#         best_score = 0
        
#         for chunk_size in param_space['chunk_size']:
#             for chunk_overlap in param_space['chunk_overlap']:
#                 for top_k in param_space['similarity_top_k']:
#                     for reranker_k in param_space['reranker_top_k']:
#                         params = {
#                             'chunk_size': chunk_size,
#                             'chunk_overlap': chunk_overlap,
#                             'similarity_top_k': top_k,
#                             'reranker_top_k': reranker_k
#                         }
                        
#                         score = self._evaluate_params(params)
#                         if score > best_score:
#                             best_score = score
#                             best_params = params
        
#         self.best_params = best_params
#         self.best_score = best_score
#         return best_params

#     def _apply_params(self, params: Dict):
#         """하이퍼파라미터 적용"""
#         try:
#             # 청크 크기와 오버랩 설정
#             self.processor.chunk_size = params['chunk_size']
#             self.processor.chunk_overlap = params['chunk_overlap']
#             self.processor.text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=params['chunk_size'],
#                 chunk_overlap=params['chunk_overlap'],
#                 length_function=len,
#                 separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
#             )
            
#             # 검색 파라미터 설정
#             for store in self.vector_stores.values():
#                 store.search_kwargs.update({
#                     "k": params['similarity_top_k']
#                 })
                
#             # 재정렬 파라미터 설정
#             self.reranker_top_k = params['reranker_top_k']
            
#             return True
#         except Exception as e:
#             logging.error(f"parameter apply error: {str(e)}")
#             return False

#     def _evaluate_params(self, params: Dict) -> float:
#         """파라미터 성능 평가"""
#         try:
#             # 랜덤 쿼리 생성
#             test_queries = self.generate_random_queries()
            
#             # 파라미터 적용
#             self._apply_params(params)
            
#             # 성능 평가 - self.evaluator 대신 직접 평가
#             metrics, _ = self.evaluate_rag_pipeline(
#                 self.vector_stores['chroma'],
#                 test_queries
#             )
            
#             # 종합 점수 계산
#             score = (
#                 float(metrics['context_precision']) * 0.3 +
#                 float(metrics['context_recall']) * 0.2 +
#                 float(metrics['faithfulness']) * 0.3 +
#                 float(metrics['answer_relevancy']) * 0.2
#             )
            
#             return score
#         except Exception as e:
#             logging.error(f"parameter eval error: {str(e)}")
#             return 0.0

#     def optimize_pipeline(self):
#         """AutoRAG 파이프라인 최적화"""
#         # 하이퍼파라미터 최적화
#         best_params = self.optimize_hyperparameters()
        
#         # 벡터 스토어 재구성
#         self._rebuild_vector_stores(best_params)
        
#         # 성능 검증
#         test_queries = self.generate_random_queries()
#         final_results = {}
        
#         for store_name, store in self.vector_stores.items():
#             metrics = self.evaluator.evaluate_rag_pipeline(store, test_queries)
#             final_results[store_name] = metrics
        
#         return {
#             'best_params': best_params,
#             'final_results': final_results
#         }

#     def _rebuild_vector_stores(self, params: Dict):
#         """최적화된 파라미터로 벡터 스토어 재구성"""
#         for store_name, store in self.vector_stores.items():
#             if hasattr(store, 'update_parameters'):
#                 store.update_parameters(params)



# class UpstageGroundnessChecker:
#     def __init__(self):
#         load_dotenv()
#         self.api_key = os.getenv("UPSTAGE_API_KEY")
#         self.base_url = "https://api.upstage.ai/v1/solar/groundedness"
        
#     def check(self, response: str, context: str) -> Dict:
#         """
#         응답의 Groundness를 검증하는 API 호출
#         Args:
#             response: LLM이 생성한 응답
#             context: 응답 생성에 사용된 컨텍스트
#         Returns:
#             groundness_score: 0~1 사이의 점수
#             explanation: 검증 결과에 대한 설명
#         """
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
        
#         payload = {
#             "response": response,
#             "context": context
#         }
        
#         try:
#             api_response = requests.post(
#                 self.base_url,
#                 headers=headers,
#                 json=payload
#             )
            
#             if api_response.status_code == 200:
#                 result = api_response.json()
#                 return {
#                     'groundness_score': result.get('score', 0.0),
#                     'explanation': result.get('explanation', ''),
#                     'is_grounded': result.get('score', 0.0) >= 0.8
#                 }
#             else:
#                 print(f"API 호출 실패: {api_response.status_code}")
#                 return {
#                     'groundness_score': 0.0,
#                     'explanation': f"API 호출 실패: {api_response.status_code}",
#                     'is_grounded': False
#                 }
                
#         except Exception as e:
#             print(f"API 호출 중 오류 발생: {str(e)}")
#             return {
#                 'groundness_score': 0.0,
#                 'explanation': f"API 호출 중 오류: {str(e)}",
#                 'is_grounded': False
#             }