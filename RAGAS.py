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
    
    def compare_vector_stores(self, vector_stores: Dict, query_set: List[Dict]):
        """벡터 스토어 성능 비교"""
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
                search_results = retriever.invoke(query['query'])
                contexts = [doc.page_content for doc in search_results]
                
                evaluation_data["question"].append(query['query'])
                evaluation_data["answer"].append("")
                evaluation_data["contexts"].append(contexts)
                evaluation_data["ground_truth"].append("")
            
            # RAGAS 평가 실행
            dataset = Dataset.from_dict(evaluation_data)
            evaluation = evaluate(
                dataset=dataset,
                metrics=[context_precision, context_recall],
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            try:
                serializable_metrics = {}
                for metric_name, values in evaluation.items():
                    if isinstance(values, (list, np.ndarray)):
                        serializable_metrics[metric_name] = float(np.mean(values))
                    else:
                        serializable_metrics[metric_name] = float(values)
            except Exception as e:
                print(f"metric error: {e}")
                serializable_metrics = {
                    'context_precision': 0.0,
                    'context_recall': 0.0
                }
            
            results[store_name] = serializable_metrics
        
        return results
    
    def optimize_pipeline(self):
        """RAG 파이프라인 최적화"""
        param_space = {
            'chunk_size': [300, 500, 700],
            'chunk_overlap': [30, 50, 70],
            'similarity_top_k': [3, 4, 5],
            'reranker_top_k': [8, 10, 12]
        }
        
        best_params = None
        best_score = 0
        
        for chunk_size in param_space['chunk_size']:
            for chunk_overlap in param_space['chunk_overlap']:
                for top_k in param_space['similarity_top_k']:
                    params = {
                        'chunk_size': chunk_size,
                        'chunk_overlap': chunk_overlap,
                        'similarity_top_k': top_k
                    }
                    
                    # 테스트 쿼리로 성능 평가
                    test_queries = self.generate_random_queries(n_samples=5)
                    results = self.compare_vector_stores(self.vector_stores, test_queries)
                    
                    # 평균 성능 계산
                    score = np.mean([
                        metrics['context_precision'] + metrics['context_recall']
                        for metrics in results.values()
                    ])
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
        
        return {
            'best_params': best_params,
            'best_score': best_score
        }

