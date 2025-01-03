from KistiAPI import KistiAPI
from Embedding import KISTIDataProcessor
from RAGAS import RAGEvaluator
import json
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from langchain_upstage import ChatUpstage
from langchain.chains import RetrievalQA

class ChatMessage(BaseModel):
    role: str
    content: str


class AssistantRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]  # Entire conversation for naive mode


class MessageRequest(BaseModel):
    message: str


def main():
    # 1. API 초기화 및 데이터 수집
    print("데이터 수집 중...")
    api = KistiAPI()
    
    # 데이터 수집
    api_results = {
        'news': api.get_weekly_news(),
        'trends': api.get_trend_analysis({"keyword": "디지털|AI|머신러닝|딥러닝|블록체인|IT|로봇|임베디드"}),
        'science': api.get_science_trend({"keyword": "2024"})
    }
    
    # 2. 데이터 전처리
    processor = KISTIDataProcessor()
    
    # 3. 벡터 스토어 생성
    print("\n데이터 전처리 중...")
    vector_stores = processor.process_and_create_stores(api_results)
    print(f"생성된 벡터 스토어: {list(vector_stores.keys())}")

    return vector_stores


pinecone_retriever = main()['pinecone'].as_retriever(
    search_type='mmr',  # default : similarity(유사도) / mmr 알고리즘
    search_kwargs={"k": 3}  # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
)
chat_upstage = ChatUpstage()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    qa = RetrievalQA.from_chain_type(llm=chat_upstage,
                                    chain_type="stuff",
                                    retriever=pinecone_retriever,
                                    return_source_documents=True)

    result = qa(req.message)
    return {"reply": result['result']}

@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

    # # 4. RAG 평가기 초기화
    # evaluator = RAGEvaluator(vector_stores=vector_stores)

    # # 5. 테스트 쿼리 생성
    # print("\n랜덤 테스트 쿼리 생성 중...")
    # test_queries = evaluator.generate_random_queries(n_samples=10)
    # print("생성된 테스트 쿼리:")
    # for i, query in enumerate(test_queries, 1):
    #     print(f"{i}. {query['query']}")

    # # 6. 벡터 스토어 성능 비교
    # print("\n벡터 스토어 성능 평가 중...")
    # results = evaluator.compare_vector_stores(vector_stores, test_queries)

    # # 7. 결과 저장
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # results_file = f"evaluation_results_{timestamp}.json"
    # with open(results_file, 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)

    # # 8. 하이퍼파라미터 최적화
    # print("\n하이퍼파라미터 최적화 중...")
    # optimization_results = evaluator.optimize_pipeline()

    # # 9. 최종 결과 출력
    # print("\n=== 평가 결과 요약 ===")
    # for store_name, metrics in results.items():
    #     print(f"\n{store_name} 성능:")
    #     print(f"Context Precision: {metrics[0]['context_precision']:.3f}")
    #     print(f"Context Recall: {metrics[0]['context_recall']:.3f}")
    #     print(f"Faithfulness: {metrics[0]['faithfulness']:.3f}")
    #     print(f"Answer Relevancy: {metrics[0]['answer_relevancy']:.3f}")

    # print("\n최적 하이퍼파라미터:")
    # print(json.dumps(optimization_results['best_params'], indent=2))
