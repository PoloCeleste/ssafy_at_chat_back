from KistiAPI import KistiAPI
from Embedding import KISTIDataProcessor
from UpdateScheduler import DataUpdateScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from langchain_upstage import ChatUpstage
from langchain.chains import RetrievalQA
import threading

class MessageRequest(BaseModel):
    message: str

class ChatMessage(BaseModel):
    role: str
    content: str

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수로 벡터 스토어와 스케줄러 저장
vector_stores = None
scheduler = None

def initialize_data():
    """초기 데이터 수집 및 벡터 스토어 생성"""
    global vector_stores, scheduler
    
    print("데이터 수집 중...")
    api = KistiAPI()
    
    # 데이터 수집
    api_results = {
        'news': api.get_weekly_news(),
        'trends': api.get_trend_analysis({"keyword": "디지털|AI|머신러닝|딥러닝|블록체인|IT|로봇|임베디드"}),
        'science': api.get_science_trend({"keyword": "2024"})
    }
    
    # 데이터 전처리
    processor = KISTIDataProcessor()
    print("\n데이터 전처리 중...")
    vector_stores = processor.process_and_create_stores(api_results)
    print(f"생성된 벡터 스토어: {list(vector_stores.keys())}")
    
    # 스케줄러 초기화 및 실행
    scheduler = DataUpdateScheduler(processor, vector_stores)
    threading.Thread(target=scheduler.schedule_daily_update, daemon=True).start()

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 초기화"""
    initialize_data()

@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    """채팅 엔드포인트"""
    chat_upstage = ChatUpstage(model='solar-pro')
    
    # Chroma 벡터 스토어 사용
    retriever = vector_stores['chroma'].as_retriever(
        search_type='mmr',
        search_kwargs={"k": 3}
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=chat_upstage,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    result = qa(req.message)
    return {"reply": result['result']}

@app.get("/health")
@app.get("/")
async def health_check():
    """헬스 체크"""
    return {
        "status": "ok",
        "vector_stores": list(vector_stores.keys()) if vector_stores else [],
        "last_update": scheduler.last_update.isoformat() if scheduler and scheduler.last_update else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
