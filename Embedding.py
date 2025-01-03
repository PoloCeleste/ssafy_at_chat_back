from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv, set_key
from langchain.schema import Document
from typing import List, Dict
from getpass4 import getpass
from pprint import pprint
import os, logging

class KISTIDataProcessor:
    def __init__(self):
        # 임베딩 모델 초기화
        self.embeddings = UpstageEmbeddings(model="embedding-query")
        # 로깅 설정
        logging.basicConfig(
            filename='embedding_logs.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        # 텍스트 분할기 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        try:
            self.set_api_keys()
        except Exception as e:
            print(f"Error setting API keys: {str(e)}")
    
    def init_env_file(self):
        """환경 변수 파일이 없으면 생성"""
        if not os.path.exists('.env'):
            with open('.env', 'w') as f:
                f.write('')
        load_dotenv()

    def set_api_keys(self):
        """API 키 설정 및 .env 파일에 저장"""
        self.init_env_file()
        
        # Upstage API 키 설정
        if "UPSTAGE_API_KEY" not in os.environ or not os.environ["UPSTAGE_API_KEY"]:
            upstage_key = getpass("Enter your Upstage API key: ")
            os.environ["UPSTAGE_API_KEY"] = upstage_key
            set_key('.env', "UPSTAGE_API_KEY", upstage_key)
            print("Upstage API key has been set successfully.")
        
        # Pinecone API 키 설정
        if "PINECONE_API_KEY" not in os.environ or not os.environ["PINECONE_API_KEY"]:
            pinecone_key = getpass("Enter your Pinecone API key: ")
            os.environ["PINECONE_API_KEY"] = pinecone_key
            set_key('.env', "PINECONE_API_KEY", pinecone_key)
            print("Pinecone API key has been set successfully.")
        
    def process_news_data(self, news_data: Dict) -> List[Document]:
        """뉴스 데이터 전처리"""
        documents = []
        for record in news_data.get('records', []):
            content = f"{record.get('sj', '')}\n\n{record.get('contents', '')}"
            metadata = {
                'source': 'news',
                'publish_date': record.get('registDt', ''),
                'category': record.get('cdNm', ''),
                'url': record.get('originUrl', ''),
                'difficulty': self._estimate_difficulty(content)
            }
            documents.extend(self.text_splitter.create_documents([content], [metadata]))
        return documents

    def process_trend_data(self, trend_data: Dict) -> List[Document]:
        """트렌드 데이터 전처리"""
        documents = []
        for record in trend_data.get('records', []):
            content = f"{record.get('Title', '')}\n\n{record.get('Definition', '')}"
            # 키워드를 문자열로 변환
            keywords = self._parse_keywords(record.get('RelatedKeywords', ''))
            keywords_str = ', '.join(keywords)
            
            metadata = {
                'source': 'trend',
                'publish_date': self._format_date(record.get('PublDate', '')),
                'keywords': keywords_str,  # 리스트 대신 문자열로 저장
                'url': record.get('ContentURL', ''),
                'difficulty': self._estimate_difficulty(content)
            }
            documents.extend(self.text_splitter.create_documents([content], [metadata]))
        return documents
    
    def process_science_data(self, science_data: Dict) -> List[Document]:
        """과학향기/동향 데이터 전처리"""
        processed = []
        
        for record in science_data.get('records', []):
            content = ""
            if record.get('Title'):
                content += f"{record.get('Title')}\n\n"
            if record.get('Definition'):
                content += f"{record.get('Definition')}\n\n"
                
            # 키워드 리스트를 문자열로 변환
            keywords = self._parse_keywords(record.get('RelatedKeywords', ''))
            keywords_str = ', '.join(keywords) if keywords else ''
            
            metadata = {
                'source': 'science_trend',
                'publish_date': self._format_date(record.get('PublDate', '')),
                'keywords': keywords_str,  # 문자열로 저장
                'content_url': record.get('ContentURL', ''),
                'pdf_url': record.get('PdfURL', ''),
                'thumbnail_url': record.get('ThumbnailURL', ''),
                'difficulty': self._estimate_difficulty(content)
            }
            
            if content.strip():
                chunks = self.text_splitter.create_documents([content], [metadata])
                processed.extend(chunks)
                
        return processed
    
    def _format_date(self, date_str: str) -> str:
        """날짜 포맷 통일"""
        try:
            if not date_str:
                return ''
            # YYYYMMDD 형식을 YYYY-MM-DD로 변환
            if len(date_str) == 8:  # YYYYMMDD 형식인 경우
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            return date_str  # 이미 올바른 형식이면 그대로 반환
        except ValueError as e:
            print(f"Date format error: {str(e)}")
            return date_str
    
    def _estimate_difficulty(self, text: str) -> str:
        """텍스트 난이도 추정"""
        # 기술 용어
        basic_terms = ['AI', '클라우드', '빅데이터', 'API']
        intermediate_terms = ['딥러닝', '머신러닝', '블록체인', '양자']
        advanced_terms = ['트랜스포머', 'BERT', 'VAE', 'GAN']
        
        # 문장 복잡도 계산
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # 용어 빈도 계산
        basic_count = sum(1 for term in basic_terms if term in text)
        intermediate_count = sum(1 for term in intermediate_terms if term in text)
        advanced_count = sum(1 for term in advanced_terms if term in text)
        
        # 종합적 난이도 평가
        if advanced_count > 2 or (intermediate_count > 3 and avg_sentence_length > 20):
            return '고급'
        elif intermediate_count > 2 or (basic_count > 4 and avg_sentence_length > 15):
            return '중급'
        return '초급'
    
    def _parse_keywords(self, keywords_str: str) -> List[str]:
        """키워드 문자열 파싱"""
        try:
            keywords_str = keywords_str.strip('[]')
            return [k.strip().strip("'") for k in keywords_str.split(',') if k.strip()]
        except:
            return []
    
    def create_vector_stores(self, documents: List[Document]):
        """벡터 스토어 생성"""
        vector_stores = {}
        
        # Chroma 벡터 스토어 생성
        vector_stores['chroma'] = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # FAISS 벡터 스토어 생성
        faiss_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        # FAISS 저장
        faiss_store.save_local("./faiss_index")
        vector_stores['faiss'] = faiss_store
        
        # Pinecone 설정
        if os.getenv("PINECONE_API_KEY"):
            from pinecone import Pinecone, ServerlessSpec
            
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            
            try:
                # 인덱스 존재 여부 확인
                if 'kisti-rag' not in pc.list_indexes().names():
                    pc.create_index(
                        name='kisti-rag',
                        dimension=4096,
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region="us-east-1"
                        )
                    )
                    print("Created new Pinecone index: kisti-rag")
                
                # Pinecone 벡터 스토어 생성
                vector_stores['pinecone'] = PineconeVectorStore.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    index_name='kisti-rag'
                )
            except Exception as e:
                print(f"Pinecone error: {str(e)}")
        
        
        return vector_stores
    
    def process_and_create_stores(self, api_results):
        """API 결과 처리 및 벡터 스토어 생성"""
        processed_docs = []
        
        # 뉴스 데이터 처리
        if 'news' in api_results:
            processed_docs.extend(self.process_news_data(api_results['news']))
            print("news")
        # 트렌드 데이터 처리    
        if 'trends' in api_results:
            processed_docs.extend(self.process_trend_data(api_results['trends']))
            print("trend")
        # 과학향기 데이터 처리
        if 'science' in api_results:
            processed_docs.extend(self.process_science_data(api_results['science']))
            print("science")
        # 문서 분할
        split_docs = self.text_splitter.split_documents(processed_docs)
        
        print("\n벡터스토어 생성 중....")
        # 벡터 스토어 생성
        vector_stores = self.create_vector_stores(split_docs)
        
        return vector_stores

