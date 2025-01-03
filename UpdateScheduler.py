from langchain.schema import Document
from typing import List, Dict
from datetime import datetime
import schedule, time, json, os
import logging
from langchain_community.vectorstores import Chroma, FAISS
from KistiAPI import KistiAPI

class DataUpdateScheduler:
    def __init__(self, processor, vector_stores):
        self.processor = processor
        self.vector_stores = vector_stores
        self.last_update = None
        self.version_history = {}
        self.current_version = None
        self.backup_dir = "./vector_store_backups"
        self.update_time = "00:00"  # 매일 자정에 업데이트
        self.update_lock = False    # 업데이트 중복 실행 방지
        
        # 로깅 설정
        logging.basicConfig(
            filename='update_logs.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # 백업 디렉토리 생성
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
    
    def _create_backup(self):
        """현재 벡터 스토어의 백업 생성"""
        version = datetime.now().strftime('%Y%m%d_%H%M')
        backup_data = {
            'stores': {},
            'metadata': {
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'document_count': sum(len(store.get()) for store in self.vector_stores.values())
            }
        }
        
        try:
            # 각 벡터 스토어 백업
            for name, store in self.vector_stores.items():
                if isinstance(store, FAISS):
                    store_path = f"{self.backup_dir}/{version}_{name}.faiss"
                    store.save_local(store_path)
                elif isinstance(store, Chroma):
                    store_path = f"{self.backup_dir}/{version}_{name}"
                    store.persist(store_path)
                
                backup_data['stores'][name] = {
                    'path': store_path,
                    'type': store.__class__.__name__
                }
            
            # 백업 메타데이터 저장
            with open(f"{self.backup_dir}/{version}_metadata.json", 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            self.version_history[version] = backup_data
            self.current_version = version
            return version
            
        except Exception as e:
            print(f"Backup creation failed: {str(e)}")
            return None
    
    def _remove_duplicates(self, docs: List[Document]) -> List[Document]:
        """중복 문서 제거"""
        seen = set()
        unique_docs = []
        for doc in docs:
            doc_id = f"{doc.metadata['source']}_{doc.metadata['publish_date']}"
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        return unique_docs
    
    def _restore_backup(self, version: str):
        """특정 버전의 백업으로 복원"""
        if version not in self.version_history:
            raise ValueError(f"Version {version} not found in backup history")
            
        backup_data = self.version_history[version]
        restored_stores = {}
        
        try:
            for name, store_info in backup_data['stores'].items():
                if store_info['type'] == 'FAISS':
                    restored_stores[name] = FAISS.load_local(
                        store_info['path'],
                        self.processor.embeddings
                    )
                elif store_info['type'] == 'Chroma':
                    restored_stores[name] = Chroma(
                        persist_directory=store_info['path'],
                        embedding_function=self.processor.embeddings
                    )
            
            self.vector_stores = restored_stores
            self.current_version = version
            print(f"Successfully restored to version {version}")
            return True
            
        except Exception as e:
            print(f"Restore failed: {str(e)}")
            return False
    
    def update_data(self):
        """데이터 업데이트"""
        # 현재 상태 백업
        backup_version = self._create_backup()
        if not backup_version:
            print("Failed to create backup, aborting update")
            return False
            
        try:
            # API에서 새 데이터 가져오기
            api = KistiAPI()
            new_news = api.get_weekly_news()
            new_trends = api.get_trend_analysis({"keyword": "디지털|AI|머신러닝|딥러닝|블록체인|IT|로봇|임베디드"})
            new_science = api.get_science_trend({"keyword": "2024"})
            
            # 데이터 전처리
            processed_docs = []
            if new_news:
                processed_docs.extend(self.processor.process_news_data(new_news))
            if new_trends:
                processed_docs.extend(self.processor.process_trend_data(new_trends))
            if new_science:
                processed_docs.extend(self.processor.process_science_data(new_science))
            
            processed_docs = self._remove_duplicates(processed_docs)
            
            # 벡터 스토어 업데이트
            for store in self.vector_stores.values():
                store.add_documents(processed_docs)
            
            self.last_update = datetime.now()
            print(f"Data updated successfully at {self.last_update}")
            return True
            
        except Exception as e:
            print(f"Update failed: {str(e)}")
            print("Initiating rollback...")
            if self._restore_backup(backup_version):
                print("Rollback successful")
            else:
                print("Rollback failed, manual intervention required")
            return False
    
    def list_versions(self):
        """사용 가능한 백업 버전 목록 반환"""
        return [{
            'version': version,
            'timestamp': data['metadata']['timestamp'],
            'document_count': data['metadata']['document_count']
        } for version, data in self.version_history.items()]

    def run_evaluation(self):
        """주간 성능 평가 실행"""
        from RAGAS import RAGEvaluator
        evaluator = RAGEvaluator()
        
        test_queries = [
            {"query": "최신 AI 트렌드는 무엇인가요?"},
            {"query": "양자 컴퓨팅의 발전 현황은?"},
            {"query": "블록체인 기술의 실제 활용 사례는?"}
        ]
        
        results = evaluator.compare_vector_stores(self.vector_stores, test_queries)
        print(f"\n{datetime.now()} 성능 평가 결과:")
        print(json.dumps(results, indent=2))
    
    def schedule_daily_update(self):
        """일간 업데이트 스케줄 설정"""
        try:
            # 이전 스케줄 모두 제거
            schedule.clear()
            
            # 매일 지정된 시간에 업데이트 실행
            schedule.every().day.at(self.update_time).do(self._run_daily_update)
            
            logging.info(f"Daily update scheduled for {self.update_time}")
            
            while True:
                schedule.run_pending()
                time.sleep(60)
                
        except Exception as e:
            logging.error(f"Schedule error: {str(e)}")
            raise e
    
    def _run_daily_update(self):
        """일간 업데이트 실행"""
        if self.update_lock:
            logging.warning("Update already in progress")
            return
            
        self.update_lock = True
        try:
            logging.info("Starting daily update")
            
            # 현재 상태 백업
            backup_version = self._create_backup()
            if not backup_version:
                raise Exception("Backup creation failed")
            
            # 데이터 수집 및 처리
            api = KistiAPI()
            processed_docs = []
            
            # 뉴스 데이터
            news_data = api.get_weekly_news()
            if news_data:
                processed_docs.extend(self.processor.process_news_data(news_data))
            
            # 트렌드 데이터
            trend_data = api.get_trend_analysis({"keyword": "디지털|AI|머신러닝|딥러닝|블록체인|IT|로봇|임베디드"})
            if trend_data:
                processed_docs.extend(self.processor.process_trend_data(trend_data))
            
            # 과학향기 데이터
            science_data = api.get_science_trend({"keyword": "2024"})
            if science_data:
                processed_docs.extend(self.processor.process_science_data(science_data))
            
            # 중복 제거
            processed_docs = self._remove_duplicates(processed_docs)
            
            if not processed_docs:
                raise Exception("No new data collected")
            
            # 벡터 스토어 업데이트
            update_success = self._update_vector_stores(processed_docs)
            if not update_success:
                raise Exception("Vector store update failed")
            
            self.last_update = datetime.now()
            logging.info(f"Update completed successfully at {self.last_update}")
            
        except Exception as e:
            logging.error(f"Update failed: {str(e)}")
            if backup_version:
                logging.info("Initiating rollback...")
                self._restore_backup(backup_version)
                
        finally:
            self.update_lock = False
    
    def _update_vector_stores(self, docs: List[Document]) -> bool:
        """벡터 스토어 업데이트"""
        try:
            for store in self.vector_stores.values():
                store.add_documents(docs)
            return True
        except Exception as e:
            logging.error(f"Vector store update error: {str(e)}")
            return False
