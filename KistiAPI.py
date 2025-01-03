import requests, json, base64, re, os
from getpass4 import getpass
from Crypto.Cipher import AES
from urllib.parse import quote
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from pprint import pprint
from dotenv import load_dotenv, set_key

class KistiAPI:
    def __init__(self):
        # .env 파일 로드
        load_dotenv()
    
        # API 키 설정
        self._set_api_credentials()
        self.base_url = "https://apigateway.kisti.re.kr"
        
        # 토큰 초기화
        self._initialize_tokens()
        
    def _init_env_file(self):
        """환경 변수 파일이 없으면 생성"""
        if not os.path.exists('.env'):
            with open('.env', 'w') as f:
                f.write('')
            load_dotenv()
    
    def _set_api_credentials(self):
        """API 인증 정보 설정"""
        self._init_env_file()
        
        # 클라이언트 ID 설정
        if "KISTI_CLIENT_ID" not in os.environ or not os.environ["KISTI_CLIENT_ID"]:
            client_id = getpass("Enter your KISTI client ID: ")
            os.environ["KISTI_CLIENT_ID"] = client_id
            set_key('.env', "KISTI_CLIENT_ID", client_id)
        self.client_id = os.environ["KISTI_CLIENT_ID"]
        
        # AUTH KEY 설정
        if "KISTI_AUTH_KEY" not in os.environ or not os.environ["KISTI_AUTH_KEY"]:
            auth_key = getpass("Enter your KISTI auth key: ")
            os.environ["KISTI_AUTH_KEY"] = auth_key
            set_key('.env', "KISTI_AUTH_KEY", auth_key)
        self.auth_key = os.environ["KISTI_AUTH_KEY"]
        
        # MAC 주소 설정
        if "KISTI_MAC_ADDRESS" not in os.environ or not os.environ["KISTI_MAC_ADDRESS"]:
            mac_address = getpass("Enter your MAC address: ")
            os.environ["KISTI_MAC_ADDRESS"] = mac_address
            set_key('.env', "KISTI_MAC_ADDRESS", mac_address)
        self.mac_address = os.environ["KISTI_MAC_ADDRESS"]
    
    def _initialize_tokens(self):
        """토큰 초기화 및 검증"""
        self.access_token = os.getenv("KISTI_ACCESS_TOKEN")
        self.refresh_token = os.getenv("KISTI_REFRESH_TOKEN")
        
        try:
            access_token_expire = os.getenv("KISTI_ACCESS_TOKEN_EXPIRE")
            refresh_token_expire = os.getenv("KISTI_REFRESH_TOKEN_EXPIRE")
            
            if access_token_expire:
                self.access_token_expire = datetime.strptime(
                    access_token_expire, 
                    "%Y-%m-%d %H:%M:%S.%f"
                )
            else:
                self.access_token_expire = None
                
            if refresh_token_expire:
                self.refresh_token_expire = datetime.strptime(
                    refresh_token_expire, 
                    "%Y-%m-%d %H:%M:%S.%f"
                )
            else:
                self.refresh_token_expire = None
                
        except Exception as e:
            print(f"Token expiration parsing error: {str(e)}")
            self.access_token_expire = None
            self.refresh_token_expire = None
    
    def _save_tokens(self):
        """토큰 정보를 환경 변수와 .env 파일에 저장"""
        # 환경 변수 업데이트
        os.environ["KISTI_ACCESS_TOKEN"] = self.access_token
        os.environ["KISTI_REFRESH_TOKEN"] = self.refresh_token
        os.environ["KISTI_ACCESS_TOKEN_EXPIRE"] = self.access_token_expire.strftime("%Y-%m-%d %H:%M:%S.%f")
        os.environ["KISTI_REFRESH_TOKEN_EXPIRE"] = self.refresh_token_expire.strftime("%Y-%m-%d %H:%M:%S.%f")
        
        # .env 파일 업데이트
        set_key('.env', "KISTI_ACCESS_TOKEN", self.access_token)
        set_key('.env', "KISTI_REFRESH_TOKEN", self.refresh_token)
        set_key('.env', "KISTI_ACCESS_TOKEN_EXPIRE", self.access_token_expire.strftime("%Y-%m-%d %H:%M:%S.%f"))
        set_key('.env', "KISTI_REFRESH_TOKEN_EXPIRE", self.refresh_token_expire.strftime("%Y-%m-%d %H:%M:%S.%f"))
        
    def _update_tokens(self, token_data):
        """토큰 정보 업데이트 및 저장"""
        self.access_token = token_data["access_token"]
        self.refresh_token = token_data["refresh_token"]
        self.access_token_expire = datetime.strptime(
            token_data["access_token_expire"], 
            "%Y-%m-%d %H:%M:%S.%f"
        )
        self.refresh_token_expire = datetime.strptime(
            token_data["refresh_token_expire"], 
            "%Y-%m-%d %H:%M:%S.%f"
        )
        self._save_tokens()
        
    def _encrypt_accounts(self):
        # AES-CBC 모드 사용
        iv = 'jvHJ1EFA0IXBrxxz'
        block_size = 16
        
        # JSON 형식으로 데이터 생성
        time = ''.join(re.findall(r"\d", datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        plain_txt = json.dumps({"datetime": time, "mac_address": self.mac_address}).replace(" ", "")
        # PKCS7 패딩
        number_of_bytes_to_pad = block_size - len(plain_txt) % block_size
        ascii_str = chr(number_of_bytes_to_pad)
        padded_txt = plain_txt + (number_of_bytes_to_pad * ascii_str)
        
        # AES-CBC 암호화
        cipher = AES.new(self.auth_key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
        encrypted = cipher.encrypt(padded_txt.encode('utf-8'))
        
        # Base64 인코딩 후 URL 인코딩
        encoded = base64.urlsafe_b64encode(encrypted).decode("utf-8")
        return encoded
    
    def get_initial_token(self):
        accounts = self._encrypt_accounts()
        url = f"{self.base_url}/tokenrequest.do"
        params = {
            "accounts": accounts,
            "client_id": self.client_id
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            token_data = response.json()
            self._update_tokens(token_data)
            return True
        return False
    
    def refresh_access_token(self):
        url = f"{self.base_url}/tokenrequest.do"
        params = {
            "refreshToken": self.refresh_token,
            "client_id": self.client_id
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            if 'errorCode' in response.text:
                return response.text
            token_data = response.json()
            self._update_tokens(token_data)
            return response.text
        return None
    
    def _check_tokens(self):
        now = datetime.now() - timedelta(hours=9)
        
        # 토큰이나 만료 시간이 없는 경우 새로운 토큰 발급
        if (self.access_token is None or 
            self.refresh_token is None or 
            self.access_token_expire is None or 
            self.refresh_token_expire is None):
            return self.get_initial_token()
            
        try:
            # 리프레시 토큰이 만료된 경우
            if now >= self.refresh_token_expire:
                print(f"Refresh Token Expired. Current: {now}, Expire: {self.refresh_token_expire}")
                return self.get_initial_token()
                
            # 액세스 토큰이 만료된 경우
            if now >= self.access_token_expire:
                print(f"Access Token Expired. Current: {now}, Expire: {self.access_token_expire}")
                return self.refresh_access_token()
                
            return True
            
        except TypeError as e:
            print(f"Token validation error: {str(e)}")
            return self.get_initial_token()

    
    def _parse_xml_to_dict(self, xml_element):
        """XML 응답을 딕셔너리로 변환"""
        result = {}
        
        # 레코드 리스트 추출
        records = []
        record_list = xml_element.find('recordList')
        if record_list is not None:
            for record in record_list.findall('record'):
                record_dict = {}
                for item in record:
                    meta_code = item.get('metaCode')
                    if meta_code:
                        record_dict[meta_code] = item.text.strip() if item.text else ''
                records.append(record_dict)
        
        # 결과 요약 정보 설정
        result['records'] = records
        result['total_count'] = str(len(records))
        result['display_count'] = str(len(records))
        
        return result
    

    def make_api_request(self, endpoint, params=None):
        if not self._check_tokens():
            return None
                
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        params.update({
            "client_id": self.client_id,
            "token": self.access_token,
            "version": "1.0"
        })
            
        response = requests.get(url, params=params)
        xml_root = ET.fromstring(response.text)
        
        # 에러 체크
        status_code = xml_root.find('resultSummary/statusCode')
        if status_code is not None and status_code.text != '200':
            error_code = xml_root.find('errorDetail/errorCode')
            if error_code is not None and error_code.text == 'E4103':
                access_token_response = self.refresh_access_token()
                if 'errorCode' in access_token_response:
                    self.get_initial_token()
                    return self.make_api_request(endpoint, params)
                return self.make_api_request(endpoint, params)
        return xml_root

    def get_science_trend(self, params=None):
        """동향/과학향기 API"""
        endpoint = "/openapicall.do"
        
        api_params = {
            "action": "search",
            "target": "ATT",
            "searchQuery": json.dumps({"BI": params.get("keyword", "")}),
            "curPage": "1",
            "rowCount": "100",
            "sortField": "pubyear"
        }
        
        xml_response = self.make_api_request(endpoint, api_params)
        if xml_response is not None:
            return self._parse_xml_to_dict(xml_response)
        return None

    def get_trend_analysis(self, params=None):
        """트렌드 분석 API"""
        endpoint = "/openapicall.do"
        
        api_params = {
            "action": "search",
            "target": "TREND",
            "searchQuery": json.dumps({"BI": params.get("keyword", "")}),
            "curPage": "1",
            "rowCount": "100"
        }
        
        xml_response = self.make_api_request(endpoint, api_params)
        if xml_response is not None:
            return self._parse_xml_to_dict(xml_response)
        return None
    
    def get_weekly_news(self, s_date=None):
        """금주의 과학기술 뉴스 API"""
        endpoint = "/openapicall.do"
        news_dates = []
        today = datetime.now()
        
        def get_news_for_date(search_date):
            """특정 날짜의 뉴스 조회"""
            api_params = {
                "action": "search",
                "target": "SNEWS",
                "searchQuery": json.dumps({"RD": search_date}),
            }
            xml_response = self.make_api_request(endpoint, api_params)
            if xml_response is not None:
                result = self._parse_xml_to_dict(xml_response)
                if result and int(result.get('total_count', '0')) > 0:
                    return result
            return None

        def find_latest_news_dates():
            """최근 2일치의 뉴스가 있는 날짜 찾기"""
            check_date = today
            found_dates = []
            max_attempts = 31  # 최대 14일까지만 확인
            
            while len(found_dates) < 10 and max_attempts > 0:
                date_str = check_date.strftime('%Y%m%d')
                # 월요일(0)이나 목요일(3)인 경우만 확인
                if check_date.weekday() in [0, 3]:
                    result = get_news_for_date(date_str)
                    if result:
                        found_dates.append(date_str)
                check_date -= timedelta(days=1)
                max_attempts -= 1
            
            return found_dates

        # 뉴스 날짜 찾기
        news_dates = find_latest_news_dates()
        
        # 찾은 날짜들의 뉴스 결과 합치기
        combined_result = {
            'total_count': '0',
            'display_count': '0',
            'records': []
        }
        
        for date in news_dates:
            result = get_news_for_date(date)
            if result:
                combined_result['records'].extend(result['records'])
                combined_result['total_count'] = str(int(combined_result['total_count']) + 
                                                int(result['total_count']))
                combined_result['display_count'] = str(len(combined_result['records']))
        
        return combined_result if combined_result['records'] else None
