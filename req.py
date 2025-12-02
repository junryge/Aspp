import requests
import json

def server_to_server_api_test():
    try:
        BASE_URL = "http://workplacedev.skhynix.com"
        GET_PATH = "/api/admin/v1/common/vgMemberByGroupId"
        TOKEN = "토큰을 넣어주세요."
        
        # 파라미터 설정
        param = {
            "SITE_ID": 1,
            "GRP_ID": 147144
        }
        
        # 헤더 설정
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "meta": json.dumps({"token": TOKEN})
        }
        
        # POST 요청
        response = requests.post(
            url=BASE_URL + GET_PATH,
            headers=headers,
            json=param,
            timeout=2  # 2초 타임아웃
        )
        
        print(f"Response Code : {response.status_code}")
        print(f"Response : {response.text}")
        
        # JSON으로 파싱하려면
        # result = response.json()
        
        return response
        
    except requests.exceptions.Timeout:
        print("타임아웃 발생")
    except requests.exceptions.RequestException as e:
        print(f"에러 발생: {e}")


if __name__ == "__main__":
    server_to_server_api_test()