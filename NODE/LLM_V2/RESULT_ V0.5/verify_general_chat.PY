import requests
import json

def test_general_chat():
    url = "http://localhost:8000/ask"
    
    # Test 1: Data context question
    payload1 = {
        "question": "이 시스템은 어떤 데이터를 가지고 있어?",
        "mode": "general"
    }
    
    print(f"Sending request 1: {payload1['question']}")
    try:
        response1 = requests.post(url, json=payload1)
        if response1.status_code == 200:
            print("Response 1 received:")
            print(response1.json()['answer'][:200] + "...")
        else:
            print(f"Error 1: {response1.status_code} - {response1.text}")
    except Exception as e:
        print(f"Exception 1: {e}")

    print("-" * 50)

    # Test 2: General greeting
    payload2 = {
        "question": "안녕, 반가워!",
        "mode": "general"
    }
    
    print(f"Sending request 2: {payload2['question']}")
    try:
        response2 = requests.post(url, json=payload2)
        if response2.status_code == 200:
            print("Response 2 received:")
            print(response2.json()['answer'])
        else:
            print(f"Error 2: {response2.status_code} - {response2.text}")
    except Exception as e:
        print(f"Exception 2: {e}")

if __name__ == "__main__":
    test_general_chat()
