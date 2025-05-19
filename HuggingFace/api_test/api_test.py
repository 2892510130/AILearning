import requests

API_URL = "https://router.huggingface.co/nebius/v1/chat/completions"
token = ""
headers = {
    "Authorization": f"Bearer {token}",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

response = query({
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "model": "Qwen/Qwen3-4B-fast"
})

print(response["choices"][0]["message"])
    
