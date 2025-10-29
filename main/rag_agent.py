import requests

# URL API
url = "http://localhost:12434/engines/llama.cpp/v1/chat/completions"

# Данные запроса
payload = {
    "model": "ai/smollm2",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Расскажи о падении Рима."}
    ]
}

# Заголовки
headers = {
    "Content-Type": "application/json"
}

# Отправка POST-запроса
response = requests.post(url, json=payload, headers=headers)

# Вывод ответа
if response.status_code == 200:
    print(response.json())
else:
    print(f"Ошибка: {response.status_code}, {response.text}")
