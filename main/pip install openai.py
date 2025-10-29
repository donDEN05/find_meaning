from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:12434/engines/llama.cpp/v1/chat/completions',
    api_key='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',
)
print('клиент создан')
dialog_history = [{
        "role": "system",
        "content": 'you are helpfull assistant, answer the questions',
    }]

while True:
    user_input = input("Введите ваше сообщение ('stop' для завершения): ")

    if user_input.lower() == "stop":
        break

    # Добавляем сообщение пользователя в историю диалога
    dialog_history.append({
        "role": "user",
        "content": user_input,
    })
    print('диалог добавлен вашей фразой')
    response = client.chat.completions.create(
        model="ai/smollm2",
        messages=dialog_history,
    )
    print('моделька подумала')
    # Извлекаем содержимое ответа
    response_content = response.choices[0].message.content
    print("Ответ модели:", response_content)

    # Добавляем ответ модели в историю диалога
    dialog_history.append({
        "role": "assistant",
        "content": response_content,
    })