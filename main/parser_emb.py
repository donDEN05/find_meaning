from telethon import TelegramClient
import asyncio
import os
from dotenv import load_dotenv

# Замените 'API_ID' и 'API_HASH' на ваши значения

load_dotenv()
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
SESSION_PATH = os.getenv("SESSION_PATH", "session")
# Создаем клиента
client = TelegramClient(SESSION_PATH, API_ID, API_HASH)
async def main():
    # Подключение к серверу Telegram
    await client.start()
    channel_username = '@muqwal_m0rtem'

    async def get_messages(channel_username):
        # Получение последних 100 сообщений из канала
        async for message in client.iter_messages(channel_username, limit=100):
            print(message.text)

    asyncio.run(get_messages(channel_username))
# Запуск основного цикла событий
asyncio.run(main())