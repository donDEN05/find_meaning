import os, asyncio
from dotenv import load_dotenv
from telethon import TelegramClient, events
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from telethon.tl.functions.messages import GetHistoryRequest
import httpx

load_dotenv()

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
SESSION_PATH = os.getenv("SESSION_PATH", "session")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "chat_messages")

LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

client = TelegramClient(session=SESSION_PATH, api_id=API_ID, api_hash=API_HASH)
qdrant = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer("intfloat/e5-small")

# создаём коллекцию (один раз)
try:
    qdrant.get_collection(COLLECTION)
except:
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

async def embed_text(text: str):
    return embedder.encode(text).tolist()

@client.on(events.NewMessage)
async def handler(event):
    # Рекомендуем добавить условие для проверки чата. В данной реализации сохраняются все сообщения
    text = await client(GetHistoryRequest(
			peer='@muqwal_m0rtem',
			offset_id=offset_msg,
			offset_date=None, add_offset=0,
			limit=limit_msg, max_id=0, min_id=0,
			hash=0))
    if not text:
        return
    emb = await embed_text(text)
    point = PointStruct(
        id=event.message.id,
        vector=emb,
        payload={"text": text, "chat_id": event.chat_id}
    )
    qdrant.upsert(COLLECTION, points=[point])
    print(f"Сохранили сообщение: {text[:50]}...")

async def main():
    await client.start()
    print("Юзербот запущен")
    await client.run_until_disconnected()
    await client.disconnect()
if __name__ == "__main__":
    asyncio.run(main())