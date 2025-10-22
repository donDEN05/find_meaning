import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date 

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_DIALOGS = os.getenv("QDRANT_COLLECTION", "dialogs")
print('Переменные глобального окружения прочитаны...')

file_url = input('Введите ссылку на файл формата tsv: ')
if file_url == '':
    file_url = 'dialogues.tsv'
data = pd.DataFrame(pd.read_csv(file_url, sep='\t'))
print('Датасет прочтен...')

qdrant = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer("intfloat/e5-small")
print('Клиент СУБД и ембеддер созданы...')

try:
    qdrant.get_collection(COLLECTION_DIALOGS)
    print(f'БД {COLLECTION_DIALOGS} уже существует, будем добавлять в нее новые точки...')
except:
    qdrant.create_collection(
    collection_name=COLLECTION_DIALOGS,
    vectors_config={
        "user_1": VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
        "user_2": VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
        "dialog": VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    },
    )
    print(f'БД {COLLECTION_DIALOGS} не существует, будем создавать...')

def html_to_embed(text):
    soup = BeautifulSoup(text, 'html.parser')
    souped = soup.get_text(strip=True)
    return embedder.encode(souped).tolist()

data['persona_1_profile'] = data['persona_1_profile'].apply(html_to_embed)
data['persona_2_profile'] = data['persona_2_profile'].apply(html_to_embed)
data['dialogue'] = data['dialogue'].apply(html_to_embed)
data['id'] = data.index
print('Наша дата изменена...')

for i in data['id']:
    row = data.loc[i]
    qdrant.upsert(
        collection_name=COLLECTION_DIALOGS,
        points=[
            PointStruct(
                id=i,
                vector={
                    'user_1': row["persona_1_profile"],
                    'user_2': row["persona_2_profile"],
                    'dialog': row["dialogue"]
                },
                payload={
                    "dataset": file_url,
                    "data": date.today()})])
    
