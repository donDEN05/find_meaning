from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import torch

load_dotenv()

HF_MODEL = os.getenv('HF_MODEL')
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dialogs")

qdrant = QdrantClient(url=QDRANT_URL)
device = "cpu" # for GPU usage or "cpu" for CPU usage

if torch.cuda.is_available():
    device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
embedder = SentenceTransformer("intfloat/e5-small")
model = AutoModelForCausalLM.from_pretrained(HF_MODEL).to(device)

def souped(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    souped = soup.get_text(strip=True)
    return souped

def find_top_n_contexts(question, n):
    vec = embedder.encode(question).tolist()
    results = qdrant.search(collection_name=QDRANT_COLLECTION, using="dialog", query_vector=vec, limit=n)
    context = "\n".join([souped(p.payload["txt"]) for p in results])
    return context
    

def llm_answer(question, n):
    context = find_top_n_contexts(question, n)
    messages = [
            {"role": "system", "text": "Ты помогаешь находить важное в чате. Отвечай коротко, по делу. На русском языке"},
            {"role": "user", "text": f"Вопрос: {question}\nКонтекст:\n{context}"}
        ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.5, do_sample=True)
    print('Вопрос: ', question)
    print('Ответ :' , tokenizer.decode(outputs[0]))


while True:
    question = input('Введите ваш вопрос:')
    if question == 'stop':
        break
    else:
        llm_answer(question, 5)

