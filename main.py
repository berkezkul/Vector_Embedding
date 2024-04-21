'''

#ASTRA_DB  my_vector_db
import time

from langchain.indexes.vectorstore import VectorStoreIndexWrapper

ASTRA_DB_SECURE_BUNDLE_PATH = "secure-connect-my-vector-db.zip"
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:gvBSjvwOkcIsSZSYUURZjFZs:ba56e0ac2e3364fa356f16017cf5623a924e6a8dabeaa7a08dfc632a31d1d3aa"
ASTRA_DB_CLIENT_ID = "gvBSjvwOkcIsSZSYUURZjFZs"
ASTRA_DB_CLIENT_SECRET = "ncethQX4LU+_ZlqgPBLRpwMm.s7fBCtKLRtxdQOGP.bgSx1T.MJAwvYyeO7rHlEpJQ761.TZebT+YAJ4gnpytwkqZZKW_Qn.DQh52cCc_By-zZLKD6c3W1qmKz0jxi.A"

ASTRA_DB_KEYSPACE = "search"
OPENAI_API_KEY = "sk-proj-dKd017OhrTYudLf9ZnCUT3BlbkFJEZFA32Oy5eFK6ZagSSIb"

from langchain_community.llms import OpenAI as CommunityOpenAI
from langchain_community.embeddings import OpenAIEmbeddings as CommunityOpenAIEmbeddings, openai
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from datasets import load_dataset

# LangChainDeprecationWarning uyarılarını filtrelemek için
import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

cloud_config= {
    'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID, ASTRA_DB_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

# Yeni OpenAI ve OpenAIEmbeddings sınıflarını kullanarak LangChain nesnelerini başlatın
llm = CommunityOpenAI(openai_api_key=OPENAI_API_KEY)
myEmbedding = CommunityOpenAIEmbeddings(model="text-embedding-3-small", openai_api_key= OPENAI_API_KEY)

# VectorStoreIndexWrapper ve Cassandra sınıfını kullanımınızın doğru olduğundan emin olun
# Bu sınıfların kullanımı güncel langchain paketlerinde değişmiş olabilir
# myCassandraVStore gibi bir sınıf kullanıyorsanız, bu sınıfın güncel kullanımını kontrol edin

from langchain.globals import set_llm_cache
from langchain_community.cache import CassandraSemanticCache

test_sentence = "This is a sample sentence."
try:
    print(f"Embedding for test sentence: {test_sentence}")
    embedding_result = myEmbedding.embed_query(test_sentence)
    print(f"Embedding result: {embedding_result}")
except Exception as e:
    print(f"An error occurred while fetching the embedding: {e}")

set_llm_cache(CassandraSemanticCache(
    embedding=myEmbedding,
    session=astraSession,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name="first_table"
))


myCassandraVStore = VectorStoreIndexWrapper(
     embedding=myEmbedding,
     session=astraSession,
     keyspace=ASTRA_DB_KEYSPACE,
     table_name="first_table"
)



my_vector_store_index_wrapper = VectorStoreIndexWrapper(vectorstore=set_llm_cache)



# OpenAIEmbeddings'den alınan sonuçları saklamak için basit bir önbellek yapısı
embeddings_cache = {}

def get_embedding_with_cache(text):
    if text not in embeddings_cache:
        # Eğer önbellekte yoksa, OpenAI API'sini çağır
        try:
            embeddings_cache[text] = myEmbedding.embed_query(text)
        except openai.RateLimitError as e:
            # Rate limit hatası durumunda, kullanıcıyı bilgilendir
            print("Rate limit exceeded, please wait and retry. More info:", e)
            # Bu durumda işleme devam etmeyin ve bir sonraki çağrı için bekleme yapın
            time.sleep(60)
            return None  # veya uygun bir hata dönüş değeri
    return embeddings_cache[text]

# HuggingFace veri seti yükleme - Eğer gerçekten kullanılacaksa
print("Loading data from huggingface")
myDataset = load_dataset("Biddls/Onion_News", split="train")
headlines = myDataset["text"][:50]

# Embeddings oluşturma ve AstraDB'de saklama
print("\nGenerating embeddings and storing in AstraDB")
for headline in headlines:
    embedding = get_embedding_with_cache(headline)
    if embedding is not None:
        # my_vector_store_index_wrapper.add_text(headline, embedding)
        print(f"Inserted headline: {headline}")
        # Her API çağrısı sonrasında bekleme süresi ekleyin

'''


from fastapi import FastAPI, File, UploadFile, HTTPException
from embedding import Embedder
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import QdrantClient, VectorParams
from fastapi.responses import JSONResponse
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from qdrant_client import QdrantClient
from qdrant_client.http import models
import requests




app = FastAPI()
embedder = Embedder(api_key="sk-proj-uiAMhVTsyLE3Ht4hW2hAT3BlbkFJE22wI4GVSSWVWkPzrHWX")
qdrant_client = QdrantClient(host="localhost", port=6333)



QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections"


@app.post("/create_collection")
def create_collection(collection_name: str, vector_size: int = 128, distance: str = "Cosine"):
    vectors_config = models.VectorParams(
        size=vector_size,
        distance=distance
    )

    # Qdrant'a koleksiyon oluşturma isteği gönderiyoruz.
    response = qdrant_client.recreate_collection(collection_name=collection_name, vectors_config=vectors_config)

    return response




    """Yeni bir koleksiyon oluşturur.

    
    
    
    
    collection_schema = {
        "name": collection_name,
        "vector_size": vector_size,
        "distance": distance  # Kullanılacak mesafe metriği
    }

    # Koleksiyon oluşturmak için POST isteği gönder
    response = requests.post(QDRANT_COLLECTION_URL, json=collection_schema)

    if response.status_code == 200:
        return {"status": "success", "data": response.text}"""


@app.get("/list_collections")
def list_collections():
    """Veritabanındaki tüm koleksiyonları listeler."""


    response = requests.get(QDRANT_COLLECTION_URL)

    if response.status_code == 200:
        # Başarılı yanıtı ve koleksiyonların listesini döndür
        return JSONResponse(status_code=200, content=response.json())
    else:
        # Bir hata oluştu
        raise HTTPException(status_code=response.status_code, detail=response.text)



def upload_vectors(collection_name, embeddings, ids):
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection_name}/points/batch"
    headers = {'Content-Type': 'application/json'}  # JSON header ekleniyor

    points = [{
        "id": id,
        "vector": embedding
    } for id, embedding in zip(ids, embeddings)]

    response = requests.put(url, json={"points": points}, headers=headers)

    if response.status_code != 200:
        # Hata mesajı ekleme
        print(f"Error: {response.status_code}, {response.text}")
        return {}

    try:
        return response.json()
    except ValueError:
        # JSON decode hatası durumunda boş bir sözlük döner
        print("Failed to decode JSON response.")
        return {}




@app.post("/upload_and_index")
async def upload_and_index(collection_name: str, file: UploadFile = File(...)):
    """PDF veya TXT dosyasını alır, içeriğini okur, gömmer ve Qdrant'a kaydeder."""
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    contentList = content.decode().split(" ")
    embeddings = embedder.generate_embeddings(contentList)

    ids = range(1, len(embeddings) + 1)
    upload_vectors(collection_name=collection_name, embeddings=embeddings, ids=ids)

    return {"status": "success", "collection_name": collection_name}


'''
    # Qdrant'a vektör ekleme
    for embedding in embeddings:
        # Vektörleri tek tek yükleyebiliriz veya toplu bir şekilde yüklemek için
        # Qdrant istemcisinin uygun fonksiyonlarını kullanabiliriz.
        qdrant_client.upsert(collection_name=collection_name,
                             wait=True,
                             points=...,
                             payload={"vector": embedding})
                             
                            
                            '''




from fastapi import HTTPException
@app.get("/vector_search")
def vector_search(query: str, collection_name: str):
    """Verilen sorgu metni için vektör benzerlik araması yapar."""
    # OpenAI ile sorgu metni için gömme vektörü üret
    query_embeddings = embedder.generate_embeddings([query])

    # Eğer gömme vektörü üretilemezse hata döndür
    if not query_embeddings:
        raise HTTPException(status_code=500, detail="Failed to generate embeddings for the query.")

    # Qdrant üzerinde arama yap
    try:

        # Gömme vektörünü Qdrant'a gönder
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embeddings[0]['vector'],  # İlk ve tek vektörü sorgu olarak gönder
            query_filter=None,
            limit=10
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate")
def calculate(operation: str, x: float, y: float) -> float:
    """Basit matematik işlemleri gerçekleştirir."""
    if operation == "multiply":
        return x * y
    elif operation == "sum":
        return x + y
    elif operation == "substract":
        return x - y
    elif operation == "division":
        return x / y
    else:
        raise HTTPException(status_code=400, detail="Unsupported operation")

# QdrantClient ve Langchain integrasyonunu detaylandır.


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm domainlerden gelen isteklere izin verir
    allow_credentials=True,
    allow_methods=["*"],  # Tüm HTTP metodlarına izin verir
    allow_headers=["*"],  # Tüm başlıklara izin verir
)





