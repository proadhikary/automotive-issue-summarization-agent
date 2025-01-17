import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = 'data/'
DB_FAISS_PATH = 'data/db_faiss'

# Create vector database
def create_vector_db():
    data = pd.read_csv('data/FLAT_RCL.csv')
    makes = data['make'].tolist()
    models = data['model'].tolist()
    years = data['year'].tolist()
    summaries = data['summary'].tolist()
    texts = [f"{make} {model} {year} {summary}" for make, model, year, summary in zip(makes, models, years, summaries)]

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cuda'})

    db = FAISS.from_texts(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()
