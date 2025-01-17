from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

DB_FAISS_PATH = 'data/db_faiss'

custom_prompt_template = """You are a summarization agent that summarizes an automotive issue. Input to the agent comes in the form of a JSON object. Return short summary.
Context: {context}
Question: {question}

Summary:
"""

prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=['context', 'question']
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'}
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", token = "hf_yourtoken").to('cuda')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)
llm = HuggingFacePipeline(pipeline=pipe)

# Create the sum chain once
sum_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=False,
    chain_type_kwargs={'prompt': prompt}
)

def generate_response(query):
    response = sum_chain({'query': query})
    generated_text = response['result'].strip()
    if 'Summary:' in generated_text:
        summary = generated_text.split('Summary:')[-1].strip()
    else:
        summary = generated_text
    return summary
