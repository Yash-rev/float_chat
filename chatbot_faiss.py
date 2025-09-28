from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import PromptTemplate

FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3:instruct"

def main():
    print("Initializing chatbot...")

    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    llm = OllamaLLM(model=LLM_MODEL_NAME)

    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embedding_function,
        allow_dangerous_deserialization=True
    )
    
    metadata_field_info = [
        AttributeInfo(name="row_number", description="The row number in the source CSV file", type="integer"),
        AttributeInfo(name="platform_number", description="The unique ID of the ARGO float platform", type="integer"),
    ]
    document_content_description = "A row from a CSV file containing an oceanographic measurement from a single ARGO float cycle."

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        document_content_description,
        metadata_field_info,
        verbose=True
    )

    template = """You are an expert AI assistant for oceanographic science, specializing in the ARGO float project. 
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't have enough information.

    Context: {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    print("Chatbot initialized. Type 'exit' to quit.")
    
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == 'exit':
            print("Exiting chatbot.")
            break
        
        result = qa_chain.invoke({"query": user_question})
        print("\nAnswer:", result["result"])

if __name__ == "__main__":
    main()