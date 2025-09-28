# import chromadb
# from langchain_community.llms import Ollama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import SentenceTransformerEmbeddings

# CHROMA_DB_PATH = "./chroma_db"
# CHROMA_COLLECTION_NAME = "csv_data"

# def main():
#     print("Initializing chatbot...")

#     embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#     vector_store = Chroma(
#         client=chromadb.PersistentClient(path=CHROMA_DB_PATH),
#         collection_name=CHROMA_COLLECTION_NAME,
#         embedding_function=embedding_function,
#     )

#     llm = Ollama(model="llama3")

#     prompt_template = """
#     You are an AI assistant that answers questions about argo float data given through a csv file.
#     Use the following retrieved context to answer the question.
#     If you don't know the answer, just say that you don't know.
#     Keep the answer concise and relevant.

#     CONTEXT: {context}
#     QUESTION: {question}
#     ANSWER:
#     """
#     prompt = ChatPromptTemplate.from_template(prompt_template)
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vector_store.as_retriever(),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt}
#     )

#     print("You can now ask questions about your CSV data.")
#     print("Type 'exit' to quit.")

#     while True:
#         user_question = input("\nYour question: ")
#         if user_question.lower() == 'exit':
#             break
#         result = qa_chain.invoke({"query": user_question})
        
#         print("\nAnswer:", result["result"])

# if __name__ == "__main__":
#     main() 

from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import PromptTemplate

CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "argo_data_collection" 
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3:instruct"

def main():
    print("Initializing chatbot...")

    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    llm = OllamaLLM(model=LLM_MODEL_NAME)
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    
    metadata_field_info = [
        AttributeInfo(name="row_number", description="The row number in the source CSV file", type="integer"),
        AttributeInfo(name="platform_number", description="The unique ID of the ARGO float platform", type="integer"),
        AttributeInfo(name="cycle_number", description="The measurement cycle number for the float", type="integer"),
        AttributeInfo(name="longitude", description="The longitude coordinate of the measurement", type="float"),
        AttributeInfo(name="latitude", description="The latitude coordinate of the measurement", type="float"),
        AttributeInfo(name="pres_adjusted", description="The adjusted water pressure in decibars, related to depth", type="float"),
        AttributeInfo(name="temp_adjusted", description="The adjusted water temperature in degrees Celsius", type="float"),
        AttributeInfo(name="psal_adjusted", description="The adjusted practical salinity of the water", type="float"),
    ]
    document_content_description = "A row from a CSV file containing an oceanographic measurement from a single ARGO float cycle. Includes data like temperature, pressure, and salinity."

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        document_content_description,
        metadata_field_info,
        verbose=True
    )

    template = """You are an expert AI assistant for oceanographic science, specializing in the ARGO float project. 
    Use the following pieces of retrieved context from ARGO float data files to answer the question.
    If you don't know the answer from the context provided, just say that you don't have enough information. Do not make up an answer.
    Keep the answer concise and to the point.

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