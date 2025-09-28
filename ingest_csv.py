# import pandas as pd
# import chromadb
# from sentence_transformers import SentenceTransformer

# CSV_FILE_PATH = "20070101_prof.csv"  
# CHROMA_DB_PATH = "./chroma_db"
# CHROMA_COLLECTION_NAME = "csv_data"

# def ingest_csv():
#     print(f"Loading data from {CSV_FILE_PATH}...")
#     try:
#         df = pd.read_csv(CSV_FILE_PATH)
#     except FileNotFoundError:
#         print(f"Error: The file {CSV_FILE_PATH} was not found.")
#         return

#     print("Data loaded successfully. Creating text documents from rows...")
    
#     documents = []
#     metadatas = []
#     ids = []
    
#     for index, row in df.iterrows():
#         doc_text = f"Row {index + 1}: " + ", ".join([f"{col}: {val}" for col, val in row.items()])
#         documents.append(doc_text)
#         metadatas.append({'row_number': index + 1})
#         ids.append(f"row_{index + 1}")

#     print(f"Created {len(documents)} text documents.")
    
#     print("Initializing embedding model and ChromaDB...")
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
#     collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

#     print("Generating embeddings... This may take a while for large files.")
#     embeddings = model.encode(documents, show_progress_bar=True).tolist()
    
#     BATCH_SIZE = 4000
    
#     print(f"Adding {len(documents)} documents to ChromaDB in batches of {BATCH_SIZE}...")
#     for i in range(0, len(documents), BATCH_SIZE):
#         end_index = i + BATCH_SIZE
#         batch_documents = documents[i:end_index]
#         batch_embeddings = embeddings[i:end_index]
#         batch_metadatas = metadatas[i:end_index]
#         batch_ids = ids[i:end_index]
        
#         collection.add(
#             embeddings=batch_embeddings,
#             documents=batch_documents,
#             metadatas=batch_metadatas,
#             ids=batch_ids
#         )
#         print(f"  > Added batch {i // BATCH_SIZE + 1}/{(len(documents) // BATCH_SIZE) + 1}")

#     print("\n--- Data ingestion complete! ---")
#     print(f"Embeddings have been stored in the '{CHROMA_COLLECTION_NAME}' collection in ChromaDB.")
# if __name__ == "__main__":
#     ingest_csv()

# import pandas as pd
# import chromadb
# from sentence_transformers import SentenceTransformer

# CSV_FILE_PATH = "20070101_prof.csv"
# CHROMA_DB_PATH = "./chroma_db"
# CHROMA_COLLECTION_NAME = "csv_data"

# def ingest_csv():
#     print(f"Loading data from {CSV_FILE_PATH}...")
#     try:
#         df = pd.read_csv(CSV_FILE_PATH)
#     except FileNotFoundError:
#         print(f"Error: The file {CSV_FILE_PATH} was not found.")
#         return

#     print("Data loaded. Cleaning and preparing data...")
#     numeric_cols = ['your_numeric_column_1', 'your_numeric_column_2']
#     for col in numeric_cols:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#     df.fillna("Not Available", inplace=True)

#     print("Creating text documents and metadata...")
#     documents = []
#     metadatas = []
#     ids = []
    
#     for index, row in df.iterrows():
#         doc_text = (
#             f"At row {index + 1}, the record shows data for the category '{row.get('your_category_column', 'N/A')}'. "
#             f"The primary measurement was {row.get('your_numeric_column_1', 'N/A')} "
#             f"with a secondary reading of {row.get('your_numeric_column_2', 'N/A')}."
#         )
#         documents.append(doc_text)
        
#         metadata = {'row_number': index + 1}
#         for col in ['your_numeric_column_1', 'your_numeric_column_2', 'your_category_column']:
#             if col in row:
#                 metadata[col] = row[col]
#         metadatas.append(metadata)
        
#         ids.append(f"row_{index + 1}")

#     print(f"Created {len(documents)} high-quality text documents.")
    
#     print("Initializing embedding model...")
#     model = SentenceTransformer('all-MiniLM-L6-v2')
    
#     print("Generating embeddings...")
#     embeddings = model.encode(documents, show_progress_bar=True).tolist()
    
#     print("Initializing ChromaDB...")
#     client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
#     collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

#     BATCH_SIZE = 4000
#     print(f"Adding {len(documents)} documents to ChromaDB in batches...")
#     for i in range(0, len(documents), BATCH_SIZE):
#         end_index = min(i + BATCH_SIZE, len(documents))
#         collection.add(
#             ids=ids[i:end_index],
#             embeddings=embeddings[i:end_index],
#             documents=documents[i:end_index],
#             metadatas=metadatas[i:end_index]
#         )
#         print(f"  > Added batch {i // BATCH_SIZE + 1}/{(len(documents) - 1) // BATCH_SIZE + 1}")

#     print("\n--- Data ingestion complete! ---")

# if __name__ == "__main__":
#     ingest_csv()

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
# IMPORTANT: Make sure this path matches your CSV file's name.
CSV_FILE_PATH = "20070101_prof.csv" 
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "argo_data_collection"

def ingest_csv():
    """
    Reads ARGO data from a CSV and creates high-quality documents and metadata
    for an effective RAG pipeline.
    """
    print(f"Loading data from {CSV_FILE_PATH}...")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: The file {CSV_FILE_PATH} was not found.")
        return

    print("Data loaded. Cleaning and preparing data...")
    # Define the actual numeric columns from your CSV file.
    numeric_cols = ['longitude', 'latitude', 'pres_adjusted', 'temp_adjusted', 'psal_adjusted']
    for col in numeric_cols:
        if col in df.columns:
            # Convert columns to numeric, turning any errors into 'Not a Number' (NaN).
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Fill any remaining missing values so the script doesn't crash.
    df.fillna("Not Available", inplace=True)

    print("Creating text documents and metadata...")
    documents = []
    metadatas = []
    ids = []
    
    for index, row in df.iterrows():
        # Create a natural language sentence from the row's data for better semantic search.
        doc_text = (
            f"Measurement from ARGO float platform {row.get('platform_number', 'N/A')}, cycle number {row.get('cycle_number', 'N/A')}. "
            f"Location: Latitude {row.get('latitude', 'N/A')}, Longitude {row.get('longitude', 'N/A')}. "
            f"Data: Adjusted pressure is {row.get('pres_adjusted', 'N/A')} dbar, "
            f"adjusted temperature is {row.get('temp_adjusted', 'N/A')} C, "
            f"and adjusted practical salinity is {row.get('psal_adjusted', 'N/A')}."
        )
        documents.append(doc_text)
        
        # Create a dictionary of metadata for advanced filtering.
        metadata = {'row_number': index + 1}
        # Add all key columns to the metadata dictionary.
        for col in ['platform_number', 'cycle_number', 'longitude', 'latitude', 'pres_adjusted', 'temp_adjusted', 'psal_adjusted']:
            if col in row and row[col] != "Not Available":
                metadata[col] = row[col]
        metadatas.append(metadata)
        
        # Create a unique ID for each document.
        ids.append(f"row_{index + 1}")

    print(f"Created {len(documents)} high-quality text documents.")
    
    print("Initializing embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()
    
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    BATCH_SIZE = 4000
    print(f"Adding {len(documents)} documents to ChromaDB in batches...")
    for i in range(0, len(documents), BATCH_SIZE):
        end_index = min(i + BATCH_SIZE, len(documents))
        collection.add(
            ids=ids[i:end_index],
            embeddings=embeddings[i:end_index],
            documents=documents[i:end_index],
            metadatas=metadatas[i:end_index]
        )
        print(f"  > Added batch {i // BATCH_SIZE + 1}/{(len(documents) - 1) // BATCH_SIZE + 1}")

    print("\n--- Data ingestion complete! ---")

if __name__ == "__main__":
    ingest_csv()