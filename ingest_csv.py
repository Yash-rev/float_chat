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

import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIRECTORY = "data"
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "argo_data_collection"

def ingest_multiple_csv():
    print(f"Scanning for CSV files in '{DATA_DIRECTORY}' directory...")
    
    all_dataframes = []
    for filename in os.listdir(DATA_DIRECTORY):
        if filename.endswith(".csv"):
            filepath = os.path.join(DATA_DIRECTORY, filename)
            print(f"  - Loading {filename}")
            try:
                df = pd.read_csv(filepath)
                all_dataframes.append(df)
            except Exception as e:
                print(f"  > Error reading {filepath}: {e}")
    
    if not all_dataframes:
        print("No CSV files found or loaded. Exiting.")
        return
        
    master_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nSuccessfully combined {len(all_dataframes)} files into a dataset with {len(master_df)} rows.")

    print("Data loaded. Cleaning and preparing data...")
    numeric_cols = ['longitude', 'latitude', 'pres_adjusted', 'temp_adjusted', 'psal_adjusted']
    for col in numeric_cols:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
    master_df.fillna("Not Available", inplace=True)

    print("Creating text documents and metadata...")
    documents = []
    metadatas = []
    ids = []
    
    for index, row in master_df.iterrows():
        doc_text = (
            f"Measurement from ARGO float platform {row.get('platform_number', 'N/A')}, cycle number {row.get('cycle_number', 'N/A')}. "
            f"Location: Latitude {row.get('latitude', 'N/A')}, Longitude {row.get('longitude', 'N/A')}. "
            f"Data: Adjusted pressure is {row.get('pres_adjusted', 'N/A')} dbar, "
            f"adjusted temperature is {row.get('temp_adjusted', 'N/A')} C, "
            f"and adjusted practical salinity is {row.get('psal_adjusted', 'N/A')}."
        )
        documents.append(doc_text)
        
        metadata = {'row_number': index + 1}
        for col in ['platform_number', 'cycle_number', 'longitude', 'latitude', 'pres_adjusted', 'temp_adjusted', 'psal_adjusted']:
            if col in row and row[col] != "Not Available":
                metadata[col] = row[col]
        metadatas.append(metadata)
        
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
    ingest_multiple_csv()