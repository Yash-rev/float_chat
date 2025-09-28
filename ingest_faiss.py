import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIRECTORY = "data"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def ingest_with_faiss():
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
    
    object_cols = master_df.select_dtypes(include=['object']).columns
    master_df[object_cols] = master_df[object_cols].fillna("Not Available")

    print("Creating text documents and metadata...")
    documents = []
    metadatas = []
    
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
            if col in row and pd.notna(row[col]):
                metadata[col] = row[col].item() if hasattr(row[col], 'item') else row[col]
        metadatas.append(metadata)
        
    print(f"Created {len(documents)} high-quality text documents.")
    
    print("Initializing embedding model...")
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print("Creating FAISS index from documents... This may take a while.")
    vector_store = FAISS.from_texts(
        texts=documents,
        embedding=embedding_function,
        metadatas=metadatas
    )

    print("Saving FAISS index to disk...")
    vector_store.save_local(FAISS_INDEX_PATH)

    print(f"\n--- Data ingestion complete! FAISS index saved to '{FAISS_INDEX_PATH}'. ---")

if __name__ == "__main__":
    ingest_with_faiss()