# FloatChat - AI-Powered Conversational Interface for Ocean Data

FloatChat is a powerful, terminal-based chatbot that allows users to ask complex questions about ARGO float oceanographic data using natural language. It is designed to make vast scientific datasets accessible and queryable for researchers, students, and enthusiasts without requiring any programming knowledge.

This project uses a Retrieval-Augmented Generation (RAG) pipeline, combining semantic search with the reasoning capabilities of Large Language Models (LLMs) to provide accurate, context-aware answers.

## Key Features

- **Natural Language Queries**: Ask questions in plain English instead of writing complex code.
- **Semantic Search**: Understands the meaning of your questions, not just keywords.
- **Advanced Filtering**: Capable of answering specific questions about numerical data (e.g., "Find measurements with a temperature above 25 degrees").
- **Local & Private**: Runs entirely on your local machine using Ollama, so your data never leaves your computer.

## Tech Stack

- **Backend**: Python
- **AI Framework**: LangChain
- **LLM**: Ollama (with llama3:instruct)
- **Vector Database**: ChromaDB
- **Embedding Model**: Sentence-Transformers
- **Data Handling**: Pandas

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.9+: [Download Python](https://www.python.org/downloads/)
- Git: [Download Git](https://git-scm.com/downloads)
- Ollama: You must have the Ollama application installed and running. [Download Ollama](https://ollama.ai/)

### 1. Setup and Installation

First, clone the repository and set up your Python environment.

```bash
# 1. Clone the GitHub repository
git clone <your-repository-url>
cd float-chat

# 2. Create and activate a Python virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# 3. Install the required libraries
pip install -r requirements.txt

# 4. Download the AI model via Ollama
# This model is specifically tuned for following instructions
ollama pull llama3:instruct
```

## 🔧 How to Use the Chatbot

Running the chatbot is a two-step process. First, you ingest your data, then you run the chatbot to ask questions about it.

### Step 1: Data Ingestion (Run Once)

This step reads your CSV file, converts its content into a searchable format, and stores it in a local vector database. You only need to run this once, or whenever your data changes.

‼️ **Important: Customize the Script for Your Data**

Before you run the script, you must configure it to work with your specific CSV file.

- **Place Your Data**: Put your CSV file (e.g., my_argo_data.csv) into the root of the float-chat project folder.
- **Edit ingest.py**: Open the ingest.py file and make the following changes:
  - Update the filename: Change the CSV_FILE_PATH variable to match your file's name.
    ```python
    # In ingest.py
    CSV_FILE_PATH = "my_argo_data.csv"  # <-- CHANGE THIS
    ```
  - Customize the document creation: Modify the doc_text f-string to create a natural language sentence using the columns from your CSV.
    ```python
    # In ingest.py, inside the for loop
    doc_text = (
        f"Measurement from ARGO float platform {row.get('platform_number', 'N/A')}. "
        f"Data: Adjusted pressure is {row.get('pres_adjusted', 'N/A')} dbar, "
        f"and adjusted temperature is {row.get('temp_adjusted', 'N/A')} C."
    )  # <-- CUSTOMIZE THIS SENTENCE
    ```
  - Customize the metadata: Update the list of columns in the for loop to include the specific columns you want to be able to filter on.
    ```python
    # In ingest.py, inside the for loop
    for col in ['platform_number', 'pres_adjusted', 'temp_adjusted']:  # <-- CUSTOMIZE THESE COLUMNS
        if col in row and row[col] != "Not Available":
            metadata[col] = row[col]
    ```

- **Run the Ingestion Script**:
  ```bash
  python ingest.py
  ```

### Step 2: Run the Chatbot

Once your data is ingested, you can start the chatbot.

‼️ **Important: Customize the Chatbot Script**

Open chatbot.py and update the metadata_field_info list to match the columns you defined in the ingestion script. The name and description must be accurate.

```python
# In chatbot.py
metadata_field_info = [
    AttributeInfo(name="platform_number", description="The unique ID of the ARGO float", type="integer"),
    AttributeInfo(name="pres_adjusted", description="The adjusted water pressure in decibars", type="float"),
    # ... add all your other filterable columns here
]  # <-- CUSTOMIZE THIS LIST
```

- **Start the Ollama Server**: Make sure the Ollama application is running in the background.
- **Run the Chatbot Script**:
  ```bash
  python chatbot.py
  ```

You can now ask questions directly in your terminal.

### Example Questions

- What can you tell me about the data from platform number 1900158?
- Find measurements where the adjusted temperature is below 25 degrees.
- What is the adjusted pressure for row 10?

## Troubleshooting

- **ConnectionRefusedError**: This error means the Ollama server is not running. Start the Ollama application and try again.

## Project File Structure

```
float-chat/
├── venv/                 # Virtual environment folder
├── data/                 # (Optional) For storing data files
├── chroma_db/            # Local vector database (created automatically)
├── ingest.py             # Script to process your CSV and create the database
├── chatbot.py            # The main chatbot application script
├── requirements.txt      # List of Python libraries
└── README.md             # This file
