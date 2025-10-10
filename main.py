import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

PDF_FOLDER_PATH = "./docs_to_analyze"
CHROMA_DB_PATH = "./chroma_db_store"
LLM_MODEL = "phi3"
EMBEDDING_MODEL = "nomic-embed-text"

# --- RAG Core Functions ---

def load_and_index():
    """
    1. Loads PDFs from the specified folder.
    2. Splits them into manageable chunks.
    3. Generates embeddings and saves them to ChromaDB.
    """
    print("--- Starting Document Loading and Indexing ---")

    # 1. Load Documents
    print(f"Loading documents from: {PDF_FOLDER_PATH}")
    if not os.path.exists(PDF_FOLDER_PATH):
        # Create the directory if it doesn't exist and exit, prompting the user to add files.
        os.makedirs(PDF_FOLDER_PATH)
        print(f"Created directory: {PDF_FOLDER_PATH}. Please place your PDFs here and run again.")
        return None

    # Use PyPDFDirectoryLoader to load all PDFs in the folder
    loader = PyPDFDirectoryLoader(PDF_FOLDER_PATH)
    try:
        documents = loader.load()
    except Exception as e:
        print(f"An error occurred during document loading (ensure 'pypdf' is installed and your PDFs are valid): {e}")
        return None

    if not documents:
        print("No documents found or loaded successfully. Please add PDFs to the directory.")
        return None
    print(f"Loaded {len(documents)} pages across all PDFs.")

    # 2. Split Documents
    print("Splitting documents into chunks...")
    # This splitter ensures chunks are manageable for the LLM's context window.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, # overlap helps retain context across splits
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} document chunks.")

    # 3. Create Embeddings and Vector Store
    print(f"Initializing Ollama Embeddings with model: {EMBEDDING_MODEL}")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print(f"Creating and persisting ChromaDB at: {CHROMA_DB_PATH}")
    # Chroma.from_documents converts the text chunks into vectors and saves them.
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    print("--- Indexing Complete! Vector Store is ready. ---")
    return vectorstore

def query_rag(llm, vectorstore: Chroma, query: str):
    """
    Performs Retrieval-Augmented Generation (RAG) to answer a query.
    """
    print(f"\n--- Running RAG Query: '{query}' ---")

    # 1. Setup Retriever
    # The retriever fetches the top K (5 in this case) most relevant documents based on the query.
    # "setting up" to fetch top K not actually "fetching" top K
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 2. Define the RAG Prompt Template
    system_prompt = (
        "You are an expert academic assistant. Use the following retrieved context "
        "to answer the user's question. If you don't know the answer, "
        "just say that you couldn't find the information in the provided documents. "
        "Always cite the source document filename (e.g., 'Source: paper_1.pdf') "
        "using the 'source' metadata field found in the context."
        "\n\nContext: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 3. Build the RAG Chain
    # Combines retrieved chunks into a single string to pass to the LLM
    document_chain = create_stuff_documents_chain(llm, prompt)
    # The main chain: runs retrieval, passes context to document_chain, gets final answer
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 4. Invoke the Chain
    response = retrieval_chain.invoke({"input": query})

    # 5. Output Result and Sources
    print("\n--- GENERATED ANSWER ---")
    print(response["answer"])

    # Extract unique source documents for citation
    unique_sources = set()
    for doc in response["context"]:
        source_path = doc.metadata.get('source')
        if source_path:
            unique_sources.add(os.path.basename(source_path))

    if unique_sources:
        print("\n--- SOURCES USED ---")
        print(" | ".join(sorted(unique_sources)))
    else:
        print("\n--- NO SOURCES CITED ---")

if __name__ == "__main__":
    llm = ChatOllama(model=LLM_MODEL)
    # response = llm.invoke ("hey whats up")
    # print(response.text)

    # If documents change, you may need to delete the './chroma_db_store' folder and re-run.
    vectorstore = load_and_index()
    if vectorstore:
        user_question = "What are the key methods used in the most recent paper?"
        query_rag(llm, vectorstore, user_question)

        user_question_2 = "Summarize the conclusion of the longest document."
        query_rag(llm, vectorstore, user_question_2)
