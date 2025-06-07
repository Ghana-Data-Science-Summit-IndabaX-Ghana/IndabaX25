import os
import pandas as pd
from typing import Iterator, AsyncIterator, List
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Load environment variables (e.g., for HuggingFace API key)
load_dotenv()

# CSV Loader
class KNUSTCsvDataLoader(BaseLoader):
    """A document loader that loads CSV documents."""
    def __init__(self, directory: str, encoding: str = 'latin1') -> None:
        """Initialize the loader with a directory.
        Args:
            directory: The path to the directory containing CSV files.
            encoding: The encoding to use for reading CSV files (default: 'latin1').
        """
        self.directory = directory
        self.encoding = encoding
    def load(self) -> List[Document]:
        return list(self.lazy_load())
    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader that reads CSV files row by row."""
        for filename in os.listdir(self.directory):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.directory, filename)
                try:
                    df = pd.read_csv(file_path, encoding=self.encoding)
                    required_columns = {"Subject", "Question", "Response"}
                    if not required_columns.issubset(df.columns):
                        raise ValueError(f"Missing required columns in {file_path}")
                    for chunk in pd.read_csv(file_path, chunksize=1000, encoding=self.encoding):
                        for row in chunk.itertuples():
                            yield Document(
                                page_content=f"Subject: {row.Subject}\nQuestion: {row.Question}\nResponse: {row.Response}",
                                metadata={"subject": row.Subject}
                            )
                except UnicodeDecodeError as e:
                    print(f"Encoding error in {file_path}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
    async def alazy_load(self) -> AsyncIterator[Document]:
        """An async lazy loader that reads CSV files row by row."""
        for filename in os.listdir(self.directory):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.directory, filename)
                try:
                    df = pd.read_csv(file_path, encoding=self.encoding)
                    required_columns = {"Subject", "Question", "Response"}
                    if not required_columns.issubset(df.columns):
                        raise ValueError(f"Missing required columns in {file_path}")
                    for chunk in pd.read_csv(file_path, chunksize=1000, encoding=self.encoding):
                        for row in chunk.itertuples():
                            yield Document(
                                page_content=f"Subject: {row.Subject}\nQuestion: {row.Question}\nResponse: {row.Response}",
                                metadata={"subject": row.Subject}
                            )
                except UnicodeDecodeError as e:
                    print(f"Encoding error in {file_path}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

# RAG Application
def build_rag_pipeline(directory: str):
    # Step 1: Load documents
    loader = KNUSTCsvDataLoader(directory, encoding='latin1')
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # Step 2: Split documents (optional, as your documents are likely short)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks")

    # Step 3: Create embeddings and vector store

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    print("Vector store created")

    # Step 4: Set up retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Step 5: Set up language model
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.7,
    )

    # Step 6: Define prompt template
    prompt_template = """Use the following pieces of context to answer the question. If you don't know the answer, say so.
    Context: {context}
    Question: {input}
    Answer: """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

    # Step 7: Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Step 8: Create RAG chain
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain

def query_rag(rag_chain, question: str):
    """Query the RAG pipeline and return the answer with sources."""
    result = rag_chain.invoke({"input": question})
    answer = result["answer"]
    sources = result["context"]
    return answer, sources

# Usage
if __name__ == "__main__":
    directory = "../data/"
    rag_chain = build_rag_pipeline(directory)

    # Example queries
    questions = [
        "Should time travelers be allowed to invest in the stock market?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        answer, sources = query_rag(rag_chain, question)
        print(f"Answer: {answer}")
        print("Sources:")
        for i, doc in enumerate(sources, 1):
            print(f"{i}. {doc.page_content[:100]}...")