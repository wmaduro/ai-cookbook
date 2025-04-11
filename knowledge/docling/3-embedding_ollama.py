from typing import List

import lancedb
import requests
import numpy as np
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from lancedb.pydantic import LanceModel, Vector
from utils.tokenizer import OpenAITokenizerWrapper

# --------------------------------------------------------------
# Custom Ollama embedding function
# --------------------------------------------------------------

class OllamaEmbedding:
    def __init__(self, model_name="nomic-embed-text", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        # Nomic-embed-text has 768 dimensions
        self._ndims = 768
        
    def ndims(self):
        return self._ndims
    
    def embed(self, texts):
        """Generate embeddings for a list of texts using Ollama API"""
        embeddings = []
        
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text}
            )
            
            if response.status_code == 200:
                embedding = response.json().get("embedding", [])
                embeddings.append(embedding)
            else:
                raise Exception(f"Error from Ollama API: {response.text}")
                
        return np.array(embeddings)
    
    def SourceField(self):
        return "text"
    
    def VectorField(self):
        return Vector(self._ndims)
    

# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------

tokenizer = OpenAITokenizerWrapper()  # We'll keep using this tokenizer for now
MAX_TOKENS = 8191  # Adjust this based on your Ollama model's context length

converter = DocumentConverter()
# result = converter.convert("https://arxiv.org/pdf/2408.09869")
result = converter.convert("/home/maduro/Downloads/CV_Welerson_Maduro_Android.pdf")


# --------------------------------------------------------------
# Apply hybrid chunking
# --------------------------------------------------------------

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)

chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)
print(chunks)
for chunk in chunks:
    print(f'----> {chunk.text}')

len(chunks)    


# --------------------------------------------------------------
# Create a LanceDB database and table
# --------------------------------------------------------------

# Create a LanceDB database
db = lancedb.connect("data/lancedb")

# Initialize our custom Ollama embedding function
# You can change the model to any embedding model available in your Ollama setup
func = OllamaEmbedding(model_name="nomic-embed-text")


# Define a simplified metadata schema
class ChunkMetadata(LanceModel):
    """
    You must order the fields in alphabetical order.
    This is a requirement of the Pydantic implementation.
    """

    filename: str | None
    page_numbers: List[int] | None
    title: str | None


# Define the main Schema
class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadata


table = db.create_table("docling", schema=Chunks, mode="overwrite")

# --------------------------------------------------------------
# Prepare the chunks for the table
# --------------------------------------------------------------

# Create table with processed chunks
processed_chunks = [
    {
        "text": chunk.text,
        "metadata": {
            "filename": chunk.meta.origin.filename,
            "page_numbers": [
                page_no
                for page_no in sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    )
                )
            ]
            or None,
            "title": chunk.meta.headings[0] if chunk.meta.headings else None,
        },
    }
    for chunk in chunks
]

# --------------------------------------------------------------
# Add the chunks to the table with custom embedding function
# --------------------------------------------------------------

# We need to manually embed the texts and add them to the table
texts = [chunk["text"] for chunk in processed_chunks]
embeddings = func.embed(texts)

# Add the embeddings to the processed chunks
for i, chunk in enumerate(processed_chunks):
    chunk["vector"] = embeddings[i]

# Add to table
table.add(processed_chunks)

# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

print(table.to_pandas())
print(table.count_rows())