from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper

# load_dotenv()

# Initialize OpenAI client (make sure you have OPENAI_API_KEY in your environment variables)
# client = OpenAI()


tokenizer = OpenAITokenizerWrapper()  # Load our custom tokenizer for OpenAI
MAX_TOKENS = 8191  # text-embedding-3-large's maximum context length

# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------

converter = DocumentConverter()
result = converter.convert("https://arxiv.org/pdf/2408.09869")
# result = converter.convert("/home/maduro/Downloads/CV_Welerson_Maduro_Android.pdf")

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
# for chunk in chunks:
#     print(f'----> {chunk.text}')
chunks
len(chunks)
