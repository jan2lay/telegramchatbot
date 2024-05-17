'''
Embedding and store to persistent db Chromadb.
user gpt4all and llama3
'''

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

local_llm = 'llama3'



