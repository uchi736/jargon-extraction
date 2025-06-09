from dotenv import load_dotenv; load_dotenv()

# debug_gemini.py
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os, sys

print("Python:", sys.version)
print("langchain-google-genai:", ChatGoogleGenerativeAI.__module__.split('.')[0])

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
print("LLM test:", llm.invoke("こんにちは、LangChain!"))

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",   # ✅
)
print("Embed test:", embeddings.embed_query("LangChain integration"))