import os
os.environ["GOOGLE_API_KEY"] = ".............."

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Configuration
video_id = "LPZh9BOjkQs"

print("Loading YouTube transcript...")
try:
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    text = " ".join([snippet.text for snippet in transcript.snippets])
    print(f"[OK] Transcript loaded: {len(text)} characters")
except TranscriptsDisabled:
    print("[ERROR] No captions available for this video")
    exit()

print("Processing text chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(text)
print(f"[OK] Text split into {len(chunks)} chunks")

print("Creating embeddings...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

chunk_documents = [Document(page_content=chunk) for chunk in chunks]
vector_store = FAISS.from_documents(chunk_documents, embedding)
print("[OK] Vector store created")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2
)

prompt = PromptTemplate(
    template="""
You are a helpful YouTube video assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient or doesn't contain the answer, say "I don't have enough information from the video to answer that question."

Context: {context}

Question: {question}

Answer:""",
    input_variables=['context','question']
)

def format_docs(retrieved_docs):
    """Format retrieved documents into a single context string"""
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

# Complete RAG chain
rag_chain = (
    RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

print("[OK] Chatbot ready!")
print("=" * 50)

# Test questions
test_questions = [
    "What is the main topic of this video?",
    "What are transformers in AI?",
    "How do large language models work?",
    "What is attention mechanism?"
]

for i, question in enumerate(test_questions, 1):
    print(f"\nTest {i}: {question}")
    print("Thinking...")
    try:
        answer = rag_chain.invoke(question)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error: {str(e)}")
    print("-" * 30)

print("\n[OK] All tests completed!")
print("\nTo run the interactive chatbot, use: python main.py")

