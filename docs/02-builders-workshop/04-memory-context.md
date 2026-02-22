# Chapter 4: Memory & Context

If the LLM is the processor (CPU), then **context is the RAM**. By default, LLMs have **amnesia**. Every API call is a brand-new event unless you deliberately pass memory forward.

This chapter shows how to give your agent a working memory, long-term memory, and retrieval using **LangChain-first examples** with clear visuals.

## What You Will Learn

- How short-term memory actually works (the pass-through trick)
- How to manage the context window (sliding window + summarization)
- How long-term memory is built with RAG
- When to use RAG vs MCP for context
- How to build a PDF Q&A agent with LangChain

## The Core Mental Model

- **Context** = what the model sees right now
- **Short-term memory** = conversation buffer
- **Long-term memory** = retrieval from a knowledge store

```mermaid
flowchart LR
  U[User Message] --> C[Context Window]
  C --> M[Model]
  M --> O[Response]
  C -. optional .-> S[Summarized Memory]
  C -. optional .-> R[Retrieved Docs]
```

---

## 1. Short-Term Memory (The Conversation Buffer)

Short-term memory is how the agent remembers what you said two turns ago. It is not magic. It is a **pass-through technique**.

Every turn, you send the **entire conversation** so far.

```mermaid
flowchart TD
  T1[Turn 1: User says Hi] --> T2[Turn 2: User+AI+User]
  T2 --> T3[Turn 3: User+AI+User+AI+User]
```

### Pass-Through Example

**Turn 1**

- Input: `User: Hi`
- Output: `AI: Hello`

**Turn 2**

- Input: `User: Hi, AI: Hello, User: My name is Sarah`
- Output: `AI: Nice to meet you, Sarah.`

**Turn 3**

- Input: `User: Hi, AI: Hello, User: My name is Sarah, AI: Nice to meet you, User: Who am I?`
- Output: `AI: You are Sarah.`

### The Context Window Problem

LLMs cannot accept infinite history. Each model has a **context window** (e.g., 16k, 128k tokens). Past that limit, the model either fails or the request becomes too expensive.

### Two Classic Solutions

**A. Sliding Window**  
Keep only the last N turns and drop the rest.

```mermaid
flowchart LR
  H[Full History] --> W[Last N Messages] --> M[Model]
```

```python
def sliding_window(messages, max_turns=10):
    return messages[-max_turns:]
```

**B. Summarization**  
Summarize older turns into a compact memory note.

```mermaid
flowchart LR
  H[Older History] --> S[Summarize]
  S --> C[Context Window]
  C --> M[Model]
```

```python
def summarize_history(llm, messages):
    prompt = (
        "Summarize the conversation in 5 bullet points. "
        "Preserve user goals and preferences. Omit small talk."
    )
    text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    return llm.invoke(prompt + "\n\n" + text).content
```

### LangChain Example: Conversation Buffer

This is the simplest short-term memory. It keeps all turns in memory and passes them through automatically.

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="My name is Sarah.")
conversation.predict(input="What is my name?")
```

## 2. Long-Term Memory (RAG)

Short-term memory disappears when your script ends. Long-term memory persists across sessions.

**RAG (Retrieval-Augmented Generation)** is the standard way to build long-term memory.

Think of RAG as an **open-book exam**:

- Standard LLM: answer from its own memory
- RAG agent: **search the library**, find the right page, then answer

```mermaid
flowchart LR
  Q[User Question] --> E[Embed Query]
  E --> R[Retrieve Top K Chunks]
  R --> P[Prompt + Chunks]
  P --> M[Model]
  M --> O[Grounded Answer]
```

### The RAG Pipeline

1. **Ingest**: Load documents (PDFs, text files, Notion pages)
2. **Chunk**: Split into smaller pieces (500-1000 words)
3. **Embed**: Convert chunks into vectors (numbers)
4. **Store**: Put vectors in a vector database
5. **Retrieve**: Fetch nearest chunks for a query
6. **Generate**: Answer based on retrieved context

### Vector Databases (Quick Start)

- **Pinecone** (cloud, managed)
- **Chroma** (local, easy dev)
- **FAISS** (local, fast)
- **PGVector** (PostgreSQL users)

## 3. The New Standard: MCP (Model Context Protocol)

In 2024-2025, a new standard emerged: **MCP (Model Context Protocol)**. Think of it as **USB-C for AI context**.

Before MCP, every integration was custom. If you wanted Google Drive + Slack + GitHub, you wrote custom code for each. That was **integration hell**.

### What MCP Does

- **MCP Server**: a connector that exposes data in a standard format
- **MCP Client**: your agent, which plugs into the server

```mermaid
flowchart LR
  A[Agent (MCP Client)] --> S[MCP Server]
  S --> D[Data Source: Drive/Slack/GitHub]
  D --> S --> A
```

### When to Use RAG vs MCP

- **Use RAG** for large, mostly static knowledge bases (manuals, policies, wikis)
- **Use MCP** for live systems and tools (databases, filesystems, APIs)

```mermaid
flowchart LR
  Q[Need Context?] --> D{Type of Source}
  D -- Static Docs --> RAG[RAG Pipeline]
  D -- Live System --> MCP[MCP Integration]
```

### Use Cases: RAG vs MCP

**RAG Use Cases (Static Knowledge)**

- Employee handbook Q&A
- Product manuals and troubleshooting
- Internal wiki and SOP lookup
- Compliance policy search
- Research paper summarization
- Customer support knowledge base

```mermaid
flowchart LR
  Q[User Question] --> RAG[Retrieve Docs]
  RAG --> A[Grounded Answer]
```

**MCP Use Cases (Live Systems)**

- Read and summarize files from a shared drive
- Query a database for real-time metrics
- Pull issues and PRs from GitHub
- Fetch tickets from a helpdesk system
- Read Slack or Teams threads for context
- Update CRM notes or create tasks

```mermaid
flowchart LR
  Q[User Request] --> MCP[Call MCP Server]
  MCP --> S[Live System]
  S --> MCP --> A[Agent Response]
```

## 4. Project: PDF Chat Agent (LangChain)

We will build a RAG agent that answers questions about a PDF.

### Install Dependencies

```bash
pip install langchain-community langchain-openai faiss-cpu pypdf
```

### Flow Overview

```mermaid
flowchart LR
  P[PDF File] --> L[Load]
  L --> C[Chunk]
  C --> E[Embed]
  E --> V[Vector Store]
  V --> R[Retriever]
  R --> QA[Retrieval QA Chain]
  QA --> A[Answer]
```

### Code: PDF Chat Agent

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# 1. Load the PDF
loader = PyPDFLoader("policy.pdf")
documents = loader.load()

# 2. Chunk the text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

# 3. Embed the chunks
embeddings = OpenAIEmbeddings()

# 4. Store in FAISS (local vector DB)
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Create the retriever
retriever = vectorstore.as_retriever()

# 6. Connect to LLM
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# 7. Ask a question
response = qa_chain.run("What is the vacation policy in this document?")
print(response)
```

### What Just Happened?

```mermaid
flowchart LR
  Q[User Question] --> R[Retriever Finds Chunks]
  R --> P[Prompt + Chunks]
  P --> M[LLM]
  M --> O[Answer]
```

The agent did **not** read the whole PDF. It found the most relevant chunk, inserted it into the prompt, and answered based on that evidence.

---

## 5. Putting It Together: Memory + RAG Agent (LangChain)

Now combine short-term memory with long-term memory.

```mermaid
flowchart LR
  U[User Question] --> H[Short-Term Memory]
  U --> R[Retriever]
  H --> P[Prompt]
  R --> P
  P --> M[Model]
  M --> O[Answer]
```

### Code: Memory + RAG

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Build vector store (same as before)
loader = PyPDFLoader("policy.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())

# Memory + Retrieval
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
llm = ChatOpenAI(temperature=0)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

chain.invoke({"question": "What is the vacation policy?"})
chain.invoke({"question": "How many days does it allow?"})
```

## Common Pitfalls

- Dumping too much history into the prompt
- Forgetting to summarize old turns
- Building RAG without citing sources
- Using memory for factual data instead of retrieval

---

## Checklist

- My agent keeps short-term context clean
- My agent summarizes or trims old history
- My agent retrieves documents before answering
- My responses are grounded in sources

---

## What Comes Next

In Chapter 5, you will build **multi-agent workflows** and learn how to coordinate agents for larger tasks.
