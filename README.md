# Agentic AI 101

Beginner-to-pro guide to building agentic AI systems that do real work. This repo is a practical path from first principles to deployment and monetization.

**[📖 Read the Docs →](https://sehgalrishabh.github.io/Agentic-AI-101/)**

## Who This Is For

- Beginners who want a structured, practical path into agentic AI
- Developers who want to build reliable agents, not just demos
- Freelancers and builders looking to monetize agentic solutions

## How To Use This Repo

- Start with `README.md` for the big picture
- Follow `SUMMARY.md` as the learning path
- Each chapter lives in `docs/` with hands-on projects and checklists

## Roadmap

### Part I: The Foundation (Understanding the Brain)

Goal: Demystify the "magic" and understand the core components of an agent.

**Chapter 1: The Agentic Shift**

- Introduction to AI Agents
- What is an AI Agent? (vs. a standard Chatbot)
- Foundations You Must Know
- Architecture of an AI Agent
- Analogy: A smart assistant is like a chef in a busy kitchen. The chef observes the order (perception), decides what to cook (reasoning), uses tools like knives and stoves (action), and tastes the result (feedback).
- The Loop: Perception -> Reasoning (Brain) -> Action (Tools) -> Feedback
- The Economy of Agents: Why companies are paying top dollar for developers who can build agents that do work, not just talk about it

**Chapter 2: Tools & Tech Stack (Your Toolkit)**

- Python Essentials: Crash course on the specific libraries you need (Requests, Pydantic, Pandas)
- The Brain (LLMs): Understanding API calls to OpenAI (GPT-4), Anthropic (Claude), and open-source models (Llama)
- The Skeleton (Frameworks):
  - LangChain / LangGraph: The industry standard for orchestration
  - CrewAI: Best for multi-agent role-playing
  - AutoGen: Microsoft’s framework for autonomous conversation

### Part II: The Builder’s Workshop (Code & Create)

Goal: Get hands dirty building functional agents.

**Chapter 3: Your First Agent (The "Hello World")**

- Project: A Simple "Search & Summarize" Agent
- Skills:
  - Setting up an Environment (.env, API keys)
  - Building a Tool: Integrating a search API (e.g., Tavily or Serper)
  - Writing the Prompt: Teaching the LLM when to use the tool

**Chapter 4: Memory & Context (Giving the Agent a Brain)**

- Short-term vs. Long-term Memory: Why agents forget and how to fix it
- RAG (Retrieval Augmented Generation)
- Project: A "PDF Chat" agent that can read a user's uploaded manual and answer technical support questions
- Vector Databases: Introduction to Pinecone or ChromaDB

**Chapter 5: Tool Use, Function Calling & Integrations**

- The Hands of the Agent: How to connect an LLM to the real world
- Project: A "Calendar Assistant" that interacts with Google Calendar API to check availability and book meetings
- Structured Output: Using Pydantic to ensure the agent outputs clean JSON, not rambling text

### Part III: Advanced Architectures (Mastering the Skill)

Goal: Build robust systems that don't break.

**Chapter 6: Multi-Agent Systems (Orchestration)**

- The Manager-Worker Model: One agent plans, others execute
- Project: A "Marketing Agency" in a Box
  - Agent A (Researcher): Scrapes trends
  - Agent B (Copywriter): Writes a tweet based on trends
  - Agent C (Editor): Critiques the tweet for tone and compliance
- Framework Focus: Deep dive into CrewAI or LangGraph for state management

**Chapter 7: Autonomous Loops & Self-Correction**

- Reflection: Teaching the agent to critique its own work before showing the user
- Error Handling: What happens when an API fails? (Retry logic, fallback models)
- Human-in-the-Loop: Designing breakpoints where the agent asks for permission before taking high-stakes actions (like sending an email)

### Part IV: Deployment & Production (Going Live)

Goal: Move from "works on my machine" to "works for the world."

**Chapter 8: Serving Your Agent**

- FastAPI: Wrapping your Python agent in a REST API so a frontend can talk to it
- Streamlit / Chainlit: Building a quick UI to demo your agent to clients
- Containerization: A crash course in Docker (packaging your agent)

**Chapter 9: Hosting & Cloud**

- Where to Host: Deploying on Railway, Render, or AWS Lambda
- Cost Management: Tracking token usage so you don't go broke (Observability with LangSmith)

**Chapter 10: Security and Cost Optimization**

### Part V: The Business of Agents (Monetization)

Goal: Turn code into cash.

**Chapter 11: Real-World Projects**

**Chapter 12: The Freelance Path**

- High-Demand Niches:
  - Customer Support Automation (reduce ticket volume)
  - Lead Generation & Outreach Agents (automated personalized emails)
  - Data Extraction & Entry Agents (killing spreadsheets)
- Pricing: How to charge for "value" (time saved) rather than "hours coded"

**Chapter 13: The SaaS Path**

- Micro-SaaS Ideas: Building a specialized agent for a specific industry (e.g., "Legal Contract Reviewer for Real Estate Agents")
- The Wrapper Trap: How to add proprietary data or unique workflows so you aren't just "reselling GPT-4"

**Chapter 14: Advanced Techniques**

**Chapter 15: Future-Proofing**

- Small Language Models (SLMs): Running agents locally on a user's laptop (Ollama)
- Voice Agents: The next frontier (Vapi, OpenAI Realtime API)

### Appendix: The "Cheat Sheets"

- Prompt Engineering Patterns: Templates for "Chain of Thought" and "ReAct" prompting
- Tool Repository: A list of 50+ APIs perfect for agent integration (Weather, Stocks, Slack, Notion, etc.)

## Repo Layout

- `README.md` is the landing page and roadmap
- `SUMMARY.md` is the full table of contents
- `docs/` contains all chapters
- `code/` contains project source code
- `diagrams/` holds system diagrams
- `assets/` stores images and media

## Contributing

PRs and issues are welcome. If you find gaps, add them. If you build a project, link it back here.

## License

Add your license in `LICENSE`.
