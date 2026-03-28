# Appendix: The Cheat Sheets

A reference you will return to repeatedly. Three sections: prompt engineering patterns you can copy directly into any project, a curated API toolkit organized by use case, and a one-page architecture decision guide for every major choice in this book.

All code examples are provided in both Python and Node.js.

---

## Cheat Sheet 1: Prompt Engineering Patterns

### Pattern 1: Zero-Shot Instruction

The baseline. Clear task, clear constraints, clear output format.

**Template**

```
You are a [ROLE].

Your task: [TASK DESCRIPTION]

Rules:
- [CONSTRAINT 1]
- [CONSTRAINT 2]
- [CONSTRAINT 3]

Output format: [FORMAT DESCRIPTION]

Input: {input}
```

::: code-group
```python [Python]
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm    = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_template(
    "You are a customer support agent for a SaaS company.\n\n"
    "Your task: write a helpful reply to this support ticket.\n\n"
    "Rules:\n"
    "- Keep the reply under 100 words\n"
    "- Never say 'I cannot help with that'\n"
    "- Always end with an offer to help further\n\n"
    "Output format: plain text, no markdown\n\n"
    "Ticket: {ticket}"
)
chain  = prompt | llm
result = chain.invoke({"ticket": "My password reset email never arrived."})
print(result.content)
```
```javascript [Node.js]
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function zeroShot(ticket) {
  const response = await client.messages.create({
    model: "claude-sonnet-4-5",
    max_tokens: 256,
    messages: [
      {
        role: "user",
        content:
          `You are a customer support agent for a SaaS company.\n\n` +
          `Your task: write a helpful reply to this support ticket.\n\n` +
          `Rules:\n` +
          `- Keep the reply under 100 words\n` +
          `- Never say "I cannot help with that"\n` +
          `- Always end with an offer to help further\n\n` +
          `Output format: plain text, no markdown\n\n` +
          `Ticket: ${ticket}`,
      },
    ],
  });
  return response.content[0].text;
}

const reply = await zeroShot("My password reset email never arrived.");
console.log(reply);
```
:::

---

### Pattern 2: Chain of Thought (CoT)

Force step-by-step reasoning before the final answer. Most effective for arithmetic, multi-step logic, and decisions with multiple factors.

**Template**

```
[TASK DESCRIPTION]

Think step by step:
1. [STEP 1 DESCRIPTION]
2. [STEP 2 DESCRIPTION]
3. [STEP 3 DESCRIPTION]
4. State your final answer

Input: {input}
```

::: code-group
```python [Python]
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def chain_of_thought(invoice_text: str) -> str:
    return llm.invoke(
        f"Does this invoice total correctly?\n\n"
        f"Invoice: {invoice_text}\n\n"
        f"Think step by step:\n"
        f"1. List each line item and calculate its subtotal (qty × unit price)\n"
        f"2. Sum all line item subtotals\n"
        f"3. Add any taxes or fees\n"
        f"4. Compare to the stated total\n"
        f"5. State whether it is correct, and by how much it differs if not"
    ).content

result = chain_of_thought("3x Widget A @ $25 = $75, 2x Widget B @ $40 = $80, Tax $10, Total: $155")
print(result)
```
```javascript [Node.js]
import OpenAI from "openai";

const openai = new OpenAI();

async function chainOfThought(invoiceText) {
  const completion = await openai.chat.completions.create({
    model: "gpt-4o",
    temperature: 0,
    messages: [
      {
        role: "user",
        content:
          `Does this invoice total correctly?\n\n` +
          `Invoice: ${invoiceText}\n\n` +
          `Think step by step:\n` +
          `1. List each line item and calculate its subtotal (qty × unit price)\n` +
          `2. Sum all line item subtotals\n` +
          `3. Add any taxes or fees\n` +
          `4. Compare to the stated total\n` +
          `5. State whether it is correct, and by how much it differs if not`,
      },
    ],
  });
  return completion.choices[0].message.content;
}

const result = await chainOfThought(
  "3x Widget A @ $25 = $75, 2x Widget B @ $40 = $80, Tax $10, Total: $155",
);
console.log(result);
```
:::

---

### Pattern 3: Few-Shot Examples

Show the model the input/output pattern you want. Three to five examples is usually enough. More examples reduce variance.

**Template**

```
[TASK DESCRIPTION]

Examples:
Input: [EXAMPLE INPUT 1]
Output: [EXAMPLE OUTPUT 1]

Input: [EXAMPLE INPUT 2]
Output: [EXAMPLE OUTPUT 2]

Input: [EXAMPLE INPUT 3]
Output: [EXAMPLE OUTPUT 3]

Now do the same:
Input: {input}
Output:
```

::: code-group
```python [Python]
from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

llm = ChatOpenAI(model="gpt-4o", temperature=0)

examples = [
    {"input": "Server is down", "output": "technical"},
    {"input": "Charge me twice",  "output": "billing"},
    {"input": "How do I export data?", "output": "faq"},
    {"input": "I can't log in", "output": "account"},
    {"input": "Add dark mode", "output": "feature_request"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Classify this support ticket into one category: technical, billing, faq, account, feature_request.\n",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)

chain  = few_shot_prompt | llm
result = chain.invoke({"input": "My API key stopped working after upgrading my plan."})
print(result.content)  # "technical"
```
```javascript [Node.js]
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const examples = [
  { input: "Server is down", output: "technical" },
  { input: "Charge me twice", output: "billing" },
  { input: "How do I export data?", output: "faq" },
  { input: "I can't log in", output: "account" },
  { input: "Add dark mode", output: "feature_request" },
];

async function fewShotClassify(ticket) {
  const exampleBlock = examples
    .map((e) => `Input: ${e.input}\nOutput: ${e.output}`)
    .join("\n\n");

  const prompt =
    `Classify this support ticket into one category: ` +
    `technical, billing, faq, account, feature_request.\n\n` +
    `${exampleBlock}\n\n` +
    `Input: ${ticket}\nOutput:`;

  const response = await client.messages.create({
    model: "claude-sonnet-4-5",
    max_tokens: 10,
    messages: [{ role: "user", content: prompt }],
  });
  return response.content[0].text.trim();
}

console.log(
  await fewShotClassify("My API key stopped working after upgrading my plan."),
);
```
:::

---

### Pattern 4: ReAct (Reason + Act)

The backbone of tool-using agents. The model alternates between Thought (what do I need to do?) and Action (call a tool) until it has enough to answer.

**Template**

```
You have access to these tools:
{tools}

Use the following format:
Thought: [your reasoning about what to do next]
Action: [tool name]
Action Input: [tool input]
Observation: [tool output — filled in by the system]
... (repeat Thought/Action/Observation as needed)
Thought: I have enough information to answer.
Final Answer: [your answer]

Question: {input}
```

::: code-group
```python [Python]
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

llm = ChatOpenAI(model="gpt-4o", temperature=0)

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"72°F, partly cloudy in {city}"

@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights between two cities on a date (YYYY-MM-DD)."""
    return f"3 flights found from {origin} to {destination} on {date}: 8AM, 12PM, 6PM"

tools    = [get_weather, search_flights]
prompt   = hub.pull("hwchase17/react")
agent    = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

result = executor.invoke({
    "input": "I'm flying from NYC to London on July 10th. What's the weather there and what flights are available?"
})
print(result["output"])
```
```javascript [Node.js]
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const tools = [
  {
    name: "get_weather",
    description: "Get current weather for a city.",
    input_schema: {
      type: "object",
      properties: { city: { type: "string", description: "City name" } },
      required: ["city"],
    },
  },
  {
    name: "search_flights",
    description: "Search for flights between two cities on a given date.",
    input_schema: {
      type: "object",
      properties: {
        origin: { type: "string" },
        destination: { type: "string" },
        date: { type: "string", description: "YYYY-MM-DD" },
      },
      required: ["origin", "destination", "date"],
    },
  },
];

function executeTool(name, input) {
  if (name === "get_weather") return `72°F, partly cloudy in ${input.city}`;
  if (name === "search_flights")
    return `3 flights from ${input.origin} to ${input.destination} on ${input.date}: 8AM, 12PM, 6PM`;
  return "Tool not found";
}

async function reactAgent(userMessage) {
  const messages = [{ role: "user", content: userMessage }];

  while (true) {
    const response = await client.messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 1024,
      tools,
      messages,
    });

    messages.push({ role: "assistant", content: response.content });

    if (response.stop_reason === "end_turn") {
      const textBlock = response.content.find((b) => b.type === "text");
      return textBlock?.text ?? "";
    }

    // Process tool calls
    const toolResults = [];
    for (const block of response.content) {
      if (block.type === "tool_use") {
        const result = executeTool(block.name, block.input);
        console.log(`Tool: ${block.name} → ${result}`);
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: result,
        });
      }
    }

    if (toolResults.length > 0) {
      messages.push({ role: "user", content: toolResults });
    }
  }
}

const answer = await reactAgent(
  "I'm flying from NYC to London on July 10th. What's the weather there and what flights are available?",
);
console.log(answer);
```
:::

---

### Pattern 5: Self-Consistency

Run the same prompt N times at high temperature, return the majority answer. Best for classification and factual lookups where one wrong answer is costly.

::: code-group
```python [Python]
from langchain_openai import ChatOpenAI
from collections import Counter

def self_consistent(prompt: str, n: int = 5, temperature: float = 0.7) -> str:
    llm     = ChatOpenAI(model="gpt-4o", temperature=temperature)
    answers = [llm.invoke(prompt).content.strip() for _ in range(n)]
    majority, count = Counter(answers).most_common(1)[0]
    print(f"Votes: {Counter(answers)} | Winner: '{majority}' ({count}/{n})")
    return majority

result = self_consistent(
    "Classify: 'My API key stopped working after plan upgrade.' "
    "Reply with ONLY ONE word: billing, technical, account, or faq.",
    n=5
)
```
```javascript [Node.js]
import OpenAI from "openai";

const openai = new OpenAI();

async function selfConsistent(prompt, n = 5, temperature = 0.7) {
  const promises = Array.from({ length: n }, () =>
    openai.chat.completions.create({
      model: "gpt-4o",
      temperature,
      max_tokens: 20,
      messages: [{ role: "user", content: prompt }],
    }),
  );

  const responses = await Promise.all(promises);
  const answers = responses.map((r) => r.choices[0].message.content.trim());

  const counts = answers.reduce((acc, a) => {
    acc[a] = (acc[a] || 0) + 1;
    return acc;
  }, {});

  const majority = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  console.log("Votes:", counts, "| Winner:", majority);
  return majority;
}

const result = await selfConsistent(
  "Classify: 'My API key stopped working after plan upgrade.' " +
    "Reply with ONLY ONE word: billing, technical, account, or faq.",
);
```
:::

---

### Pattern 6: Structured Output (Pydantic / Zod)

Force the model to return typed, validated data instead of freeform text.

::: code-group
```python [Python]
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional

class ExtractedInvoice(BaseModel):
    invoice_number: str            = Field(description="Invoice identifier")
    vendor_name:    str            = Field(description="Name of the vendor")
    invoice_date:   str            = Field(description="Date in YYYY-MM-DD")
    total_amount:   float          = Field(description="Total amount due")
    currency:       str            = Field(default="USD")
    due_date:       Optional[str]  = Field(default=None)

llm        = ChatOpenAI(model="gpt-4o", temperature=0)
structured = llm.with_structured_output(ExtractedInvoice)

result = structured.invoke(
    "Extract the invoice data from this text:\n\n"
    "INVOICE #INV-2025-0042 | Acme Corp | Date: July 1, 2025 | "
    "Due: July 31, 2025 | Total Due: $1,250.00 USD"
)

print(result.invoice_number)  # INV-2025-0042
print(result.total_amount)    # 1250.0
```
```javascript [Node.js]
import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";

const client = new Anthropic();

const InvoiceSchema = z.object({
  invoice_number: z.string(),
  vendor_name: z.string(),
  invoice_date: z.string(),
  total_amount: z.number(),
  currency: z.string().default("USD"),
  due_date: z.string().nullable().optional(),
});

async function extractInvoice(text) {
  const response = await client.messages.create({
    model: "claude-sonnet-4-5",
    max_tokens: 512,
    messages: [
      {
        role: "user",
        content:
          `Extract invoice data from this text. ` +
          `Respond ONLY with a JSON object matching this schema exactly — ` +
          `no markdown, no explanation:\n` +
          `{"invoice_number": string, "vendor_name": string, "invoice_date": "YYYY-MM-DD", ` +
          `"total_amount": number, "currency": string, "due_date": "YYYY-MM-DD" | null}\n\n` +
          `Text: ${text}`,
      },
    ],
  });

  const raw = response.content[0].text.trim();
  const parsed = InvoiceSchema.parse(JSON.parse(raw));
  return parsed;
}

const invoice = await extractInvoice(
  "INVOICE #INV-2025-0042 | Acme Corp | Date: July 1, 2025 | Due: July 31, 2025 | Total Due: $1,250.00 USD",
);
console.log(invoice.invoice_number); // INV-2025-0042
console.log(invoice.total_amount); // 1250
```
:::

---

### Pattern 7: Constitutional AI (Principle Checklist)

Apply a list of principles to an output sequentially, revising at each violation.

::: code-group
```python [Python]
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

PRINCIPLES = [
    "The response must not exceed 120 words.",
    "The response must not make any promises about delivery dates unless explicitly confirmed.",
    "The response must address the customer by name if provided.",
    "The response must end with a clear next step or call to action.",
]

class Check(BaseModel):
    violates: bool
    revised:  str

llm    = ChatOpenAI(model="gpt-4o", temperature=0)
critic = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(Check)

def apply_constitution(text: str, customer_name: str = "") -> str:
    current = text
    for principle in PRINCIPLES:
        result = critic.invoke(
            f"Principle: {principle}\n"
            f"Context: customer name is '{customer_name}'\n"
            f"Response: {current}\n\n"
            f"Does this response violate the principle? "
            f"If yes, provide a revised version that complies. "
            f"If no, return the original unchanged."
        )
        if result.violates:
            print(f"  Violated: {principle[:60]}...")
            current = result.revised
    return current
```
```javascript [Node.js]
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const PRINCIPLES = [
  "The response must not exceed 120 words.",
  "The response must not make any promises about delivery dates unless explicitly confirmed.",
  "The response must address the customer by name if provided.",
  "The response must end with a clear next step or call to action.",
];

async function applyConstitution(text, customerName = "") {
  let current = text;

  for (const principle of PRINCIPLES) {
    const response = await client.messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 512,
      messages: [
        {
          role: "user",
          content:
            `Principle: ${principle}\n` +
            `Context: customer name is '${customerName}'\n` +
            `Response: ${current}\n\n` +
            `Does this response violate the principle? ` +
            `Respond ONLY with JSON: {"violates": true/false, "revised": "...response..."}\n` +
            `If it does not violate, set revised to the original text unchanged.`,
        },
      ],
    });

    const raw = response.content[0].text.trim();
    const result = JSON.parse(raw.replace(/```json|```/g, "").trim());

    if (result.violates) {
      console.log(`  Violated: ${principle.slice(0, 60)}...`);
      current = result.revised;
    }
  }

  return current;
}
```
:::

---

### Pattern 8: Reflexion (Failure Memory Loop)

The agent builds a growing list of lessons from past failures, carried into every subsequent attempt.

::: code-group
```python [Python]
from langchain_openai import ChatOpenAI

llm    = ChatOpenAI(model="gpt-4o", temperature=0.5)
critic = ChatOpenAI(model="gpt-4o", temperature=0)

def reflexion(task: str, max_attempts: int = 4) -> str:
    reflections = []

    for attempt in range(max_attempts):
        lesson_block = ""
        if reflections:
            lessons      = "\n".join(f"- {r}" for r in reflections)
            lesson_block = f"\n\nLessons from previous attempts (do not repeat these):\n{lessons}"

        output = llm.invoke(f"{task}{lesson_block}").content

        evaluation = critic.invoke(
            f"Task: {task}\nAttempt: {output}\n\n"
            f"Does this fully and correctly complete the task? "
            f"Reply with PASS or FAIL followed by your reasoning."
        ).content

        print(f"Attempt {attempt + 1}: {'✓' if evaluation.startswith('PASS') else '✗'}")

        if evaluation.strip().upper().startswith("PASS"):
            return output

        lesson = critic.invoke(
            f"Task: {task}\nFailed attempt: {output}\nFeedback: {evaluation}\n\n"
            f"Write one lesson under 20 words that would prevent this failure next time."
        ).content.strip()
        reflections.append(lesson)

    return output  # best effort

result = reflexion("Write a Python function that validates an ISO 8601 date string.")
```
```javascript [Node.js]
import OpenAI from "openai";

const openai = new OpenAI();

async function reflexion(task, maxAttempts = 4) {
  const reflections = [];

  async function invoke(prompt) {
    const r = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: prompt }],
    });
    return r.choices[0].message.content;
  }

  let lastOutput = "";

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const lessonBlock =
      reflections.length > 0
        ? `\n\nLessons from previous attempts (do not repeat these):\n` +
          reflections.map((r) => `- ${r}`).join("\n")
        : "";

    lastOutput = await invoke(`${task}${lessonBlock}`);

    const evaluation = await invoke(
      `Task: ${task}\nAttempt: ${lastOutput}\n\n` +
        `Does this fully and correctly complete the task? ` +
        `Reply with PASS or FAIL followed by your reasoning.`,
    );

    console.log(
      `Attempt ${attempt + 1}: ${evaluation.startsWith("PASS") ? "✓" : "✗"}`,
    );

    if (evaluation.trim().toUpperCase().startsWith("PASS")) return lastOutput;

    const lesson = await invoke(
      `Task: ${task}\nFailed attempt: ${lastOutput}\nFeedback: ${evaluation}\n\n` +
        `Write one lesson under 20 words that would prevent this failure next time.`,
    );
    reflections.push(lesson.trim());
  }

  return lastOutput;
}
```
:::

---

### Quick Reference: Pattern Selection Guide

| Situation                         | Pattern           | Why                                            |
| --------------------------------- | ----------------- | ---------------------------------------------- |
| Simple instruction following      | Zero-Shot         | Fast, cheap, usually sufficient                |
| Math, logic, multi-step decisions | Chain of Thought  | Forces reasoning before conclusion             |
| Consistent output format needed   | Few-Shot          | Examples teach format better than instructions |
| Agent needs tools                 | ReAct             | Industry standard for tool-using agents        |
| High-stakes classification        | Self-Consistency  | Majority vote reduces single-sample error      |
| Output fed to another system      | Structured Output | Enforces schema at the type level              |
| Compliance / policy requirements  | Constitutional AI | Auditable, principle-by-principle enforcement  |
| Quality ceiling not improving     | Reflexion         | Builds explicit failure memory across attempts |

---

## Cheat Sheet 2: Tool Repository

50+ APIs organized by category, with agent use cases for each.

---

### Communication

| API                       | What It Does                 | Agent Use Case                                        | Docs                                  |
| ------------------------- | ---------------------------- | ----------------------------------------------------- | ------------------------------------- |
| **SendGrid**              | Transactional email          | Send follow-up emails, delivery confirmations         | sendgrid.com/docs                     |
| **Mailgun**               | Email send + tracking        | Outreach agents, open/click tracking                  | mailgun.com/docs                      |
| **Twilio SMS**            | Programmatic SMS             | Appointment reminders, OTP, status alerts             | twilio.com/docs/sms                   |
| **Twilio Voice**          | Programmable phone calls     | Voice agents (combine with Vapi/Realtime API)         | twilio.com/docs/voice                 |
| **Slack API**             | Post messages, read channels | Escalation alerts, team notifications, HITL approvals | api.slack.com                         |
| **Discord API**           | Bot messages and commands    | Community agents, moderation bots                     | discord.com/developers                |
| **Telegram Bot API**      | Messaging bot                | Customer-facing chat agents, notifications            | core.telegram.org/bots                |
| **WhatsApp Business API** | WhatsApp messaging           | Customer support in markets where WA dominates        | developers.facebook.com/docs/whatsapp |

::: code-group
```python [Python]
from slack_sdk import WebClient

slack = WebClient(token=SLACK_BOT_TOKEN)

def send_escalation_alert(ticket_id: str, reason: str, channel: str = "#support-escalations"):
    slack.chat_postMessage(
        channel=channel,
        text=f":warning: *Ticket {ticket_id} escalated*\nReason: {reason}",
        blocks=[
            {"type": "section", "text": {"type": "mrkdwn", "text": f":warning: *Ticket `{ticket_id}` escalated to human review*"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Reason:* {reason}"}},
            {"type": "actions", "elements": [
                {"type": "button", "text": {"type": "plain_text", "text": "View Ticket"}, "url": f"https://your-app.com/tickets/{ticket_id}"}
            ]}
        ]
    )
```
```javascript [Node.js]
import { WebClient } from "@slack/web-api";

const slack = new WebClient(process.env.SLACK_BOT_TOKEN);

async function sendEscalationAlert(
  ticketId,
  reason,
  channel = "#support-escalations",
) {
  await slack.chat.postMessage({
    channel,
    text: `:warning: Ticket ${ticketId} escalated`,
    blocks: [
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text: `:warning: *Ticket \`${ticketId}\` escalated*`,
        },
      },
      {
        type: "section",
        text: { type: "mrkdwn", text: `*Reason:* ${reason}` },
      },
    ],
  });
}
```
:::

---

### Search & Web Intelligence

| API                  | What It Does                   | Agent Use Case                                  | Docs                 |
| -------------------- | ------------------------------ | ----------------------------------------------- | -------------------- |
| **Tavily**           | LLM-optimized web search       | Research agents, news monitoring, fact checking | tavily.com           |
| **Serper**           | Google Search API              | Lead research, competitive intelligence         | serper.dev           |
| **Exa**              | Semantic web search            | Find similar companies, concept-based lookup    | exa.ai               |
| **Brave Search API** | Privacy-first search           | Alternative to Google for general research      | api.search.brave.com |
| **SerpApi**          | Scrape Google / Bing / YouTube | SERP rank monitoring, research pipelines        | serpapi.com          |
| **Jina Reader**      | URL → clean markdown           | Web scraping for RAG, article extraction        | r.jina.ai            |
| **Firecrawl**        | Crawl entire sites to markdown | Knowledge base ingestion, site audits           | firecrawl.dev        |
| **Diffbot**          | Entity and article extraction  | Structured data from news, company pages        | diffbot.com          |

::: code-group
```python [Python]
from tavily import TavilyClient
from langchain_core.tools import tool

tavily = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def search_web(query: str) -> str:
    """Search the web for current information. Returns top 3 results with snippets."""
    results = tavily.search(query=query, max_results=3)
    return "\n\n".join(
        f"[{r['title']}]\n{r['content']}\nURL: {r['url']}"
        for r in results["results"]
    )
```
```javascript [Node.js]
import { tavily } from "@tavily/core";

const client = tavily({ apiKey: process.env.TAVILY_API_KEY });

async function searchWeb(query) {
  const response = await client.search(query, { maxResults: 3 });
  return response.results
    .map((r) => `[${r.title}]\n${r.content}\nURL: ${r.url}`)
    .join("\n\n");
}
```
:::

---

### Productivity & Knowledge

| API                         | What It Does                          | Agent Use Case                                    | Docs                                              |
| --------------------------- | ------------------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| **Notion API**              | Read/write Notion databases and pages | KB management, task creation, meeting notes       | developers.notion.com                             |
| **Confluence API**          | Atlassian wiki read/write             | Internal knowledge base for support agents        | developer.atlassian.com                           |
| **Google Drive API**        | File read/write, search               | Document agents, report generation                | developers.google.com/drive                       |
| **Google Docs API**         | Create and edit documents             | Report writing, proposal drafting                 | developers.google.com/docs                        |
| **Google Sheets API**       | Spreadsheet read/write                | Data extraction output, KPI dashboards            | developers.google.com/sheets                      |
| **Airtable API**            | Database read/write                   | CRM-lite operations, structured data storage      | airtable.com/developers                           |
| **Dropbox API**             | File operations                       | Document processing pipelines                     | dropbox.com/developers                            |
| **Obsidian Local REST API** | Local vault read/write                | Local knowledge base for privacy-sensitive agents | github.com/coddingtonbear/obsidian-local-rest-api |

---

### Calendar & Scheduling

| API                            | What It Does                     | Agent Use Case                                             | Docs                           |
| ------------------------------ | -------------------------------- | ---------------------------------------------------------- | ------------------------------ |
| **Google Calendar API**        | Event CRUD, availability check   | Appointment scheduling, meeting booking                    | developers.google.com/calendar |
| **Microsoft Graph (Calendar)** | Outlook calendar operations      | Enterprise scheduling agents                               | learn.microsoft.com/graph      |
| **Calendly API**               | Read availability, create events | Lead scheduling automation                                 | developer.calendly.com         |
| **Cal.com API**                | Open-source scheduling           | Self-hosted booking agents                                 | cal.com/docs/api               |
| **Nylas**                      | Unified calendar + email API     | Multi-provider scheduling without per-provider integration | developer.nylas.com            |

::: code-group
```python [Python]
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from datetime import datetime, timedelta, timezone

def get_free_slots(date: str, duration_minutes: int = 30) -> list[str]:
    """Return available time slots on a given date."""
    creds   = Credentials.from_authorized_user_file("token.json")
    service = build("calendar", "v3", credentials=creds)

    day_start = f"{date}T09:00:00Z"
    day_end   = f"{date}T17:00:00Z"

    events = service.events().list(
        calendarId="primary", timeMin=day_start, timeMax=day_end,
        singleEvents=True, orderBy="startTime"
    ).execute().get("items", [])

    busy_times = [
        (e["start"].get("dateTime"), e["end"].get("dateTime"))
        for e in events
    ]

    slots, current = [], datetime.fromisoformat(day_start.replace("Z", "+00:00"))
    end_of_day     = datetime.fromisoformat(day_end.replace("Z", "+00:00"))

    while current + timedelta(minutes=duration_minutes) <= end_of_day:
        slot_end = current + timedelta(minutes=duration_minutes)
        is_free  = all(
            slot_end <= datetime.fromisoformat(b[0].replace("Z", "+00:00")) or
            current  >= datetime.fromisoformat(b[1].replace("Z", "+00:00"))
            for b in busy_times
        )
        if is_free:
            slots.append(current.strftime("%H:%M"))
        current += timedelta(minutes=30)

    return slots
```
```javascript [Node.js]
// npm install googleapis
import { google } from "googleapis";

async function getFreeSlots(date, durationMinutes = 30) {
  const auth    = new google.auth.GoogleAuth({ scopes: ["https://www.googleapis.com/auth/calendar.readonly"] });
  const calendar = google.calendar({ version: "v3", auth });

  const dayStart = `${date}T09:00:00Z`;
  const dayEnd   = `${date}T17:00:00Z`;

  const { data } = await calendar.events.list({
    calendarId: "primary",
    timeMin: dayStart,
    timeMax: dayEnd,
    singleEvents: true,
    orderBy: "startTime",
  });

  const busyTimes = (data.items || []).map((e) => ({
    start: new Date(e.start.dateTime),
    end:   new Date(e.end.dateTime),
  }));

  const slots     = [];
  let   current   = new Date(dayStart);
  const endOfDay  = new Date(dayEnd);

  while (new Date(current.getTime() + durationMinutes * 60000) <= endOfDay) {
    const slotEnd = new Date(current.getTime() + durationMinutes * 60000);
    const isFree  = busyTimes.every((b) => slotEnd <= b.start || current >= b.end);
    if (isFree) slots.push(current.toISOString().slice(11, 16));
    current = new Date(current.getTime() + 30 * 60000);
  }

  return slots;
}
```
:::

---

### CRM & Sales

| API                | What It Does                  | Agent Use Case                                        | Docs                               |
| ------------------ | ----------------------------- | ----------------------------------------------------- | ---------------------------------- |
| **HubSpot API**    | Contacts, deals, companies    | Lead enrichment, deal stage updates, activity logging | developers.hubspot.com             |
| **Salesforce API** | Full CRM operations           | Enterprise CRM automation, opportunity management     | developer.salesforce.com           |
| **Pipedrive API**  | Deals and contacts            | SMB sales pipeline automation                         | developers.pipedrive.com           |
| **Apollo.io API**  | Prospecting, enrichment       | Lead generation, contact discovery                    | apolloio.github.io/apollo-api-docs |
| **Hunter.io**      | Email finder and verifier     | Outreach preparation, email validation                | hunter.io/api-documentation        |
| **Clearbit**       | Company and person enrichment | ICP scoring, lead qualification                       | dashboard.clearbit.com/docs        |
| **ZoomInfo API**   | B2B contact data              | Enterprise prospecting                                | zoominfo.com/developer             |

---

### Finance & Payments

| API                | What It Does                      | Agent Use Case                                             | Docs                            |
| ------------------ | --------------------------------- | ---------------------------------------------------------- | ------------------------------- |
| **Stripe API**     | Payments, subscriptions, invoices | Billing agents, refund automation, subscription management | stripe.com/docs/api             |
| **Plaid**          | Bank account data                 | Financial analysis agents, expense categorization          | plaid.com/docs                  |
| **QuickBooks API** | Accounting read/write             | Invoice creation, expense categorization, P&L agents       | developer.intuit.com            |
| **Xero API**       | Accounting operations             | AP/AR automation, reconciliation agents                    | developer.xero.com              |
| **Alpha Vantage**  | Stock and forex data              | Market monitoring agents, portfolio tracking               | alphavantage.co/documentation   |
| **CoinGecko API**  | Crypto market data                | Crypto portfolio agents, price alerts                      | coingecko.com/api/documentation |

::: code-group
```python [Python]
import stripe
from langchain_core.tools import tool

stripe.api_key = STRIPE_SECRET_KEY

@tool
def process_refund(payment_intent_id: str, amount_cents: int, reason: str) -> str:
    """
    Process a refund for a Stripe payment.
    amount_cents: amount to refund in cents (e.g. 5000 = $50.00)
    reason: duplicate, fraudulent, or requested_by_customer
    Only call for amounts under 10000 cents ($100). Escalate larger refunds.
    """
    if amount_cents > 10000:
        return "ESCALATE: Refund exceeds $100 automated limit."
    try:
        refund = stripe.Refund.create(
            payment_intent=payment_intent_id,
            amount=amount_cents,
            reason=reason
        )
        return f"Refund {refund.id} processed: ${amount_cents / 100:.2f}"
    except stripe.error.StripeError as e:
        return f"Refund failed: {e.user_message}"
```
```javascript [Node.js]
import Stripe from "stripe";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);

async function processRefund(paymentIntentId, amountCents, reason) {
  if (amountCents > 10000) {
    return "ESCALATE: Refund exceeds $100 automated limit.";
  }
  try {
    const refund = await stripe.refunds.create({
      payment_intent: paymentIntentId,
      amount: amountCents,
      reason,
    });
    return `Refund ${refund.id} processed: $${(amountCents / 100).toFixed(2)}`;
  } catch (err) {
    return `Refund failed: ${err.message}`;
  }
}
```
:::

---

### Developer & Infrastructure

| API                 | What It Does               | Agent Use Case                                      | Docs                                          |
| ------------------- | -------------------------- | --------------------------------------------------- | --------------------------------------------- |
| **GitHub API**      | Repos, issues, PRs, code   | DevOps agents, code review automation, issue triage | docs.github.com/rest                          |
| **Linear API**      | Issue tracker              | Engineering workflow agents, sprint management      | linear.app/docs/graphql                       |
| **Jira API**        | Project management         | Enterprise ticket management, sprint reporting      | developer.atlassian.com/cloud/jira            |
| **Vercel API**      | Deploy and manage projects | CI/CD agents, deployment monitoring                 | vercel.com/docs/rest-api                      |
| **AWS SDK**         | Full AWS services          | Cloud infrastructure agents                         | boto3.amazonaws.com (Python) / aws-sdk (Node) |
| **Supabase API**    | Postgres + auth + storage  | Agent state persistence, user management            | supabase.com/docs/reference                   |
| **PlanetScale API** | Serverless MySQL           | Agent database operations                           | planetscale.com/docs/reference                |
| **Redis**           | In-memory data store       | Agent session cache, rate limiting, pub/sub         | redis.io/docs                                 |

::: code-group
```python [Python]
from github import Github
from langchain_core.tools import tool

gh   = Github(GITHUB_TOKEN)
repo = gh.get_repo("your-org/your-repo")

@tool
def get_open_issues(label: str = "", limit: int = 10) -> str:
    """Get open GitHub issues, optionally filtered by label."""
    issues = repo.get_issues(state="open", labels=[label] if label else [])
    result = []
    for issue in list(issues)[:limit]:
        result.append(f"#{issue.number}: {issue.title} (opened {issue.created_at.date()})")
    return "\n".join(result) if result else "No open issues found."

@tool
def add_issue_label(issue_number: int, label: str) -> str:
    """Add a label to a GitHub issue."""
    issue = repo.get_issue(issue_number)
    issue.add_to_labels(label)
    return f"Label '{label}' added to issue #{issue_number}"
```
```javascript [Node.js]
// npm install @octokit/rest
import { Octokit } from "@octokit/rest";

const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });
const OWNER = "your-org";
const REPO  = "your-repo";

async function getOpenIssues(label = "", limit = 10) {
  const params = { owner: OWNER, repo: REPO, state: "open", per_page: limit };
  if (label) params.labels = label;

  const { data } = await octokit.issues.listForRepo(params);
  if (data.length === 0) return "No open issues found.";
  return data
    .map((i) => `#${i.number}: ${i.title} (opened ${i.created_at.slice(0, 10)})`)
    .join("\n");
}

async function addIssueLabel(issueNumber, label) {
  await octokit.issues.addLabels({
    owner: OWNER,
    repo:  REPO,
    issue_number: issueNumber,
    labels: [label],
  });
  return `Label '${label}' added to issue #${issueNumber}`;
}
```
:::

---

### Data & Documents

| API                      | What It Does                 | Agent Use Case                                   | Docs                                   |
| ------------------------ | ---------------------------- | ------------------------------------------------ | -------------------------------------- |
| **AWS Textract**         | OCR + form/table extraction  | Invoice processing, form digitization            | aws.amazon.com/textract                |
| **Google Document AI**   | Intelligent document parsing | Structured extraction from varied document types | cloud.google.com/document-ai           |
| **Reducto**              | PDF and document extraction  | Clean markdown from complex PDFs for RAG         | reducto.ai                             |
| **Unstructured.io**      | Partition any document type  | Pre-processing pipeline for RAG knowledge bases  | unstructured-io.github.io/unstructured |
| **Pandas AI**            | Conversational data analysis | Natural language queries on CSV/Excel data       | pandas-ai.com                          |
| **E2B Code Interpreter** | Sandboxed code execution     | Data analysis agents, chart generation           | e2b.dev/docs                           |
| **Browserbase**          | Headless browser automation  | Web scraping, form filling, screenshot agents    | browserbase.com/docs                   |

---

### AI & ML Utilities

| API                   | What It Does              | Agent Use Case                                         | Docs                                       |
| --------------------- | ------------------------- | ------------------------------------------------------ | ------------------------------------------ |
| **OpenAI Embeddings** | Text → vector             | RAG pipelines, semantic search, clustering             | platform.openai.com/docs/guides/embeddings |
| **Cohere Rerank**     | Re-rank retrieved chunks  | Improve RAG precision post-retrieval                   | cohere.com/rerank                          |
| **Pinecone**          | Managed vector database   | Production RAG, multi-tenant knowledge bases           | docs.pinecone.io                           |
| **Weaviate**          | Open-source vector DB     | Self-hosted RAG with hybrid search                     | weaviate.io/developers                     |
| **AssemblyAI**        | Speech-to-text + analysis | Voice agent transcription, meeting summarization       | assemblyai.com/docs                        |
| **ElevenLabs**        | Text-to-speech            | Voice agent response synthesis                         | elevenlabs.io/docs                         |
| **Replicate**         | Run ML models via API     | Image generation, audio processing, specialized models | replicate.com/docs                         |
| **Together AI**       | Open-source LLM inference | Cost-effective alternative to OpenAI for many tasks    | docs.together.ai                           |

---

## Cheat Sheet 3: Architecture Decision Guide

Every major choice in this book, distilled to the key question and the right answer for each scenario.

---

### Decision 1: Single Agent vs. Multi-Agent

| Use Single Agent                  | Use Multi-Agent                             |
| --------------------------------- | ------------------------------------------- |
| Task fits in one context window   | Task naturally splits into specialist roles |
| Linear steps, no parallelism      | Steps can run in parallel                   |
| Same expertise needed end to end  | Quality improves with a separate critic     |
| Tight deadline, simpler debugging | Different tools needed per stage            |

**Rule of thumb**: start with a single agent. Split when you hit a quality ceiling or a context window limit — not before.

---

### Decision 2: RAG vs. Fine-Tuning vs. Prompt Engineering

```
Need the model to know specific facts or documents?
  └── Yes → Use RAG (retrieval, not training)
  └── No, but outputs are consistently wrong in the same way?
        └── Yes, and you have 100+ labeled examples → Fine-tune
        └── Yes, but no labeled data → Fix the prompt first
        └── No → You have a prompt problem. Improve the prompt.
```

| Approach           | Best for                                   | Cost                      | Updateable?         |
| ------------------ | ------------------------------------------ | ------------------------- | ------------------- |
| Prompt Engineering | Behavior shaping, format, tone             | Free                      | Yes, instantly      |
| RAG                | Domain facts, documents, live data         | Low (inference + storage) | Yes, re-index       |
| Fine-tuning        | Consistent style/format, domain vocabulary | Medium (training job)     | Requires retraining |

---

### Decision 3: Memory Strategy

| Memory Type                 | Implementation            | Use When                              |
| --------------------------- | ------------------------- | ------------------------------------- |
| **Short-term (in-context)** | Pass full message history | Conversation < 50 turns               |
| **Sliding window**          | Keep last N messages      | Long conversations, fixed cost        |
| **Summarization**           | LLM compresses old turns  | Very long conversations               |
| **RAG (long-term)**         | Vector store retrieval    | Cross-session memory, large knowledge |
| **External state (DB)**     | Redis, Postgres, SQLite   | Multi-user, persistent agent state    |

---

### Decision 4: Which Model to Use

```
Is speed and cost the primary constraint?
  └── Yes → gpt-4o-mini / claude-haiku / local Ollama model
Is complex reasoning or code generation required?
  └── Yes → gpt-4o / claude-sonnet or claude-opus
Is privacy / on-premise required?
  └── Yes → Ollama with llama3 / mistral / phi3
Is the task well-defined and repetitive with labeled data?
  └── Yes → Fine-tune gpt-4o-mini for cost + quality
```

---

### Decision 5: Streaming vs. Request-Response

| Use Streaming                  | Use Request-Response                       |
| ------------------------------ | ------------------------------------------ |
| User-facing chat interface     | Batch processing, no user watching         |
| Response takes > 3 seconds     | Response is fast (< 2s)                    |
| Output is read as it arrives   | Output processed as a whole (JSON parsing) |
| Token-by-token display matters | Atomic response needed                     |

---

### Decision 6: Deployment Platform

| Platform           | Choose When                                          |
| ------------------ | ---------------------------------------------------- |
| **Railway**        | Solo dev, fast deploy, MVP, generous free tier       |
| **Render**         | Team deployment, Heroku-style, `render.yaml` IaC     |
| **AWS Lambda**     | Spiky traffic, short tasks (< 15 min), AWS ecosystem |
| **Docker + VPS**   | Full control, cost optimization at scale             |
| **Ollama (local)** | Privacy-sensitive, offline, zero per-token cost      |

---

### Decision 7: When to Add Human-in-the-Loop

Always add a HITL gate before:

| Action                                                | Why             |
| ----------------------------------------------------- | --------------- |
| Sending any outbound communication (email, SMS, post) | Cannot unsend   |
| Deleting or modifying records                         | Data loss       |
| Any financial transaction (charge, refund, transfer)  | Monetary impact |
| Deploying code to production                          | Outage risk     |
| Any action on behalf of another user                  | Liability       |

**Never** automate these without HITL in the first 30 days, regardless of test accuracy.

---

### Decision 8: Evaluation Strategy

| What to measure                       | Eval type            | When to run                |
| ------------------------------------- | -------------------- | -------------------------- |
| Response length, format, structure    | Deterministic        | Every commit               |
| Output quality, accuracy, helpfulness | LLM-as-judge         | Every release              |
| Real user satisfaction                | CSAT / thumbs up     | Continuously in production |
| Regression on known failure cases     | Regression suite     | Every deployment           |
| A/B prompt variants                   | Experiment framework | When changing prompts      |

**Minimum viable eval suite**: 20 deterministic checks + 50 model-judged cases. Anything less and your pass rate is noise.

---

### The One-Page Agent Architecture Checklist

Before you ship any agent to a real user, confirm all of these:

**Input layer**

- [ ] Input length capped
- [ ] Known injection patterns checked
- [ ] External content treated as data (delimiters in prompt)

**Agent layer**

- [ ] `max_iterations` set
- [ ] All LLM calls wrapped with retry and exponential backoff
- [ ] Fallback model configured

**Tool layer**

- [ ] Tools scoped to minimum necessary permissions
- [ ] Tool inputs validated with Pydantic / Zod
- [ ] External HTTP calls restricted to allowlist

**Output layer**

- [ ] Output guardrail checks for policy violations
- [ ] PII not leaked from retrieved context
- [ ] Irreversible actions have HITL gate

**Production layer**

- [ ] `/health` endpoint returns 200
- [ ] LangSmith (or equivalent) tracing active
- [ ] Budget hard limit set on all LLM providers
- [ ] Rate limiting per user enforced
- [ ] Deterministic eval suite passes before every deploy
