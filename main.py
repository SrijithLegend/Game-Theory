import os
import json
import getpass
import functools
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.tools import tool
from deepagents import create_deep_agent

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key: ")

gemini_model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
gemini_model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
gemini_model3 = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

file_path = "C:/Users/sriji/Downloads/jk.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

vector_store = Chroma(
    collection_name="strategic_intel",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
vector_store.add_documents(documents=all_splits)

COMPETITOR_SYSTEM_PROMPT = """You are a COMPETITOR INTELLIGENCE AGENT. Analyze threats. Return JSON."""
CUSTOMER_SYSTEM_PROMPT = """You are a CUSTOMER BEHAVIOR AGENT. Analyze churn. Return JSON."""
MARKET_SYSTEM_PROMPT = """You are a MARKET AND MACRO RISK AGENT. Analyze macro risks. Return JSON."""

@tool
def vector_search(query: str) -> str:
    """Search internal documents and financial reports."""
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])

def run_specialist_logic(decision_context: str, prompt: str, llm: ChatGoogleGenerativeAI, peer_feedback: str = "") -> dict:
    search_results = vector_search.invoke(decision_context)
    
    # We add a section for peer feedback so the agent can react to others
    full_prompt = (
        f"{prompt}\n\n"
        f"DECISION CONTEXT: {decision_context}\n\n"
        f"PEER ANALYST FEEDBACK (REPLY TO THIS): {peer_feedback}\n\n"
        f"INTERNAL DATA:\n{search_results}"
    )
    
    response = llm.invoke(full_prompt)
    content = response.content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)

@tool
def competitor_analysis(decision_context: str) -> dict:
    """Run competitive threat analysis on the company."""
    return run_specialist_logic(decision_context, COMPETITOR_SYSTEM_PROMPT, gemini_model3)

@tool
def customer_analysis(decision_context: str) -> dict:
    """Run customer behavior and churn analysis on the company."""
    return run_specialist_logic(decision_context, CUSTOMER_SYSTEM_PROMPT, gemini_model2)

@tool
def market_analysis(decision_context: str) -> dict:
    """Run market and macro risk analysis on the company."""
    return run_specialist_logic(decision_context, MARKET_SYSTEM_PROMPT, gemini_model1)


SUPERVISOR_SYSTEM_PROMPT = """
You are the STRATEGIC MODERATOR. You do not just collect reports; you facilitate a DEBATE.

### YOUR COLLABORATION WORKFLOW:
1. **Initial Investigation:** Call all 3 specialist tools to get the baseline data.
2. **The Debate Phase:** Take the 'threats' from the Competitor Agent and feed them back into the Customer Agent's 'decision_context' to see if they agree. 
3. **Synthesis:** Explicitly show the conversation. (e.g., "The Competitor Agent raised a risk of price wars, which the Market Agent confirmed is likely due to new regulations.")
4. **Final Verdict:** GO/NO-GO based on the resolved debate.

You must show your 'Thinking' process so the user can see how you are making the agents talk.
"""

# Re-initialize the agent with the new prompt
supervisor_agent = create_deep_agent(
    gemini_model1,
    tools=[competitor_analysis, customer_analysis, market_analysis],
    system_prompt=SUPERVISOR_SYSTEM_PROMPT
)

if __name__ == "__main__":
    strategic_request = (
        "Perform a Red Team analysis. I want to see the agents challenge each other's "
        "assumptions regarding the company's sustainability."
    )

    inputs = {"messages": [{"role": "user", "content": strategic_request}]}
    
    for s in supervisor_agent.stream(inputs, stream_mode="values"):
        message = s["messages"][-1]
        
        # Identify if this is a tool result (The Specialist talking)
        if message.role == "tool":
            print(f"\n[RECEIVED REPORT FROM {message.name.upper()}]")
        
        # Identify if the AI is providing a synthesis (The Supervisor talking)
        if message.role == "assistant" and not message.tool_calls:
            print("\n[SUPERVISOR SYNTHESIS & DEBATE]")

        message.pretty_print()