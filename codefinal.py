from langchain.tools import tool
from langchain.chat_models import init_chat_model
import redis
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-4", temperature=0,openai_api_key= os.getenv("OPENAI_API_KEY"))
from sentence_transformers import SentenceTransformer

# This MUST be named 'embedder' to match your function
embedder = SentenceTransformer('all-MiniLM-L6-v2') 

# Also ensure EMBEDDING_DIM matches the model (384 for this model)
EMBEDDING_DIM = 384

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator




class AgentState(TypedDict):
    """State that flows through our agent graph."""
    messages: Annotated[list[AnyMessage],operator.add]
    user_message: str
    user_id: str
    session_id: str
    working_memory: dict
    episodic_context: list
    semantic_context: list
    procedural_context: list
    next_memory_step: str
    response: str


from langchain.messages import SystemMessage
from langchain.messages import ToolMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END


# UPDATE THIS FUNCTION
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "end" # LangGraph uses this internally for the final state

REDIS_URL = "redis://localhost:6379"
QDRANT_URL = "http://localhost:6333"
MONGODB_URL = "mongodb://localhost:27017"

# Initialize clients
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
# Rename this to something very unique to avoid any shadowing
APP_VECTOR_STORE = QdrantClient(url=QDRANT_URL)
mongo_client = MongoClient(MONGODB_URL)
mongo_db = mongo_client["agent_memory"]

import json
from datetime import datetime, timedelta
from uuid import uuid4
import json
class MemoryExtraction(BaseModel):
    key: str = Field(description="The category of the information (e.g., 'diet', 'hobby', 'location')")
    value: str = Field(description="The specific fact to remember")

def extract_to_working_memory(state: AgentState):
    user_msg = state["messages"][-1].content
    session_id = state["session_id"]
    
    # Standardize to function_calling to stop the warnings
    extractor = llm.with_structured_output(MemoryExtraction, method="function_calling")
    prompt = f"Extract a personal fact (key and value) from: {user_msg}. If the message is a question or contains no factual info, return nothing."
    
    try:
        result = extractor.invoke(prompt)
        # Check that we got a valid result and it's not 'unknown'
        if result and result.value.lower() not in ["unknown", "none", "n/a"]:
            redis_key = f"wm:{session_id}:{result.key.lower()}"
            redis_client.set(redis_key, result.value, ex=1800) 
            print(f"--- NODE: EXTRACTED -> {result.key}: {result.value} ---")
    except:
        pass
    
    return state
def working_memory_node(state: AgentState):
    """Fetches immediate context from Redis."""
    session_id = state["session_id"]
    # Logic to fetch keys related to this session
    keys = redis_client.keys(f"wm:{session_id}:*")
    values = {k.split(":")[-1]: redis_client.get(k) for k in keys}
    
    print("--- NODE: WORKING MEMORY RETRIEVED ---")
    return {"working_memory": values}

def episodic_memory_node(state: AgentState):
    """
    Search episodic memory for past experiences using query_points.
    """
    user_id = state.get("user_id")
    if not state["messages"]:
        return {"episodic_context": []}

    # Extract the user's latest query to search against past events
    query_text = state["messages"][-1].content
    query_embedding = embedder.encode(query_text).tolist()
    collection_name = "episodic_memory"

    # Define Filters (matching the structure of your semantic search)
    conditions = []
    if user_id:
        conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
    
    search_filter = Filter(must=conditions) if conditions else None

    try:
        # Check if collection exists
        collections = [c.name for c in APP_VECTOR_STORE.get_collections().collections]
        if collection_name not in collections:
            print(f"--- Creating collection: {collection_name} ---")
            APP_VECTOR_STORE.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            return {"episodic_context": []}

        # Perform the search using query_points
        response = APP_VECTOR_STORE.query_points(
            collection_name=collection_name,
            query=query_embedding,
            query_filter=search_filter,
            limit=3  # Typically retrieve more for episodic context
        )

        # Extract narratives from the response points
        experiences = []
        for point in response.points:
            payload = point.payload
            # We usually look for 'content' or 'event' in episodic storage
            content = payload.get("content") or payload.get("event") or "No record detail"
            timestamp = payload.get("timestamp", "Unknown date")
            experiences.append(f"[{timestamp}] {content}")

        print(f"--- NODE: EPISODIC MEMORY ({len(experiences)} hits) ---")
        return {"episodic_context": experiences}

    except Exception as e:
        print(f"CRITICAL DEBUG in Episodic Node: {e}")
        return {"episodic_context": []}

def semantic_memory_node(state: AgentState):
    """
    Search semantic memory using modern query_points method.
    """
    user_id = state.get("user_id")
    if not state["messages"]:
        return {"semantic_context": []}

    query = state["messages"][-1].content
    query_embedding = embedder.encode(query).tolist()
    collection_name = "semantic_memory"

    # Build filters dynamically as per your requirement
    conditions = []
    if user_id:
        conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
    
    # Example: you could add knowledge_type here if it was in your AgentState
    search_filter = Filter(must=conditions) if conditions else None

    try:
        # Check if collection exists
        collections = [c.name for c in APP_VECTOR_STORE.get_collections().collections]
        if collection_name not in collections:
            print(f"--- Creating collection: {collection_name} ---")
            APP_VECTOR_STORE.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            return {"semantic_context": []}

        # Use query_points (Modern Qdrant API)
        response = APP_VECTOR_STORE.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=5
        )

        # In query_points, results are in 'response.points'
        facts = []
        for point in response.points:
            payload = point.payload
            # Adapt this to your specific metadata structure
            concept = payload.get("concept", "Preference")
            content = payload.get("definition") or payload.get("content") or "No detail"
            facts.append(f"{concept}: {content}")

        print(f"--- NODE: SEMANTIC MEMORY ({len(facts)} hits) ---")
        return {"semantic_context": facts}

    except Exception as e:
        print(f"CRITICAL DEBUG in Semantic Node: {e}")
        return {"semantic_context": []}
    

def procedural_memory_node(state: AgentState):
    """Fetches how-to instructions or rules from MongoDB."""
    query_text = state["messages"][-1].content
    collection = mongo_db["procedures"]
    
    # Simple regex search for procedural keywords in the user input
    # In a production app, you might use MongoDB Atlas Vector Search
    keywords = query_text.split()
    search_query = {"title": {"$regex": keywords[0], "$options": "i"}}
    
    procedure = collection.find_one(search_query)
    
    context = []
    if procedure:
        steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(procedure.get("steps", []))])
        context.append(f"PROCEDURE: {procedure.get('title')}\nSteps:\n{steps}")
    
    print(f"--- NODE: PROCEDURAL MEMORY ({len(context)} hits) ---")
    return {"procedural_context": context}

def final_response(state: AgentState):
    context_str = f"""
    WORKING MEMORY: {state.get('working_memory', {})}
    EPISODIC CONTEXT: {state.get('episodic_context', [])}
    SEMANTIC CONTEXT: {state.get('semantic_context', [])}
    PROCEDURAL CONTEXT: {state.get('procedural_context', [])}
    """
    
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant with access to the following memory context:\n"
            f"{context_str}\n"
            "Use this context to provide a personalized and accurate response."
        )
    )
    
    response = llm.invoke([system_message] + state["messages"])
    return {"messages": [response]}

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage

# Define the structure for the LLM's decision
class MemoryRoute(BaseModel):
    # Added 'procedural' to the Literal
    step: Literal["working", "episodic", "semantic", "arithmetic", "procedural"] = Field(
        description="The path to take based on user intent."
    )

def memory_manager(state: dict):
    # Use method="function_calling" to fix that GPT-4 warning
    router_model = llm.with_structured_output(MemoryRoute, method="function_calling")
    
    system_prompt = SystemMessage(
        content=(
            "You are a Router Manager. Decide the next step:\n"
            "- 'arithmetic': For math queries.\n"
            "- 'working': For casual conversation.\n"
            "- 'episodic': For past session events.\n"
            "- 'semantic': For facts/preferences.\n"
            "- 'procedural': If the user asks 'How do I...' or needs a step-by-step guide."
        )
    )
    decision = router_model.invoke([system_prompt] + state["messages"])
    print(f"--- ROUTING TO: {decision.step.upper()} ---")
    return {"next_memory_step": decision.step}

def arithmetic_agent(state: dict):
    """Specialized node for math"""
    # This uses the 'model_with_tools' you defined earlier
    msg = model_with_tools.invoke([
        SystemMessage(content="You are a math specialist. Use tools to calculate.")
    ] + state["messages"])
    return {"messages": [msg]}

from qdrant_client.models import PointStruct
import uuid

def consolidate_memory_node(state: AgentState):
    """
    Moves data from Working Memory (Redis) to Long-Term Memory (Qdrant).
    """
    user_id = state["user_id"]
    session_id = state["session_id"]
    
    # 1. Fetch everything currently in Redis for this session
    keys = redis_client.keys(f"wm:{session_id}:*")
    consolidated_count = 0
    
    for key in keys:
        value = redis_client.get(key)
        concept = key.split(":")[-1] # e.g., 'dietary_pref'
        
        # 2. Create an embedding for the fact
        fact_text = f"{concept}: {value}"
        vector = embedder.encode(fact_text).tolist()
        # In consolidate_memory_node, add a print to verify it's working:
        print(f"--- ATTEMPTING CONSOLIDATION FOR {concept}: {value} ---")
        
        # 3. Upsert into Semantic Memory
        APP_VECTOR_STORE.upsert(
            collection_name="semantic_memory",
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        "concept": concept,
                        "definition": value,
                        "consolidated_at": str(datetime.now())
                    }
                )
            ]
        )
        # 4. Optional: Clear from Redis after moving to long-term
        # redis_client.delete(key) 
        consolidated_count += 1

    print(f"--- NODE: CONSOLIDATED {consolidated_count} FACTS TO QDRANT ---")
    
    return {"working_memory": {}} # Clear working memory in state

from langgraph.graph import StateGraph, START, END
def seed_procedural_data():
    collection = mongo_db["procedures"]
    sample_procedure = {
        "title": "vegan",
        "steps": [
            "Check for honey, dairy, or eggs in ingredients.",
            "Verify if the sugar is processed with bone black.",
            "Confirm the cooking surfaces are cleaned of animal fats."
        ]
    }
    collection.update_one({"title": "vegan"}, {"$set": sample_procedure}, upsert=True)

seed_procedural_data()

# 1. The Conditional Logic (The Router)
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools" # This must match the string in workflow.add_node("tools", ...)
    return "end"

# 2. Define the Graph
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("manager", memory_manager)
workflow.add_node("working", working_memory_node)
workflow.add_node("episodic", episodic_memory_node)
workflow.add_node("semantic", semantic_memory_node)
workflow.add_node("procedural", procedural_memory_node)
workflow.add_node("consolidate", consolidate_memory_node)
workflow.add_node("extract", extract_to_working_memory)
workflow.add_node("arithmetic", arithmetic_agent)
workflow.add_node("tools", tool_node)
workflow.add_node("final_response", final_response)

# --- Define Routing Path ---
def should_end_or_save(state: AgentState) -> Literal["consolidate", "end"]:
    last_msg = state["messages"][-1].content.lower()
    if any(word in last_msg for word in ["bye", "exit", "quit", "done"]):
        return "consolidate"
    return "end"

# 3. Link final_response to either END or Consolidate
workflow.add_conditional_edges(
    "final_response",
    should_end_or_save,
    {
        "consolidate": "consolidate",
        "end": END
    }
)

# Start with the Manager
workflow.add_edge(START, "extract")
workflow.add_edge("extract", "manager")

# Router decides between Memory and Arithmetic
def route_decision(state: AgentState):
    return state["next_memory_step"]

workflow.add_conditional_edges(
    "manager",
    route_decision,
    {
        "working": "working",
        "episodic": "episodic",
        "semantic": "semantic",
        "arithmetic": "arithmetic",
        "procedural": "procedural"  # Added this link
    }
)

# --- The Arithmetic Loop ---
# This is where the 'should_continue' logic creates the loop
workflow.add_conditional_edges(
    "arithmetic",
    should_continue,
    {
        "tools": "tools",
        "end": END  # This links the string "end" to the LangGraph END object
    }
)

# After tools are executed, ALWAYS go back to the arithmetic agent to see if it's 'done'
workflow.add_edge("tools", "arithmetic")

# --- Memory Path ---
# Memory nodes gather context, then pass it to final_response
workflow.add_edge("working", "final_response")
workflow.add_edge("episodic", "final_response")
workflow.add_edge("semantic", "final_response")
workflow.add_edge("procedural", "final_response")
workflow.add_edge("final_response", END)
workflow.add_edge("consolidate", END)

# Compile
app = workflow.compile()
# Assuming your compiled graph is called 'app'
if __name__ == "__main__":
    config = {"user_id": "user_1", "session_id": "sess_1"}
    # Seed Redis
    redis_client.set(f"wm:{config['session_id']}:diet", "vegan", ex=1800)
    
    print("--- Memory Seeded! ---")
    print("--- Agent is Ready! (Type 'exit' to quit) ---")

    messages = []

    def debug_check_qdrant():
        try:
            results = APP_VECTOR_STORE.scroll(
                collection_name="semantic_memory",
                limit=10,
                with_payload=True
            )
            print("--- DEBUG: CURRENT QDRANT ENTRIES ---")
            for point in results[0]:
                print(f"ID: {point.id} | Fact: {point.payload.get('concept')}: {point.payload.get('definition')}")
        except Exception as e:
            print(f"Qdrant Debug Error: {e}")

    # Call it here
    debug_check_qdrant()

    while True:
        user_input = input("\nUser: ")
        
        from langchain_core.messages import HumanMessage
        messages.append(HumanMessage(content=user_input))

        initial_state = {
            "messages": messages,
            "user_id": config["user_id"],
            "session_id": config["session_id"],
            "working_memory": {},
            "episodic_context": [],
            "semantic_context": [],
            "procedural_context": [],
            "next_memory_step": "",
            "response": ""
        }

        try:
            # Run the graph
            for output in app.stream(initial_state):
                if "__end__" in output:
                    continue

                for key, value in output.items():
                    print(f"--- Finished Node: {key} ---")
                    
                    # Check if we just finished consolidating
                    if key == "consolidate":
                        print("!!! DATA PERMANENTLY SAVED TO QDRANT !!!")

                    if isinstance(value, dict) and "messages" in value:
                        for m in value["messages"]:
                            if not messages or m.content != messages[-1].content:
                                messages.append(m)

            if messages:
                print(f"\nAssistant: {messages[-1].content}")

            # NOW check if we should close the script after the graph has run
            if user_input.lower() in ["exit", "quit", "done"]:
                print("--- Closing Session ---")
                break

        except Exception as e:
            print(f"Graph Error: {e}")