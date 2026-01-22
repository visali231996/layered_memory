"""
Agentic Memory Tutorial - Simple Single-File Implementation
============================================================

This script teaches the four types of agentic memory:
1. Working Memory (Redis) - Current session state
2. Episodic Memory (Qdrant) - Past experiences  
3. Semantic Memory (Qdrant) - Facts and knowledge
4. Procedural Memory (MongoDB) - Learned skills

Run with: python agentic_memory_tutorial.py

Prerequisites:
    pip install redis qdrant-client pymongo sentence-transformers langgraph langchain-core
    
    # Start services:
    docker run -d -p 6379:6379 redis:7-alpine
    docker run -d -p 6333:6333 qdrant/qdrant
    docker run -d -p 27017:27017 mongo:7
"""

import json
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

# =============================================================================
# IMPORTS
# =============================================================================

import redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add


# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = "redis://localhost:6379"
QDRANT_URL = "http://localhost:6333"
MONGODB_URL = "mongodb://localhost:27017"

# Initialize clients
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
qdrant_client = QdrantClient(url=QDRANT_URL)
mongo_client = MongoClient(MONGODB_URL)
mongo_db = mongo_client["agent_memory"]

# Initialize embedding model (small model for demo)
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2


# =============================================================================
# PART 1: WORKING MEMORY (Redis)
# =============================================================================
# Working memory is like your mental scratchpad - it holds what you're 
# currently thinking about and expires automatically.

def working_memory_set(session_id: str, key: str, value: dict, ttl: int = 300):
    """
    Store something in working memory.
    
    Args:
        session_id: Unique session identifier
        key: What we're storing (e.g., "current_goal", "user_intent")
        value: The data to store
        ttl: Time-to-live in seconds (default 5 minutes)
    """
    redis_key = f"wm:{session_id}:{key}"
    redis_client.setex(redis_key, ttl, json.dumps(value))


def working_memory_get(session_id: str, key: str) -> dict | None:
    """Retrieve something from working memory."""
    redis_key = f"wm:{session_id}:{key}"
    data = redis_client.get(redis_key)
    return json.loads(data) if data else None


def working_memory_add_observation(session_id: str, observation: dict, max_items: int = 10):
    """
    Add an observation to the observation stream.
    Uses a Redis list to maintain ordered observations.
    """
    redis_key = f"wm:{session_id}:observations"
    observation["timestamp"] = datetime.utcnow().isoformat()
    
    redis_client.lpush(redis_key, json.dumps(observation))
    redis_client.ltrim(redis_key, 0, max_items - 1)  # Keep only recent items
    redis_client.expire(redis_key, 600)  # Expire after 10 minutes


def working_memory_get_observations(session_id: str, limit: int = 5) -> list:
    """Get recent observations."""
    redis_key = f"wm:{session_id}:observations"
    items = redis_client.lrange(redis_key, 0, limit - 1)
    return [json.loads(item) for item in items]


def working_memory_get_full_context(session_id: str) -> dict:
    """Get all working memory for a session."""
    context = {}
    
    # Get all keys for this session
    pattern = f"wm:{session_id}:*"
    for key in redis_client.scan_iter(match=pattern):
        short_key = key.split(":")[-1]
        if short_key == "observations":
            context["observations"] = working_memory_get_observations(session_id)
        else:
            context[short_key] = working_memory_get(session_id, short_key)
    
    return context


def demo_working_memory():
    """Demonstrate working memory operations."""
    print("\n" + "="*60)
    print("📝 WORKING MEMORY (Redis)")
    print("="*60)
    
    session_id = f"demo_{uuid4().hex[:8]}"
    
    # Set current goal
    working_memory_set(session_id, "current_goal", {
        "goal": "Help user analyze sales data",
        "priority": "high"
    })
    print("✓ Set current goal")
    
    # Set user intent
    working_memory_set(session_id, "user_intent", {
        "intent": "data_analysis",
        "confidence": 0.9
    })
    print("✓ Set user intent")
    
    # Add some observations
    working_memory_add_observation(session_id, {"type": "user_upload", "file": "sales.csv"})
    working_memory_add_observation(session_id, {"type": "user_question", "text": "Show me trends"})
    print("✓ Added observations")
    
    # Retrieve everything
    context = working_memory_get_full_context(session_id)
    print(f"\n📋 Full Working Memory Context:")
    print(json.dumps(context, indent=2))
    
    # Show TTL
    ttl = redis_client.ttl(f"wm:{session_id}:current_goal")
    print(f"\n⏰ TTL remaining: {ttl} seconds (auto-expires!)")


# =============================================================================
# PART 2: EPISODIC MEMORY (Qdrant)
# =============================================================================
# Episodic memory stores past experiences - conversations, interactions,
# events. It's searchable by semantic similarity.

def setup_episodic_collection():
    """Create the episodic memory collection if it doesn't exist."""
    collections = [c.name for c in qdrant_client.get_collections().collections]
    
    if "episodic_memory" not in collections:
        qdrant_client.create_collection(
            collection_name="episodic_memory",
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        print("✓ Created episodic_memory collection")


def episodic_memory_store(user_id: str, content: str, summary: str, metadata: dict = None):
    """
    Store an episode (experience/interaction).
    
    Args:
        user_id: Who this memory belongs to
        content: Full content of the interaction
        summary: Brief summary for context
        metadata: Additional info (topics, sentiment, etc.)
    """
    # Generate embedding from content
    embedding = embedder.encode(f"{summary} {content}").tolist()
    
    # Create unique ID
    point_id = uuid4().hex
    
    # Build payload
    payload = {
        "user_id": user_id,
        "content": content,
        "summary": summary,
        "timestamp": datetime.utcnow().isoformat(),
        "episode_id": point_id,
        **(metadata or {})
    }
    
    # Store in Qdrant
    qdrant_client.upsert(
        collection_name="episodic_memory",
        points=[PointStruct(id=point_id, vector=embedding, payload=payload)]
    )
    
    return point_id


def episodic_memory_search(query: str, user_id: str = None, limit: int = 5) -> list:
    """
    Search episodic memory by semantic similarity.
    
    Args:
        query: What to search for
        user_id: Optional filter by user
        limit: Max results
    
    Returns:
        List of relevant episodes with similarity scores
    """
    # Generate query embedding
    query_embedding = embedder.encode(query).tolist()
    
    # Build filter if user_id provided
    search_filter = None
    if user_id:
        search_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
    
    # Search
    results = qdrant_client.query_points(
        collection_name="episodic_memory",
        query=query_embedding,
        query_filter=search_filter,
        limit=limit
    )
    
    return [
        {
            "content": r[1][0].payload["content"],
            "summary": r[1][0].payload["summary"],
            "timestamp": r[1][0].payload["timestamp"],
            "score": r[1][0].score,
            **{k: v for k, v in r[1][0].payload.items() if k not in ["content", "summary", "timestamp", "user_id"]}
        }
        for r in results
    ]


def demo_episodic_memory():
    """Demonstrate episodic memory operations."""
    print("\n" + "="*60)
    print("🎬 EPISODIC MEMORY (Qdrant)")
    print("="*60)
    
    setup_episodic_collection()
    user_id = f"user_{uuid4().hex[:8]}"
    
    # Store some episodes
    episodes = [
        ("We discussed Python best practices for data processing. User mentioned they prefer pandas over polars.",
         "Python data processing discussion",
         {"topics": ["python", "pandas", "data"]}),
        
        ("User asked about machine learning model deployment. Recommended using FastAPI with Docker.",
         "ML deployment advice",
         {"topics": ["ml", "deployment", "docker"]}),
        
        ("Helped debug a SQL query performance issue. The problem was missing indexes on join columns.",
         "SQL debugging session",
         {"topics": ["sql", "performance", "debugging"]}),
        
        ("User shared their project goals: build a recommendation system for e-commerce.",
         "Project planning discussion",
         {"topics": ["recommendation", "ecommerce", "planning"]}),
    ]
    
    for content, summary, metadata in episodes:
        episodic_memory_store(user_id, content, summary, metadata)
    print(f"✓ Stored {len(episodes)} episodes")
    
    # Search for relevant episodes
    print("\n🔍 Searching for 'machine learning project':")
    results = episodic_memory_search("machine learning project", user_id, limit=3)
    
    for i, r in enumerate(results, 1):
        print(f"\n   {i}. [{r['score']:.2f}] {r['summary']}")
        print(f"      Topics: {r.get('topics', [])}")


# =============================================================================
# PART 3: SEMANTIC MEMORY (Qdrant)
# =============================================================================
# Semantic memory stores facts and knowledge - things that are true
# regardless of when they were learned.

def setup_semantic_collection():
    """Create the semantic memory collection if it doesn't exist."""
    collections = [c.name for c in qdrant_client.get_collections().collections]
    
    if "semantic_memory" not in collections:
        qdrant_client.create_collection(
            collection_name="semantic_memory",
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        print("✓ Created semantic_memory collection")


def semantic_memory_store(
    user_id: str,
    knowledge: str,
    knowledge_type: str = "fact",  # fact, preference, skill, relationship
    confidence: float = 0.8,
    metadata: dict = None
):
    """
    Store a piece of knowledge.
    
    Args:
        user_id: Who this knowledge is about
        knowledge: The factual statement
        knowledge_type: Type of knowledge (fact/preference/skill/relationship)
        confidence: How confident we are (0-1)
        metadata: Additional context
    """
    embedding = embedder.encode(knowledge).tolist()
    point_id = uuid4().hex
    
    payload = {
        "user_id": user_id,
        "knowledge": knowledge,
        "knowledge_type": knowledge_type,
        "confidence": confidence,
        "created_at": datetime.utcnow().isoformat(),
        "memory_id": point_id,
        **(metadata or {})
    }
    
    qdrant_client.upsert(
        collection_name="semantic_memory",
        points=[PointStruct(id=point_id, vector=embedding, payload=payload)]
    )
    
    return point_id


def semantic_memory_search(
    query: str,
    user_id: str = None,
    knowledge_type: str = None,
    min_confidence: float = 0.0,
    limit: int = 5
) -> list:
    """
    Search semantic memory.
    
    Can filter by user, knowledge type, and minimum confidence.
    """
    query_embedding = embedder.encode(query).tolist()
    
    # Build filters
    conditions = []
    if user_id:
        conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
    if knowledge_type:
        conditions.append(FieldCondition(key="knowledge_type", match=MatchValue(value=knowledge_type)))
    
    search_filter = Filter(must=conditions) if conditions else None
    
    results = qdrant_client.query_points(
        collection_name="semantic_memory",
        query=query_embedding,
        query_filter=search_filter,
        limit=limit
    )
    
    # Filter by confidence after retrieval
    return [
        {
            "knowledge": r[1][0].payload["knowledge"],
            "type": r[1][0].payload["knowledge_type"],
            "confidence": r[1][0].payload["confidence"],
            "score": r[1][0].score
        }
        for r in results
        if r[1][0].payload["confidence"] >= min_confidence
    ]


def demo_semantic_memory():
    """Demonstrate semantic memory operations."""
    print("\n" + "="*60)
    print("🧠 SEMANTIC MEMORY (Qdrant)")
    print("="*60)
    
    setup_semantic_collection()
    user_id = f"user_{uuid4().hex[:8]}"
    
    # Store various types of knowledge
    knowledge_items = [
        ("User prefers Python over JavaScript for backend work", "preference", 0.9),
        ("User works at a fintech startup as a data engineer", "fact", 0.95),
        ("User is experienced with PostgreSQL and Redis", "skill", 0.85),
        ("User's team lead is named Sarah", "relationship", 0.8),
        ("User prefers morning meetings over afternoon ones", "preference", 0.7),
        ("User is learning Rust in their spare time", "fact", 0.75),
    ]
    
    for knowledge, ktype, confidence in knowledge_items:
        semantic_memory_store(user_id, knowledge, ktype, confidence)
    print(f"✓ Stored {len(knowledge_items)} knowledge items")
    
    # Search for preferences
    print("\n🔍 Searching for 'programming preferences':")
    results = semantic_memory_search(
        "programming language preferences",
        user_id=user_id,
        knowledge_type="preference",
        limit=3
    )
    
    for r in results:
        print(f"   [{r['confidence']:.0%}] {r['knowledge']}")
    
    # Search for skills
    print("\n🔍 Searching for 'database skills':")
    results = semantic_memory_search(
        "database experience",
        user_id=user_id,
        knowledge_type="skill",
        limit=3
    )
    
    for r in results:
        print(f"   [{r['confidence']:.0%}] {r['knowledge']}")


# =============================================================================
# PART 4: PROCEDURAL MEMORY (MongoDB)
# =============================================================================
# Procedural memory stores "how to do things" - tool usage patterns,
# successful strategies, and learned workflows.

def procedural_memory_store(
    user_id: str,
    name: str,
    tool_name: str,
    steps: list,
    trigger_patterns: list = None
):
    """
    Store a procedure (learned skill).
    
    Args:
        user_id: Who learned this
        name: Name of the procedure
        tool_name: What tool this is for
        steps: List of steps to execute
        trigger_patterns: When to use this procedure
    """
    procedure = {
        "procedure_id": uuid4().hex,
        "user_id": user_id,
        "name": name,
        "tool_name": tool_name,
        "steps": steps,
        "trigger_patterns": trigger_patterns or [],
        "created_at": datetime.utcnow(),
        "total_executions": 0,
        "successful_executions": 0,
        "success_rate": 0.0
    }
    
    mongo_db.procedures.insert_one(procedure)
    return procedure["procedure_id"]


def procedural_memory_record_execution(
    procedure_id: str,
    success: bool,
    duration_ms: int,
    error: str = None
):
    """
    Record an execution of a procedure.
    Updates success rate automatically.
    """
    # Record the trace
    trace = {
        "trace_id": uuid4().hex,
        "procedure_id": procedure_id,
        "timestamp": datetime.utcnow(),
        "success": success,
        "duration_ms": duration_ms,
        "error": error
    }
    mongo_db.traces.insert_one(trace)
    
    # Update procedure stats
    procedure = mongo_db.procedures.find_one({"procedure_id": procedure_id})
    if procedure:
        total = procedure["total_executions"] + 1
        successful = procedure["successful_executions"] + (1 if success else 0)
        
        mongo_db.procedures.update_one(
            {"procedure_id": procedure_id},
            {"$set": {
                "total_executions": total,
                "successful_executions": successful,
                "success_rate": successful / total
            }}
        )


def procedural_memory_find_best(tool_name: str, user_id: str = None) -> dict | None:
    """
    Find the best procedure for a tool based on success rate.
    """
    query = {"tool_name": tool_name}
    if user_id:
        query["user_id"] = user_id
    
    procedures = list(mongo_db.procedures.find(query).sort("success_rate", -1).limit(1))
    return procedures[0] if procedures else None


def procedural_memory_get_stats(procedure_id: str) -> dict:
    """Get statistics for a procedure."""
    procedure = mongo_db.procedures.find_one({"procedure_id": procedure_id})
    if not procedure:
        return {}
    
    # Get recent traces
    recent_traces = list(
        mongo_db.traces.find({"procedure_id": procedure_id})
        .sort("timestamp", -1)
        .limit(10)
    )
    
    return {
        "name": procedure["name"],
        "total_executions": procedure["total_executions"],
        "success_rate": procedure["success_rate"],
        "recent_outcomes": [t["success"] for t in recent_traces]
    }


def demo_procedural_memory():
    """Demonstrate procedural memory operations."""
    print("\n" + "="*60)
    print("⚙️ PROCEDURAL MEMORY (MongoDB)")
    print("="*60)
    
    user_id = f"user_{uuid4().hex[:8]}"
    
    # Create a procedure
    procedure_id = procedural_memory_store(
        user_id=user_id,
        name="CSV Data Analysis",
        tool_name="data_analyzer",
        steps=[
            {"action": "load_csv", "params": {"encoding": "utf-8"}},
            {"action": "check_missing_values", "params": {}},
            {"action": "generate_summary_stats", "params": {"columns": "all"}},
            {"action": "create_visualizations", "params": {"type": "auto"}}
        ],
        trigger_patterns=["analyze csv", "data analysis", "explore data"]
    )
    print(f"✓ Created procedure: CSV Data Analysis")
    
    # Simulate some executions
    executions = [
        (True, 150),
        (True, 180),
        (True, 165),
        (False, 200),  # One failure
        (True, 145),
        (True, 170),
    ]
    
    for success, duration in executions:
        procedural_memory_record_execution(procedure_id, success, duration)
    print(f"✓ Recorded {len(executions)} executions")
    
    # Get stats
    stats = procedural_memory_get_stats(procedure_id)
    print(f"\n📊 Procedure Statistics:")
    print(f"   Name: {stats['name']}")
    print(f"   Total Executions: {stats['total_executions']}")
    print(f"   Success Rate: {stats['success_rate']:.0%}")
    print(f"   Recent Outcomes: {['✓' if s else '✗' for s in stats['recent_outcomes']]}")
    
    # Find best procedure for tool
    best = procedural_memory_find_best("data_analyzer", user_id)
    if best:
        print(f"\n🏆 Best procedure for 'data_analyzer': {best['name']} ({best['success_rate']:.0%} success)")


# =============================================================================
# PART 5: LANGGRAPH INTEGRATION
# =============================================================================
# Bringing it all together with LangGraph for orchestrated memory retrieval.

class AgentState(TypedDict):
    """State that flows through our agent graph."""
    user_message: str
    user_id: str
    session_id: str
    working_memory: dict
    episodic_context: list
    semantic_context: list
    procedural_context: list
    response: str


def node_load_working_memory(state: AgentState) -> dict:
    """Load current working memory."""
    context = working_memory_get_full_context(state["session_id"])
    return {"working_memory": context}


def node_retrieve_episodic(state: AgentState) -> dict:
    """Retrieve relevant past episodes."""
    results = episodic_memory_search(
        state["user_message"],
        user_id=state["user_id"],
        limit=3
    )
    return {"episodic_context": results}


def node_retrieve_semantic(state: AgentState) -> dict:
    """Retrieve relevant knowledge."""
    results = semantic_memory_search(
        state["user_message"],
        user_id=state["user_id"],
        min_confidence=0.5,
        limit=3
    )
    return {"semantic_context": results}


def node_retrieve_procedural(state: AgentState) -> dict:
    """Find relevant procedures."""
    # Simple keyword matching for demo
    procedures = list(mongo_db.procedures.find({"user_id": state["user_id"]}).limit(2))
    return {"procedural_context": [
        {"name": p["name"], "tool": p["tool_name"], "success_rate": p["success_rate"]}
        for p in procedures
    ]}


def node_generate_response(state: AgentState) -> dict:
    """Generate response based on all memory context."""
    # In a real system, this would call an LLM
    # For demo, we just summarize what we found
    
    response_parts = [f"Processing: '{state['user_message']}'"]
    
    if state["working_memory"]:
        response_parts.append(f"\n📝 Working Memory: {len(state['working_memory'])} items loaded")
    
    if state["episodic_context"]:
        response_parts.append(f"\n🎬 Found {len(state['episodic_context'])} relevant past interactions")
        for ep in state["episodic_context"][:2]:
            response_parts.append(f"   - {ep['summary']}")
    
    if state["semantic_context"]:
        response_parts.append(f"\n🧠 Found {len(state['semantic_context'])} relevant facts")
        for sm in state["semantic_context"][:2]:
            response_parts.append(f"   - {sm['knowledge']}")
    
    if state["procedural_context"]:
        response_parts.append(f"\n⚙️ Found {len(state['procedural_context'])} applicable procedures")
        for pm in state["procedural_context"]:
            response_parts.append(f"   - {pm['name']} ({pm['success_rate']:.0%} success)")
    
    return {"response": "\n".join(response_parts)}


def build_memory_agent():
    """Build a LangGraph agent with memory retrieval."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("load_working_memory", node_load_working_memory)
    workflow.add_node("retrieve_episodic", node_retrieve_episodic)
    workflow.add_node("retrieve_semantic", node_retrieve_semantic)
    workflow.add_node("retrieve_procedural", node_retrieve_procedural)
    workflow.add_node("generate_response", node_generate_response)
    
    # Define the flow
    workflow.set_entry_point("load_working_memory")
    workflow.add_edge("load_working_memory", "retrieve_episodic")
    workflow.add_edge("retrieve_episodic", "retrieve_semantic")
    workflow.add_edge("retrieve_semantic", "retrieve_procedural")
    workflow.add_edge("retrieve_procedural", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()


def demo_langgraph_integration():
    """Demonstrate LangGraph memory integration."""
    print("\n" + "="*60)
    print("🔗 LANGGRAPH INTEGRATION")
    print("="*60)
    
    # Setup - reuse data from previous demos
    # user_id = f"user_{uuid4().hex[:8]}"
    # session_id = f"session_{uuid4().hex[:8]}"
    user_id = "user_12345678"
    session_id = "session_12345678"
    
    # Populate some memory first
    print("\n📦 Setting up memory...")
    
    # Working memory
    working_memory_set(session_id, "current_goal", {"goal": "Help with data analysis"})
    working_memory_add_observation(session_id, {"type": "greeting", "text": "User said hello"})
    
    # Episodic memory
    setup_episodic_collection()
    episodic_memory_store(user_id, 
        "Helped user set up a Python data pipeline with pandas",
        "Python data pipeline setup",
        {"topics": ["python", "pandas", "pipeline"]}
    )
    
    # Semantic memory
    setup_semantic_collection()
    semantic_memory_store(user_id, "User prefers pandas for data manipulation", "preference", 0.9)
    semantic_memory_store(user_id, "User works with CSV files frequently", "fact", 0.85)
    
    # Procedural memory
    procedural_memory_store(
        user_id, "CSV Processing", "pandas_tool",
        [{"action": "read_csv"}, {"action": "clean_data"}, {"action": "analyze"}],
        ["process csv", "analyze data"]
    )
    procedural_memory_record_execution(
        mongo_db.procedures.find_one({"user_id": user_id})["procedure_id"],
        True, 100
    )
    
    print("✓ Memory populated")
    
    # Build and run the agent
    agent = build_memory_agent()
    
    initial_state = {
        "user_message": "Can you help me analyze some CSV data?",
        "user_id": user_id,
        "session_id": session_id,
        "working_memory": {},
        "episodic_context": [],
        "semantic_context": [],
        "procedural_context": [],
        "response": ""
    }
    
    print("\n🤖 Running memory-aware agent...")
    print(f"   User: '{initial_state['user_message']}'")
    
    # Execute the graph
    final_state = agent.invoke(initial_state)
    
    print("\n" + "-"*40)
    print("📤 Agent Response:")
    print(final_state["response"])


# =============================================================================
# PART 6: MEMORY CONSOLIDATION (Simple Version)
# =============================================================================
# Consolidation extracts patterns from episodic memory into semantic memory.

def consolidate_memories(user_id: str):
    """
    Simple consolidation: look for patterns in episodic memory
    and extract them as semantic facts.
    
    In production, you'd use an LLM for this.
    """
    print("\n" + "="*60)
    print("🔄 MEMORY CONSOLIDATION")
    print("="*60)
    
    # Get recent episodes
    episodes = episodic_memory_search("", user_id=user_id, limit=20)
    
    if len(episodes) < 3:
        print("   Not enough episodes to consolidate")
        return
    
    # Simple pattern detection: count topic occurrences
    topic_counts = {}
    for ep in episodes:
        for topic in ep.get("topics", []):
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    # Topics that appear multiple times become semantic facts
    frequent_topics = [t for t, c in topic_counts.items() if c >= 2]
    
    print(f"   Found {len(episodes)} episodes")
    print(f"   Frequent topics: {frequent_topics}")
    
    for topic in frequent_topics:
        knowledge = f"User frequently discusses {topic}"
        semantic_memory_store(user_id, knowledge, "fact", 0.7)
        print(f"   ✓ Extracted: '{knowledge}'")
    
    # Analyze procedural success patterns
    procedures = list(mongo_db.procedures.find({"user_id": user_id}))
    for proc in procedures:
        if proc["success_rate"] > 0.8 and proc["total_executions"] >= 3:
            knowledge = f"User successfully uses {proc['tool_name']} for {proc['name']}"
            semantic_memory_store(user_id, knowledge, "skill", proc["success_rate"])
            print(f"   ✓ Extracted: '{knowledge}'")


# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    """Run all demonstrations."""
    print("="*60)
    print("🧠 AGENTIC MEMORY TUTORIAL")
    print("="*60)
    print("\nThis tutorial demonstrates four types of agent memory:")
    print("1. Working Memory (Redis) - Current session state")
    print("2. Episodic Memory (Qdrant) - Past experiences")
    print("3. Semantic Memory (Qdrant) - Facts and knowledge")
    print("4. Procedural Memory (MongoDB) - Learned skills")
    print("\nPlus: LangGraph integration and Memory Consolidation")
    
    try:
        # Test connections
        print("\n🔌 Testing connections...")
        redis_client.ping()
        print("   ✓ Redis connected")
        qdrant_client.get_collections()
        print("   ✓ Qdrant connected")
        mongo_client.admin.command('ping')
        print("   ✓ MongoDB connected")
        
        # Run demos
        demo_working_memory()
        demo_episodic_memory()
        demo_semantic_memory()
        demo_procedural_memory()
        demo_langgraph_integration()
        
        # Consolidation demo (uses data from semantic demo)
        user_id = list(mongo_db.procedures.find().limit(1))[0]["user_id"]
        consolidate_memories(user_id)
        
        print("\n" + "="*60)
        print("✅ TUTORIAL COMPLETE!")
        print("="*60)
        print("\nKey Takeaways:")
        print("• Working Memory: Fast, ephemeral, auto-expires (Redis)")
        print("• Episodic Memory: Experiences, similarity search (Qdrant)")
        print("• Semantic Memory: Facts with confidence scores (Qdrant)")
        print("• Procedural Memory: Skills that improve with use (MongoDB)")
        print("• LangGraph: Orchestrates memory retrieval in agent workflows")
        print("• Consolidation: Extracts patterns into long-term memory")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure all services are running:")
        print("  docker run -d -p 6379:6379 redis:7-alpine")
        print("  docker run -d -p 6333:6333 qdrant/qdrant")
        print("  docker run -d -p 27017:27017 mongo:7")


if __name__ == "__main__":
    main()
