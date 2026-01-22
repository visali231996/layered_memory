# Layered memory
in terminal:
pip install redis
wsl -d Ubuntu redis-cli ping
for monitoring purpose:wsl -d Ubuntu -e /usr/bin/redis-cli monitor

how to open ubuntu in PS:
to know what are running right now  : wsl -l -v
opens ubuntu in PS                  : wsl -d Ubuntu
in ubuntu:
for installation    :        sudo apt install redis-server -y
runs server client  :      sudo service redis-server start
sudo password for visali: visali

## About
Has a manager which routes to different memories based on user query , and if asked about arithmetic operations ,transfers to llm,which will call tools 
 
 
## About redis:
 Redis is acting as a High-Speed Volatile Cache


1. Why is Redis used here?
In your script, Redis is acting as a High-Speed Volatile Cache.

Automation: Notice your code doesn't have a DELETE command. By using ex=TTL_SECONDS, you are telling Redis to manage the memory for you.

Latency Reduction: If "hello" was a complex calculation that took 5 seconds to generate, you could store it in Redis for 10 seconds. Any other user asking for it during that window gets the answer instantly.

Layered Strategy: It allows you to store data that is "important right now" but "useless in 10 minutes" (like a one-time password or a temporary login session).

2. Which memory is called?
When your script runs r.set or r.get, it is calling RAM (Random Access Memory).

RAM vs. Disk: Standard databases (like MySQL or PostgreSQL) primarily store data on the Hard Drive (SSD/HDD). Redis stores its data directly in the system's memory chips.

Speed: Accessing RAM is roughly 1,000 to 10,000 times faster than accessing a disk. This is why Redis can handle millions of operations per second while a traditional database might struggle with thousands.

Mongom Db: No sql database...stores data in Harddisk,...usually stores data in json format..no need of columns and rows...While Redis is your "Speed Layer" (L1), MongoDB is typically your "Storage Layer" (L2). They are often used together because they have opposite strengths.this allows you to store complex, nested information like a whole user profile, a blog post with comments, or a shopping cart in a single "file."
in mongo db, data is stored  permenantly,query retrieval is complex,filter based and data capacity is massive(based on disk size)

## 1. Working Memory (Short-Term / Cache)
Think of this as the "Post-it Note" of your agent. It handles immediate, active context.

Technology: Redis (with a TTL/Time-to-Live of 30 minutes).

What is stored:

Facts mentioned in the current conversation (e.g., "I'm hungry," "My name is Alex").

Transient data that doesn't need to be kept forever but is vital for the next 5 minutes of chat.

Data Structure: Key-Value pairs (e.g., user_1:hobby: "hiking").

## 2. Episodic Memory (Experience / Timeline)
This is the "Autobiography" of the agent. It stores specific events tied to a time and place.

Technology: Qdrant (Vector Store).

What is stored:

Summaries of past interactions ("On Monday, we discussed the vegan diet project").

Specific "Episodes" or logs of what happened.

Data Structure: Vectors + Metadata (Timestamp, UserID, Event Summary).

## 3. Semantic Memory (Knowledge / Facts)
This is the "Encyclopedia" or "Personal Profile." It stores general knowledge and learned facts about the user that are independent of a specific conversation.

Technology: Qdrant (Vector Store).

What is stored:

Permanent user preferences ("User is allergic to peanuts").

World knowledge the agent has learned.

Professional details ("User is a Software Engineer").

Data Structure: Concept-based Vectors (e.g., Concept: Job, Definition: Software Engineer).

## 4. Procedural Memory (Skills / Rules)
This is the "Instruction Manual." It stores the "How-to" for specific tasks.

Technology: MongoDB (Document Store).

What is stored:

Step-by-step guides ("How to check if food is vegan").

Business rules or protocols the agent must follow.

Coding standards or workflows.

Data Structure: Structured JSON/Documents (Title, Steps, Requirements).

## Running
` python codefinal.py`
make sure that u are runnin qdrant url,mongodb,redis url
REDIS_URL = "redis://localhost:6379"
QDRANT_URL = "http://localhost:6333"
MONGODB_URL = "mongodb://localhost:27017"