# 🧠 Knowledge Graph RAG with Neo4j + LangChain

This repository demonstrates how to build a **Knowledge Graph** for use in **Graph RAG** (Retrieval-Augmented Generation) using [Neo4j](https://neo4j.com/) and [LangChain](https://www.langchain.com/). It includes both a Jupyter notebook for Knowledge Graph construction and a **GraphRAG MCP Server** that can be deployed with Docker and integrated with tools like n8n.

---

## 🎯 Features

- **Knowledge Graph RAG**: Build and query knowledge graphs using Neo4j
- **MCP Server**: Model Context Protocol server for integration with n8n and other tools
- **Docker Support**: Containerized deployment with Docker Compose
- **Neo4j AuraDB Support**: Connect to cloud-hosted Neo4j instances
- **Hybrid Search**: Combines graph traversal and vector similarity search

---

## 📋 Requirements

Before getting started, make sure the following tools are installed on your machine:

### 🛠️ Tools

* **Python 3.8+**
  [https://www.python.org/downloads/](https://www.python.org/downloads/)

* **Docker Desktop**
  [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)


---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Kanisorn-S/agentic-graphrag-langchain.git
cd agentic-graphrag-langchain
```

---

### 2. Set Up Environment Variables

Copy the `.env.example` file and populate with your credentials:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
# .env
GEMINI_API_KEY=your-gemini-api-key
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-auradb-password
LLM_TYPE=ollama  # or 'google' for Gemini
OLLAMA_MODEL=llama3.1
```

---

### 3. Choose Your Deployment Method

#### Option A: Docker Deployment (Recommended)

**For GraphRAG MCP Server with AuraDB:**
```bash
# Start the GraphRAG MCP Server (connects to your AuraDB)
docker-compose up --build 
```

The services will be available at:
- **GraphRAG MCP Server**: [http://localhost:8000](http://localhost:8000)
- **Neo4j Browser** (if running locally): [http://localhost:7474](http://localhost:7474)

#### Option B: Local Python Development

1. **Set Up a Python Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate       # Mac/Linux
   venv\Scripts\activate          # Windows
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Server:**
   ```bash
   python graphrag_mcp_server.py
   ```

---

### 4. Using the GraphRAG MCP Server

#### Available Endpoints:

- **`GET /query`**: Query the knowledge graph with natural language
- **`GET /entities`**: Extract entities from text
- **`GET /graph_search`**: Search for entity relationships
- **`GET /vector_search`**: Perform vector similarity search
- **`GET /health`**: Health check

#### Example API Calls:

```bash
# Query the knowledge graph
curl "http://localhost:8000/query?query=การจ่ายภาษีสินค้าต่างประเทศ"

# Extract entities
curl "http://localhost:8000/entities?text=ภาษีมูลค่าเพิ่มของราชอาณาจักร"

# Search graph relationships
curl "http://localhost:8000/graph_search?entity=ราชอาณาจักร"
```

---

### 5. Integration with n8n

The GraphRAG MCP Server implements the Model Context Protocol and can be integrated with n8n:

**n8n MCP Configuration:**
- **Command**: `python`
- **Arguments**: `["path/to/graphrag_mcp_server.py"]`
- **Environment Variables**: Your `.env` variables

See `n8n_configuration.md` for detailed setup instructions.

---

### 6. Jupyter Notebook Development


Open the notebook file (`graphrag.ipynb`) and execute the cells step by step to build and query your knowledge graph. Make sure the notebook is using the virtual environment you've just created.

---

### 7. Stop Services 

When you are done, run

```bash
docker-compose down
```

This will stop all running services.

---

## 🐳 Docker Services

The `docker-compose.yml` file includes:

- **stirling-pdf**: PDF processing tools (port 8080)
- **graphrag-mcp**: GraphRAG MCP Server (port 8000)

### Environment Variables

All Neo4j and LLM configurations are loaded from your `.env` file:

- `NEO4J_URI` - Neo4j connection URI (local or AuraDB)
- `NEO4J_USERNAME` - Neo4j username
- `NEO4J_PASSWORD` - Neo4j password
- `LLM_TYPE` - LLM provider (`ollama` or `google`)
- `OLLAMA_MODEL` - Ollama model name
- `GEMINI_API_KEY` - Google Gemini API key

---

## 🔧 Model Context Protocol (MCP)

The server implements MCP specification for integration with AI tools:

**Available Tools:**
1. **graphrag_query** - Query the knowledge graph
2. **extract_entities** - Extract entities from text
3. **search_graph** - Search graph relationships

**Usage with MCP Clients:**
```json
{
  "tool": "graphrag_query",
  "arguments": {
    "query": "your question here"
  }
}
```

## 🧱 Folder Structure

```
agentic-graphrag-langchain/
├── docs/                    # Document files for knowledge graph
│   ├── 83_84.txt
│   ├── revenue_dep.txt
│   └── ...
├── n8n/                      # n8n workflows
│   ├── subworkflow/          # Sub-agents and tools
|   │   ├── Google_Search_and_Scrape.json  # Tools for searching Google
|   │   |── Retriever_Agent_HTTP.json           # Retriver Agent with access to Hybrid Search
|   │   └── Retriever_Agent_MCP.json            # Retriver Agent with access to Vector Search and Graph Search tools
│   └── workflow/
|       |── Document_Parsing_MultiModal_Graph_API.json   # Main Workflow for Document Insertion and GraphRAG Using HTTP Retriever
|       └── Document_Parsing_MultiModal_Graph_MCP.json   # Main Workflow for Document Insertion and GraphRAG Using MCP Retriever
├── .env.example             # Template env file
├── .env                     # Your environment variables
├── docker-compose.yml       # Docker services definition
├── Dockerfile               # GraphRAG MCP Server container
├── requirements.txt         # Python dependencies
├── graphrag_mcp_server.py   # GraphRAG MCP Server (FastAPI + MCP)
├── graphrag.ipynb          # Jupyter notebook for exploration
├── n8n_configuration.md    # n8n integration guide
└── README.md               # This file
```
