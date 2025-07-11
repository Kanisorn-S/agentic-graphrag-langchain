# 🧠 Knowledge Graph RAG with Neo4j + LangChain

This repository demonstrates how to build a **Knowledge Graph** for use in **Graph RAG** (Retrieval-Augmented Generation) using [Neo4j](https://neo4j.com/) and [LangChain](https://www.langchain.com/). It’s designed to run locally with Docker and Python.

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
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=testpassword
LLM_TYPE=ollama  # Specify the LLM use, either 'ollama' or 'google'
OLLAMA_MODEL=llama3.1 # Specify the model to use with Ollama, e.g., 'llama3.1', 'llama2', etc.
```

---

### 3. Start Neo4j Locally with Docker

```bash
docker-compose up -d
```

This will launch Neo4j at:

* Browser: [http://localhost:7474](http://localhost:7474)
* Bolt endpoint: `bolt://localhost:7687`
* Default credentials:

  * **Username**: `neo4j`
  * **Password**: `testpassword` *(or as set in `docker-compose.yml`)*

---

### 4. Set Up a Python Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

Install required libraries:

```bash
pip install -r requirements.txt
```

---

### 5. Run the Jupyter Notebook


Open the notebook file (`graphrag.ipynb`) and execute the cells step by step to build and query your knowledge graph. Make sure the notebook is using the virtual environment you've just created.

---

## 🧱 Folder Structure

```
agentic-graphrag-langchain/
├── db                   # Files related to Neo4j
|   ├── conf             
|   ├── data             
|   ├── logs             
|   └── plugins          
├── .env.example             # Template env file
├── docker-compose.yml       # Neo4j service definition
├── requirements.txt         # Python dependencies
├── graphrag.ipynb      # Main notebook with Graph RAG logic
└── README.md                # Setup & usage guide
```

---

## 📌 Notes

* The notebook assumes Neo4j is accessible at `bolt://localhost:7687`.
* Update the connection settings in the notebook if your Neo4j credentials or ports differ.
* You can use tools like [Neo4j Desktop](https://neo4j.com/download/) to visualize the graph if needed.
* There are already some documents that have been added to the knowledge graph:
  * 83_84.txt
  * revenue_dep.txt
