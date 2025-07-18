"""
GraphRAG MCP Server using FastAPI
Implements Model Context Protocol server for GraphRAG functionality
"""

import os
import asyncio
from typing import List, Optional, Any, Dict
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel, Field

# FastAPI and MCP imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi_mcp import FastApiMCP

# LangChain and GraphRAG imports
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="GraphRAG MCP Server",
    description="Model Context Protocol server for GraphRAG functionality",
    version="1.0.0"
)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
LLM_TYPE = os.getenv("LLM_TYPE", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qllama/multilingual-e5-small")

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Global variables for initialized components
graph: Optional[Neo4jGraph] = None
llm = None
entity_chain = None
vector_retriever = None
final_chain = None


class Entities(BaseModel):
    """Identifying information about entities"""
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appears in the text",
    )


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="The query to process through GraphRAG")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    result: str = Field(..., description="The result from the GraphRAG chain")
    entities_found: List[str] = Field(..., description="Entities extracted from the query")


async def initialize_components():
    """Initialize all GraphRAG components"""
    global graph, llm, entity_chain, vector_retriever, final_chain
    
    try:
        # Initialize Neo4j Graph
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
        )
        
        # Initialize LLM based on type
        if LLM_TYPE == "ollama":
            print("Using ollama")
            llm = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0,
                base_url="http://host.docker.internal:11434"
            )
        else:
            print("Using Google Gemini")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
            )
        
        # Initialize entity extraction chain
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are extracting organization and person entities from the text"
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            )
        ])
        
        entity_chain = prompt | llm.with_structured_output(Entities)
        
        # Initialize vector embeddings and retriever
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url="http://host.docker.internal:11434"
        )
        
        vector_index = Neo4jVector.from_existing_graph(
            embedding=embeddings,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )
        
        vector_retriever = vector_index.as_retriever()
        
        # Initialize final chain
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        Use natural language and add as much detail as possible. Add your sources if possible.
        Answer:
        """
        
        final_prompt = ChatPromptTemplate.from_template(template)

        def debug_inputs(inputs):
            context = inputs.get("context", "")
            contexts = context.split("\n")
            for el in contexts:
                print(el.strip())
            return inputs

        debug_node = RunnableLambda(debug_inputs)
        
        final_chain = (
            {
                "context": full_retriever,
                "question": RunnablePassthrough()
            }
            | debug_node
            | final_prompt
            | llm 
            | StrOutputParser()
        )
        
        print("All components initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing components: {e}")
        raise


def graph_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit: 2})
            YIELD node, score
            CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 20
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    
    return result

def vector_retrieve(question: str) -> str:
    """
    Using vector similarity search to retrieve relevant documents
    """
    response = vector_retriever.invoke(question)
    result = "\n".join([el.page_content for el in response])
    return result


def full_retriever(question: str) -> str:
    """Full retriever that combines graph and vector data"""
    graph_data = graph_retriever(question)
    vector_data = vector_retrieve(question)
    final_data = f"""Graph data:
    {graph_data}
    Vector data:
    {vector_data}
    """
    return final_data


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    await initialize_components()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "llm_type": LLM_TYPE}

@app.get("/vector_search", operation_id="vector_search")
async def vector_search(query: str):
    """
    Perform a vector search on the knowledge graph
    
    Args:
        query: The query string to search for
        
    Returns:
        List of documents matching the query
    """
    try:
        if not vector_retriever:
            raise HTTPException(status_code=500, detail="Server not properly initialized")
        
        results = vector_retrieve(query)
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing vector search: {str(e)}")

@app.get("/query", response_model=QueryResponse, operation_id="query_knowledge_graph")
async def process_query(query: str):
    """
    Process a query through the GraphRAG pipeline
    
    Args:
        query: The query string to process
        
    Returns:
        QueryResponse containing the result and extracted entities
    """
    try:
        if not final_chain:
            raise HTTPException(status_code=500, detail="Server not properly initialized")
        
        # Extract entities for debugging/transparency
        entities = entity_chain.invoke({"question": query})
        
        # Process through the full chain
        result = final_chain.invoke(query)
        
        return QueryResponse(
            result=result,
            entities_found=entities.names
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def process_query_post(request: QueryRequest):
    """
    Process a query through the GraphRAG pipeline (POST method)
    
    Args:
        request: QueryRequest containing the query
        
    Returns:
        QueryResponse containing the result and extracted entities
    """
    return await process_query(request.query)


@app.get("/entities", operation_id="extract_entities")
async def extract_entities(text: str):
    """
    Extract entities from text
    
    Args:
        text: The text to extract entities from
        
    Returns:
        List of extracted entities
    """
    try:
        if not entity_chain:
            raise HTTPException(status_code=500, detail="Server not properly initialized")
        
        entities = entity_chain.invoke({"question": text})
        return {"entities": entities.names}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting entities: {str(e)}")


@app.get("/graph_search", operation_id="search_graph")
async def search_graph(entity: str):
    """
    Search the graph for relationships around a specific entity
    
    Args:
        entity: The entity to search for
        
    Returns:
        Graph relationships data
    """
    try:
        if not graph:
            raise HTTPException(status_code=500, detail="Server not properly initialized")
        
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit: 5})
            YIELD node, score
            CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 20
            """,
            {"query": entity},
        )
        
        return {"entity": entity, "relationships": [el['output'] for el in response]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching graph: {str(e)}")


mcp = FastApiMCP(
  app,
  include_operations=[
    "search_graph",
    "vector_search",
  ]
)

mcp.mount()



if __name__ == "__main__":

  uvicorn.run(
        "graphrag_mcp_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
  )
