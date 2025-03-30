from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import os
import chromadb
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Study Buddy")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-pro')

# Initialize ChromaDB
chroma_client = chromadb.Client()

# Define categories and their knowledge base
CATEGORIES = [
    {
        "id": "rag",
        "name": "RAG Systems",
        "description": "Questions about Retrieval-Augmented Generation systems",
        "knowledge": [
            "RAG combines retrieval and generation to improve LLM responses",
            "RAG helps reduce hallucinations by grounding responses in retrieved context",
            "RAG systems typically use vector databases for efficient retrieval"
        ]
    },
    {
        "id": "vector_db",
        "name": "Vector Databases",
        "description": "Questions about vector databases and embeddings",
        "knowledge": [
            "Vector databases store and search through embeddings efficiently",
            "Embeddings are numerical representations of text or other data",
            "Similarity search is a key feature of vector databases"
        ]
    },
    {
        "id": "fine_tuning",
        "name": "LLM Fine-tuning",
        "description": "Questions about language model fine-tuning",
        "knowledge": [
            "Fine-tuning adapts pre-trained models to specific tasks",
            "LoRA is a popular fine-tuning technique that reduces resource requirements",
            "Fine-tuning requires careful dataset preparation and validation"
        ]
    },
    {
        "id": "verification",
        "name": "Verification Tools",
        "description": "Questions about LLM verification and validation",
        "knowledge": [
            "Verification tools help ensure LLM outputs are accurate and safe",
            "A/B testing is common in LLM validation",
            "Human-in-the-loop validation is important for critical applications"
        ]
    },
    {
        "id": "orchestration",
        "name": "Orchestration",
        "description": "Questions about LLM system orchestration",
        "knowledge": [
            "Orchestration manages complex LLM workflows and interactions",
            "Task queues help manage LLM system complexity",
            "Orchestration systems often include monitoring and logging"
        ]
    }
]

# Get or create collections
try:
    knowledge_collection = chroma_client.create_collection(name="knowledge_base")
    print("Created new knowledge_base collection")
except ValueError:
    knowledge_collection = chroma_client.get_collection(name="knowledge_base")
    print("Retrieved existing knowledge_base collection")

try:
    user_progress_collection = chroma_client.create_collection(name="user_progress")
    print("Created new user_progress collection")
except ValueError:
    user_progress_collection = chroma_client.get_collection(name="user_progress")
    print("Retrieved existing user_progress collection")

# Rich knowledge base content
KNOWLEDGE_BASE = {
    "rag": [
        {
            "content": "RAG (Retrieval-Augmented Generation) is an AI framework that enhances LLM responses by combining information retrieval with text generation",
            "type": "definition",
            "source": "research_paper",
            "difficulty": "beginner"
        },
        {
            "content": "The RAG architecture consists of three main components: a retriever that finds relevant documents, a generator that produces text, and a knowledge base that stores information",
            "type": "architecture",
            "source": "technical_doc",
            "difficulty": "intermediate"
        },
        {
            "content": "Common challenges in RAG systems include context window limitations, retrieval accuracy, and maintaining up-to-date knowledge bases",
            "type": "challenges",
            "source": "practical_guide",
            "difficulty": "advanced"
        },
        {
            "content": "RAG systems can be evaluated using metrics like retrieval precision, generation quality, and factual accuracy",
            "type": "evaluation",
            "source": "best_practices",
            "difficulty": "intermediate"
        },
        {
            "content": "Implementing RAG requires careful consideration of chunking strategies, embedding models, and retrieval mechanisms",
            "type": "implementation",
            "source": "technical_guide",
            "difficulty": "advanced"
        }
    ],
    "vector_db": [
        {
            "content": "Vector databases are specialized systems designed to store and efficiently search through high-dimensional vector embeddings",
            "type": "definition",
            "source": "documentation",
            "difficulty": "beginner"
        },
        {
            "content": "HNSW (Hierarchical Navigable Small World) is a graph-based indexing algorithm that enables fast approximate nearest neighbor search in vector databases",
            "type": "algorithm",
            "source": "research_paper",
            "difficulty": "advanced"
        },
        {
            "content": "Vector similarity can be measured using metrics like cosine similarity, euclidean distance, or dot product",
            "type": "concept",
            "source": "textbook",
            "difficulty": "intermediate"
        },
        {
            "content": "Scaling vector databases requires consideration of memory usage, CPU performance, and storage requirements",
            "type": "operations",
            "source": "best_practices",
            "difficulty": "advanced"
        },
        {
            "content": "Vector databases often support hybrid search combining traditional filters with vector similarity",
            "type": "feature",
            "source": "documentation",
            "difficulty": "intermediate"
        }
    ],
    # Add more categories as needed
}

def load_knowledge_base():
    """Load the knowledge base into ChromaDB"""
    global knowledge_collection
    print("Loading knowledge base into ChromaDB...")
    
    # Clear existing collection to avoid duplicates
    try:
        knowledge_collection.delete_collection()
        knowledge_collection = chroma_client.create_collection(name="knowledge_base")
        print("Cleared and recreated knowledge_base collection")
    except Exception as e:
        print(f"Error clearing collection: {e}")
    
    # Load from KNOWLEDGE_BASE dictionary
    for category, items in KNOWLEDGE_BASE.items():
        for idx, item in enumerate(items):
            entry_id = f"{category}-{datetime.now().timestamp()}-{idx}"
            try:
                knowledge_collection.add(
                    documents=[item["content"]],
                    metadatas=[{
                        "category": category,
                        "type": item["type"],
                        "source": item["source"],
                        "difficulty": item["difficulty"]
                    }],
                    ids=[entry_id]
                )
                print(f"Added {item['type']} content for category: {category}")
            except Exception as e:
                print(f"Error adding entry {entry_id}: {e}")
    
    # Load from CATEGORIES
    for category in CATEGORIES:
        for idx, knowledge in enumerate(category["knowledge"]):
            entry_id = f"cat-{category['id']}-{idx}"
            try:
                knowledge_collection.add(
                    documents=[knowledge],
                    metadatas=[{
                        "category": category["id"],
                        "type": "knowledge",
                        "source": "categories",
                        "difficulty": "intermediate"  # Default difficulty for category knowledge
                    }],
                    ids=[entry_id]
                )
                print(f"Added category knowledge for: {category['id']}")
            except Exception as e:
                print(f"Error adding category entry {entry_id}: {e}")

# Load knowledge base at startup
load_knowledge_base()
print("Knowledge base loaded successfully!")

class QuestionResponse(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    category_id: str

@app.get("/")
async def read_root():
    """Serve the main application page."""
    return FileResponse("app/static/index.html")

@app.get("/categories")
async def get_categories():
    """Get all available quiz categories."""
    return {"categories": CATEGORIES}

@app.post("/quiz/{category_id}")
async def generate_question(category_id: str):
    """Generate a new question for the specified category."""
    print(f"Generating question for category: {category_id}")
    
    # Validate category
    category = next((cat for cat in CATEGORIES if cat["id"] == category_id), None)
    if not category:
        print(f"Invalid category ID: {category_id}")
        raise HTTPException(status_code=400, detail=f"Invalid category ID: {category_id}")

    # Retrieve relevant knowledge
    try:
        results = knowledge_collection.query(
            query_texts=[f"Generate a question about {category['name']}"],
            where={"category": category_id},
            n_results=3
        )
        print(f"Retrieved {len(results['documents'][0])} relevant documents")
        
        if not results['documents'][0]:
            print("No documents found for category")
            raise HTTPException(status_code=404, detail="No knowledge base content found for this category")
            
        if not results['metadatas'][0]:
            print("No metadata found for documents")
            raise HTTPException(status_code=500, detail="Invalid knowledge base state: missing metadata")
            
    except Exception as e:
        print(f"Error querying knowledge base: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve knowledge base content")

    # Prepare context from retrieved knowledge
    context = "\n".join(results['documents'][0])
    metadata = results['metadatas'][0][0]  # Get first metadata entry
    print(f"Context length: {len(context)} characters")
    print(f"Using metadata: {metadata}")

    # Generate question using Gemini
    prompt = f"""You are a quiz generator for teaching about {category['name']}. 
    Using this context: {context}
    
    The content type is {metadata.get('type', 'general')} and the difficulty level is {metadata.get('difficulty', 'intermediate')}.
    Generate a {metadata.get('difficulty', 'intermediate')}-level multiple choice question that tests understanding of {category['name']},
    focusing on {metadata.get('type', 'general')} aspects.
    
    Format your response as a valid JSON object with these exact fields:
    - question: the question text
    - options: array of 4 options prefixed with A), B), C), D)
    - correct_answer: just the letter (A, B, C, or D)
    - explanation: brief explanation of the correct answer
    
    Keep the response concise and ensure it's valid JSON."""

    try:
        print("Generating question with Gemini...")
        response = await model.generate_content_async(prompt)
        print("Raw response:", response.text)
        
        # Clean the response text to ensure valid JSON
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        print("Cleaned response:", cleaned_text)
        question_data = json.loads(cleaned_text)
        
        # Validate the response format
        required_fields = ["question", "options", "correct_answer", "explanation"]
        if not all(field in question_data for field in required_fields):
            missing_fields = [field for field in required_fields if field not in question_data]
            print(f"Missing required fields: {missing_fields}")
            raise ValueError(f"Response missing required fields: {missing_fields}")
        if len(question_data["options"]) != 4:
            print(f"Invalid number of options: {len(question_data['options'])}")
            raise ValueError("Response must have exactly 4 options")
            
        # Store question in user progress collection
        try:
            user_progress_collection.add(
                documents=[json.dumps(question_data)],
                metadatas=[{
                    "category": category_id,
                    "timestamp": datetime.now().isoformat(),
                    "type": "question"
                }],
                ids=[f"q-{datetime.now().timestamp()}"]
            )
            print("Question stored in user progress collection")
        except Exception as e:
            print(f"Error storing question in user progress: {e}")
            # Continue even if storage fails
        
        return question_data
    except json.JSONDecodeError as je:
        print(f"JSON Decode Error: {je}")
        print("Response text:", response.text)
        raise HTTPException(status_code=500, detail="Failed to generate a valid question. Please try again.")
    except Exception as e:
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the question: {str(e)}")

@app.post("/feedback/{question_id}")
async def submit_feedback(question_id: str, feedback: dict):
    """Submit feedback for a question."""
    try:
        user_progress_collection.add(
            documents=[json.dumps(feedback)],
            metadatas=[{
                "question_id": question_id,
                "timestamp": datetime.now().isoformat(),
                "type": "feedback"
            }],
            ids=[f"f-{datetime.now().timestamp()}"]
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}") 