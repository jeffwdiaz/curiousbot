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
knowledge_collection = chroma_client.create_collection(name="knowledge_base")
user_progress_collection = chroma_client.create_collection(name="user_progress")

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

# Initialize knowledge base
for category in CATEGORIES:
    for idx, knowledge in enumerate(category["knowledge"]):
        knowledge_collection.add(
            documents=[knowledge],
            metadatas=[{"category": category["id"], "type": "knowledge"}],
            ids=[f"{category['id']}-{idx}"]
        )

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
    # Validate category
    category = next((cat for cat in CATEGORIES if cat["id"] == category_id), None)
    if not category:
        raise HTTPException(status_code=400, detail=f"Invalid category ID: {category_id}")

    # Retrieve relevant knowledge
    results = knowledge_collection.query(
        query_texts=[f"Generate a question about {category['name']}"],
        where={"category": category_id},
        n_results=3
    )

    # Prepare context from retrieved knowledge
    context = "\n".join(results['documents'][0])

    # Generate question using Gemini
    prompt = f"""You are a quiz generator for teaching about {category['name']}. 
    Using this context: {context}
    
    Generate a multiple choice question that tests understanding of {category['name']}.
    
    Format your response as a valid JSON object with these exact fields:
    - question: the question text
    - options: array of 4 options prefixed with A), B), C), D)
    - correct_answer: just the letter (A, B, C, or D)
    - explanation: brief explanation of the correct answer
    
    Keep the response concise and ensure it's valid JSON."""

    try:
        response = await model.generate_content_async(prompt)
        print("Raw response:", response.text)  # Debug print
        
        # Clean the response text to ensure valid JSON
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        question_data = json.loads(cleaned_text)
        
        # Validate the response format
        required_fields = ["question", "options", "correct_answer", "explanation"]
        if not all(field in question_data for field in required_fields):
            raise ValueError("Response missing required fields")
        if len(question_data["options"]) != 4:
            raise ValueError("Response must have exactly 4 options")
            
        # Store question in user progress collection
        user_progress_collection.add(
            documents=[json.dumps(question_data)],
            metadatas=[{
                "category": category_id,
                "timestamp": datetime.now().isoformat(),
                "type": "question"
            }],
            ids=[f"q-{datetime.now().timestamp()}"]
        )
        
        return question_data
    except json.JSONDecodeError as je:
        print("JSON Decode Error. Response text:", response.text)
        raise HTTPException(status_code=500, detail="Failed to generate a valid question. Please try again.")
    except Exception as e:
        print("Error type:", type(e))
        print("Error details:", str(e))
        raise HTTPException(status_code=500, detail="An error occurred while generating the question. Please try again.")

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