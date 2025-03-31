# AI Study Buddy - LLM Learning Assistant

An intelligent study application focused on teaching users about Language Models (LLMs) through interactive quizzes and personalized learning experiences.

## Features

- Real-time quiz generation using Google's Gemini API
- Intelligent explanations with context from a knowledge base
- Personalized learning experience based on user progress
- Five specialized categories:
  - RAG Systems
  - Vector Databases
  - LLM Fine-tuning
  - Verification Tools
  - Orchestration

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
4. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## Usage

1. Access the application at `http://localhost:8000`
2. Select a category to start learning
3. Answer questions and receive detailed explanations
4. Track your progress and get personalized content

## Architecture

The application uses:
- FastAPI for the backend
- Google's Gemini API for question generation
- ChromaDB for vector storage and retrieval
- RAG for enhanced explanations
- Modern async/await patterns for efficient processing 