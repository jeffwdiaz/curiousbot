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

# Define knowledge base for each category
KNOWLEDGE_BASE = {
    "rag": [
        {
            "content": "RAG combines retrieval and generation to improve LLM responses",
            "type": "definition",
            "source": "technical_doc",
            "difficulty": "beginner"
        },
        {
            "content": "RAG helps reduce hallucinations by grounding responses in retrieved context",
            "type": "benefit",
            "source": "technical_doc",
            "difficulty": "intermediate"
        },
        {
            "content": "RAG systems typically use vector databases for efficient retrieval",
            "type": "implementation",
            "source": "technical_doc",
            "difficulty": "advanced"
        }
    ],
    "vector_db": [
        {
            "content": "Vector databases store and search through embeddings efficiently",
            "type": "definition",
            "source": "technical_doc",
            "difficulty": "beginner"
        },
        {
            "content": "Embeddings are numerical representations of text or other data",
            "type": "concept",
            "source": "technical_doc",
            "difficulty": "intermediate"
        },
        {
            "content": "Similarity search is a key feature of vector databases",
            "type": "feature",
            "source": "technical_doc",
            "difficulty": "advanced"
        }
    ],
    "fine_tuning": [
        {
            "content": "Fine-tuning adapts pre-trained models to specific tasks",
            "type": "definition",
            "source": "technical_doc",
            "difficulty": "beginner"
        },
        {
            "content": "LoRA is a popular fine-tuning technique that reduces resource requirements",
            "type": "technique",
            "source": "technical_doc",
            "difficulty": "intermediate"
        },
        {
            "content": "Fine-tuning requires careful dataset preparation and validation",
            "type": "best_practice",
            "source": "technical_doc",
            "difficulty": "advanced"
        }
    ],
    "verification": [
        {
            "content": "Verification tools help ensure LLM outputs are accurate and safe",
            "type": "definition",
            "source": "technical_doc",
            "difficulty": "beginner"
        },
        {
            "content": "A/B testing is common in LLM validation",
            "type": "technique",
            "source": "technical_doc",
            "difficulty": "intermediate"
        },
        {
            "content": "Human-in-the-loop validation is important for critical applications",
            "type": "best_practice",
            "source": "technical_doc",
            "difficulty": "advanced"
        }
    ],
    "orchestration": [
        {
            "content": "Orchestration manages complex LLM workflows and interactions",
            "type": "definition",
            "source": "technical_doc",
            "difficulty": "beginner"
        },
        {
            "content": "Task queues help manage LLM system complexity",
            "type": "implementation",
            "source": "technical_doc",
            "difficulty": "intermediate"
        },
        {
            "content": "Orchestration systems often include monitoring and logging",
            "type": "feature",
            "source": "technical_doc",
            "difficulty": "advanced"
        }
    ]
} 