from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Category(Base):
    __tablename__ = "categories"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    knowledge_items = relationship("KnowledgeItem", back_populates="category")

class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    type = Column(String)  # definition, architecture, challenges, etc.
    source = Column(String)  # research_paper, technical_doc, etc.
    difficulty = Column(String)  # beginner, intermediate, advanced
    category_id = Column(String, ForeignKey("categories.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    category = relationship("Category", back_populates="knowledge_items")

class UserProgress(Base):
    __tablename__ = "user_progress"
    
    id = Column(Integer, primary_key=True)
    question_id = Column(String)
    category_id = Column(String, ForeignKey("categories.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    type = Column(String)  # question, feedback
    content = Column(Text)  # JSON string of question or feedback data 