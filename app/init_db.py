from sqlalchemy.orm import Session
from .database import engine, Base
from .models import Category, KnowledgeItem
from .data_definitions import CATEGORIES, KNOWLEDGE_BASE

def init_db():
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create a new session
    db = Session(engine)
    
    try:
        # Add categories if they don't exist
        for category in CATEGORIES:
            existing_category = db.query(Category).filter(Category.id == category["id"]).first()
            if not existing_category:
                db_category = Category(
                    id=category["id"],
                    name=category["name"],
                    description=category["description"]
                )
                db.add(db_category)
                print(f"Added category: {category['name']}")
            else:
                print(f"Category already exists: {category['name']}")
        
        # Add knowledge items if they don't exist
        for category_id, items in KNOWLEDGE_BASE.items():
            for item in items:
                existing_item = db.query(KnowledgeItem).filter(
                    KnowledgeItem.content == item["content"],
                    KnowledgeItem.category_id == category_id
                ).first()
                
                if not existing_item:
                    db_item = KnowledgeItem(
                        content=item["content"],
                        type=item["type"],
                        source=item["source"],
                        difficulty=item["difficulty"],
                        category_id=category_id
                    )
                    db.add(db_item)
                    print(f"Added knowledge item for category: {category_id}")
                else:
                    print(f"Knowledge item already exists for category: {category_id}")
        
        # Commit all changes
        db.commit()
        print("Database initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_db() 