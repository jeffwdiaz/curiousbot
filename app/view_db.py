import sqlite3
from pathlib import Path

def view_database():
    # Get the database path
    db_path = Path("data/curiousbot.db")
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("\n=== Database Tables ===")
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        print("-" * 50)
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        print("Columns:", ", ".join(column_names))
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"Total rows: {count}")
        
        # Get sample data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
        rows = cursor.fetchall()
        if rows:
            print("\nSample data:")
            for row in rows:
                print(row)
    
    conn.close()

if __name__ == "__main__":
    view_database() 