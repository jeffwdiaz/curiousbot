import uvicorn
import os

if __name__ == "__main__":
    # Get the absolute path to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the project root directory
    os.chdir(current_dir)
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",  # This makes the server accessible from any IP
        port=8000,
        reload=True  # Enable auto-reload for development
    ) 