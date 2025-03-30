import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure API key
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

try:
    # Initialize the model
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Simple test prompt
    response = model.generate_content("Say 'Hello! The API is working!'")
    
    print("API Key Status: Valid")
    print("Model: gemini-1.5-pro")
    print("Response:", response.text)
    
except Exception as e:
    print("API Key Status: Invalid")
    print("Error:", str(e)) 