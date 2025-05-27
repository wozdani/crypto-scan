import openai
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def test_openai_connection():
    try:
        print("🔌 Sprawdzanie połączenia z OpenAI (nowy interfejs)...")
        models = client.models.list()
        print("✅ Połączenie działa. Dostępne modele:")
        for model in models.data:
            print("  -", model.id)
    except Exception as e:
        print("❌ Błąd połączenia z OpenAI:")
        print(e)

if __name__ == "__main__":
    test_openai_connection()

