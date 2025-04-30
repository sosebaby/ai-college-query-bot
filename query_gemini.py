import os
import faiss  
import numpy as np
import pyttsx3
import speech_recognition as sr
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer

DATA_DIR = '/Users/eseoseebalu/Desktop/data'

for filename in os.listdir(DATA_DIR):
    filepath = os.path.join(DATA_DIR, filename)
    print(f"Found file: {filepath}")

model = SentenceTransformer("all-MiniLM-L6-v2")

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak your question about Durham College...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I didn't catch that."

def generate_prompt(user_query):
    return f"""You are an assistant trained to answer questions specifically about Durham College. 
Use only the data from the documents provided.

User query: {user_query}

Answer:"""


def read_file(filepath):
    ext = filepath.split(".")[-1].lower()
    text = ""
    try:
        if ext not in ["pdf", "docx", "csv", "json", "txt", "md"]:
            print(f"Unsupported file type: {ext}")
            return ""
        if ext == "pdf":
            reader = PdfReader(filepath)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == "docx":
            doc = Document(filepath)
            text = " ".join([para.text for para in doc.paragraphs])
        elif ext == "csv":
            
            try:
                df = df = pd.read_csv(filepath, on_bad_lines='skip')  
                text = df.to_string()
            except Exception as e:
                print(f"Error reading CSV file {filepath}: {e}")
        elif ext == "json":
            df = pd.read_json(filepath)
            text = df.to_string()
        elif ext in ["txt", "md"]:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return text

def index_documents(folder_path):
    texts, filepaths = [], []

    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        content = read_file(full_path)
        if content:
            texts.append(content)
            filepaths.append(file)

    embeddings = model.encode(texts)
    embeddings = np.array(embeddings)  
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, texts, filepaths


def search(query, index, texts, filepaths, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in indices[0]:
        results.append(texts[i][:1000]) 
    return results

def main():
    print("📚 Indexing Durham College files...")
    index, texts, files = index_documents(DATA_DIR)  
    user_query = speech_to_text()
    print("🗣️ You asked:", user_query)

    if user_query.lower().startswith("sorry") or not user_query.strip():
        text_to_speech("Sorry, I did not understand your question.")
        return

    prompt = generate_prompt(user_query)
    print("\n🔍 Searching for the best answer...")

    results = search(prompt, index, texts, files)
    answer = results[0] if results else "Sorry, no information found."

    print("\n✅ Answer:")
    print(answer)
    text_to_speech(answer)

if __name__ == "__main__":
    main()
