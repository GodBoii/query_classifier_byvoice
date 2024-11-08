import os
import threading
import json
import subprocess
import sys
import speech_recognition as sr
import time
import signal
from pathlib import Path
import simple_task
import automation
import adv_automation
import general_query
import realtime
import logging
import warnings
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings('ignore')
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

SCRIPT_DIR = Path(__file__).parent
TEMP_DIR = SCRIPT_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True, parents=True)

class SuppressOutput:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

CATEGORIES = {
    "general": {
        "keywords": ["general information", "basic questions", "simple queries"],
        "priority": 1
    },
    "realtime": {
        "keywords": ["current", "find", "show", "today", "now", "search", "news", 
                    "price", "temperature", "stock", "weather", "traffic"],
        "priority": 2
    },
    "simple": {
        "keywords": ["open", "create", "increase", "decrease", "mute", "empty", 
                    "screenshot", "volume", "app", "desktop", "file", "folder"],
        "priority": 3
    },
    "automation": {
        "keywords": ["schedule", "reminder", "automate", "backup", "set", "daily", 
                    "weekly", "automatic", "alert", "workflow", "monitor"],
        "priority": 4
    },
    "advautomation": {
        "keywords": ["workflow", "report", "transcription", "send", "script", 
                    "monitor", "system", "generate", "monthly", "optimize", "sort", "email"],
        "priority": 5
    }
}

def send_frontend_message(message, message_type="state"):
    if message_type == "output":
        sys.stdout.write(f"OUTPUT: {message}\n")
    elif message_type == "voice":
        sys.stdout.write(f"VOICE: {message}\n")
    else:
        sys.stdout.write(f"{message}\n")
    sys.stdout.flush()

class QueryClassifier:
    def __init__(self):
        with SuppressOutput():
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _keyword_match(self, query):
        query_lower = query.lower()
        matches = []
        
        for category, info in CATEGORIES.items():
            if any(keyword in query_lower for keyword in info["keywords"]):
                matches.append((category, info["priority"]))
        
        if matches:
            return sorted(matches, key=lambda x: x[1], reverse=True)[0][0]
        return None

    def _embedding_classify(self, query):
        examples = {
            "general": ["what is this", "tell me about", "how does this work"],
            "realtime": ["show current weather", "find latest news", "what's the temperature"],
            "simple": ["open application", "create new folder", "change volume"],
            "automation": ["set a reminder", "schedule task", "create alert"],
            "advautomation": ["generate report", "setup workflow", "automate process"]
        }
        
        with SuppressOutput():
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            best_score = -1
            best_category = None
            
            for category, examples in examples.items():
                example_embeddings = self.embedder.encode(examples, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(query_embedding, example_embeddings)
                avg_score = similarity.mean().item()
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_category = category
            
            return best_category if best_score > 0.3 else None

    def classify(self, query):
        category = self._keyword_match(query)
        
        if not category:
            category = self._embedding_classify(query)
        
        if not category:
            category = "general"
        
        return category

def process_text_input(text):
    send_frontend_message(text, "voice")
    
    classifier = QueryClassifier()
    category = classifier.classify(text)
    
    send_frontend_message("PROCESSING")
    success = process_command(category, text)
    
    return success

def capture_voice_input():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            send_frontend_message("LISTENING")
            
            recognizer.dynamic_energy_threshold = True
            recognizer.energy_threshold = 4000
            recognizer.adjust_for_ambient_noise(source, duration=2)
            
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            text = recognizer.recognize_google(audio)
            
            with open(TEMP_DIR / "voice_input.txt", "w", encoding='utf-8') as f:
                f.write(text)
            
            send_frontend_message(text, "voice")
            return text

    except (sr.RequestError, sr.UnknownValueError, Exception) as e:
        send_frontend_message(f"Error capturing voice: {str(e)}", "output")
        return None

def process_command(category, text):
    handlers = {
        "general": "general_query.py",
        "realtime": "realtime.py",
        "simple": "simple_task.py",
        "automation": "automation.py",
        "advautomation": "adv_automation.py"
    }
    
    handler_script = handlers.get(category)
    if not handler_script:
        send_frontend_message(f"Unknown category: {category}", "output")
        return False

    send_frontend_message("PROCESSING")
    
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / handler_script), text],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line.strip():  
                    send_frontend_message(line.strip(), "output")
        if result.stderr:
            send_frontend_message(f"Error: {result.stderr}", "output")
        return result.returncode == 0
    except Exception as e:
        send_frontend_message(f"Error processing command: {str(e)}", "output")
        return False

def main_loop():
    classifier = QueryClassifier()
    
    while True:
        text = capture_voice_input()
        if not text:
            continue
        if text.lower() in ['exit', 'quit', 'stop']:
            send_frontend_message("STOPPING")
            break
        send_frontend_message("THINKING")
        category = classifier.classify(text)
        process_command(category, text)
        send_frontend_message("LISTENING")

def main():
    send_frontend_message("STARTING")

    def signal_handler(signum, frame):
        send_frontend_message("STOPPING")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main_loop()
    send_frontend_message("STOPPED")

if __name__ == "__main__":
    main()