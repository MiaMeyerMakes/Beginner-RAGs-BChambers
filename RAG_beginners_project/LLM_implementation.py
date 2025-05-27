import subprocess
import requests
import time
import json
from Corpus_file import corpus_of_documents
from Basic_RAG_implementation import return_response, jaccard_similarity


def is_ollama_running():
    """Check if Ollama server is responding"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False


def start_ollama():
    """Start Ollama server"""
    try:
        # Start Ollama serve in the background
        subprocess.Popen(['ollama', 'serve'],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)

        # Wait a bit for it to start
        time.sleep(3)

        # Check if it's running
        for i in range(10):  # Try for 10 seconds
            if is_ollama_running():
                print("Ollama server started successfully!")
                return True
            time.sleep(1)

        print("Failed to start Ollama server")
        return False
    except Exception as e:
        print(f"Error starting Ollama: {e}")
        return False


def ensure_ollama_running():
    """Ensure Ollama is running before proceeding"""
    if not is_ollama_running():
        print("Ollama not running, starting it...")
        if not start_ollama():
            raise Exception("Could not start Ollama server")
    else:
        print("Ollama is already running!")


# Use it in your main script
if __name__ == "__main__":
    ensure_ollama_running()

    user_input = "I like to hike"
    relevant_document = return_response(user_input, corpus_of_documents)
    full_response = []

    # https://github.com/jmorganca/ollama/blob/main/docs/api.md

    prompt = """
    You are a bot that makes recommendations for activities. You answer in very short sentences and do not include extra information.
    
    This is the recommended activity: {relevant_document}
    
    The user input is: {user_input}
    
    Compile a recommendation to the user based on the recommended activity and the user input.
    """

    url = 'http://localhost:11434/api/generate'
    data = {
        "model": "llama3.1:8b",
        "prompt": prompt.format(user_input=user_input, relevant_document=relevant_document)
    }

    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)

    try:
        count = 0
        for line in response.iter_lines():
            # filter out keep-alive new lines
            # count += 1
            # if count % 5== 0:
            #     print(decoded_line['response']) # print every fifth token
            if line:
                decoded_line = json.loads(line.decode('utf-8'))

                full_response.append(decoded_line['response'])
    finally:
        response.close()
    print(''.join(full_response))