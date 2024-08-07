import csv
import fitz
from docx import Document as DocxDocument
from llama_index.core import Document
import requests
import json

def process_file(file):
    file_extension = file.split(".")[-1].lower()

    if file_extension == 'txt':
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()

    elif file_extension == 'csv':
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            text = '\n'.join(','.join(row) for row in reader)

    elif file_extension == 'pdf':
        pdf_document = fitz.open(file, filetype=file_extension)
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text")
        pdf_document.close()
        
    elif file_extension == 'docx':
        docx_document = DocxDocument(file)
        text = ""
        for paragraph in docx_document.paragraphs:
            text += paragraph.text + "\n"

    return [Document(text=text)]

def send_request_2_llm(prompt: str):
    url = "http://localhost:8000/generate"
    if len(prompt) > 27_000: # model-len is set to 27k on vllm.entrypoints.api_server
        prompt = prompt[:27_000]
    payload = {
        "prompt": prompt,
        "stream": True,
        "min_tokens": 256,
        "max_tokens": 1024
    }
    last_response_len = len(prompt)
    with requests.post(url, json=payload, stream=True) as response:
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        yield chunk.text[last_response_len:]
                        last_response_len = len(chunk.text[last_response:])#json.dumps(chunk) 
                    except json.JSONDecodeError:
                        print(f"Failed to decode JSON: {line.decode('utf-8')}")
        else:
            yield json.dumps({"text":f"Error: Received status code {response.status_code}"})
