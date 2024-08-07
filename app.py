from flask import Flask, request, render_template, jsonify
from flask import stream_with_context, Response, send_from_directory
from werkzeug.utils import secure_filename
import os
import requests
from llama_index.core import Settings
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
import logging
logging.basicConfig(level=logging.INFO)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json
from core import process_and_respond


if __name__ == '__main__':
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config['UPLOAD_FOLDER'] = 'uploads'

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    embedding = HuggingFaceEmbedding(
        model_name=EMBEDDING_NAME,
        device="cuda:2",
        trust_remote_code=True,
        )
    Settings.embed_model = embedding
    
    @app.route('/', methods=['GET', 'POST'])
    def home():
        if request.method == 'POST':
           return upload_file()
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            question = request.form.get('question', '')
            
            def generate():
                system_prompt_length, response_generator = process_and_respond(file_path, question)
                logging.debug(f"System prompt length: {system_prompt_length}")
                last_response = ""
                try:
                   for chunk in response_generator:
                       try:
                          # Attempt to parse the chunk as JSON
                          json_chunk = json.loads(chunk)
                          if "text" in json_chunk:
                             full_text = json_chunk["text"]
                             if len(full_text) > system_prompt_length:
                                new_content = full_text[system_prompt_length:]
                                if len(new_content) > len(last_response):
                                   yield new_content[len(last_response):] + '\n'
                                   last_response = new_content
                          else:
                              logging.debug("Received JSON chunk without 'text' key")
                       except json.JSONDecodeError:
                           # If it's not JSON, process it as before
                           if len(chunk) > system_prompt_length:
                              new_content = chunk[system_prompt_length:]
                              if len(new_content) > len(last_response):
                                 yield new_content[len(last_response):] + '\n'
                                 last_response = new_content
                           else:
                               logging.debug("Chunk not longer than system prompt")
                finally:
                    os.remove(file_path)

            return Response(stream_with_context(generate()), content_type='text/plain')
    app.run(host='0.0.0.0', port=5000)

