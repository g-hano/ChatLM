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
from configs import EMBEDDING_NAME
import time

def parse_json_stream(line):
    decoded_line = line.decode('utf-8')
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(decoded_line):
        try:
            result, json_end = decoder.raw_decode(decoded_line[pos:])
            if "text" in result:
                #print(result["text"]) # debugging
                if result["text"]:
                   yield result["text"][0]
                   time.sleep(0.1)
            pos += json_end
        except json.JSONDecodeError:
            # if can't decode JSON, go next character
            pos += 1
            
if __name__ == '__main__':
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config['UPLOAD_FOLDER'] = 'uploads'

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    embedding = HuggingFaceEmbedding(
        model_name=EMBEDDING_NAME,
        device="cuda:2", # on third GPU
        trust_remote_code=True,
    )
    Settings.embed_model = embedding # Llama-index will use our custom embed model for retrieval
    
    @app.route('/', methods=['GET', 'POST'])
    def home():
        if request.method == 'POST':
           return upload_file()
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        # if file not provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # if file provided
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(f"file saved {file_path}")
            question = request.form.get('question', '')
            
        def generate():
            try:
               # we can use prompt_len to slice down the text, not using rn
               prompt_len, response = process_and_respond(file_path, question)
               print("********* Generate Funct **********")
               cumulative_text = ""
               for line in response.iter_lines():
                   if line:
                      generator = parse_json_stream(line)
                      for parsed_text in generator:
                          if parsed_text:
                             new_text = parsed_text[len(cumulative_text):]
                             if new_text:
                                yield new_text
                             cumulative_text = parsed_text
            finally:
                print(f"Deleting File {file_path}")
                os.remove(file_path)


        return Response(stream_with_context(generate()), content_type="text/plain")
    app.run(host='0.0.0.0', port=5000, debug=True)

