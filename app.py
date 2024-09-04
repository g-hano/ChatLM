from flask import Flask, request, render_template, jsonify
from flask import stream_with_context, Response, session
from werkzeug.utils import secure_filename
import os
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from core import (parse_json_stream,
                   get_hybrid_retriever, 
                   respond)
from configs import EMBEDDING_NAME

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import logging
logging.basicConfig(level=logging.INFO)


state = {}

import uuid
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = os.urandom(24)
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
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    user_id = session["user_id"]
    
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
    os.makedirs(user_folder, exist_ok=True)
    file_path = os.path.join(user_folder, filename)

    # Check if the file path is the same as the last uploaded file
    if "file_path" in session and session["file_path"] == file_path:
        print(f"File {filename} is already uploaded.")
        return jsonify({'message': f'File {filename} is already uploaded'}), 200

    # If we reach here, the file is new or different

    # Delete old file if it exists
    if "file_path" in session:
        old_file_path = session["file_path"]
        if os.path.exists(old_file_path):
            os.remove(old_file_path)
            print(f"Previous file {old_file_path} is removed.")

    # Save new file
    file.save(file_path)
    print(f"New file saved: {file_path}")

    # Update session with new file path
    session["file_path"] = file_path

    return jsonify({'message': f'File {filename} uploaded successfully'}), 200

@app.route("/generate", methods=["POST"])
def generate_response():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({'error': 'No active session. Please upload a file first.'}), 400

    question = request.form.get('question', '')
    file_path = session.get("file_path")

    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'No file available for processing. Please upload a file first.'}), 400

    hb_retriever = get_hybrid_retriever(file_path)

    def generate():
        response = respond(hb_retriever, question)
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


    return Response(stream_with_context(generate()), content_type="text/plain")
app.run(host='0.0.0.0', port=5000, debug=True)
