from flask import (Flask, request)
from ai_bot import ConstructIndex, GetPrompt, Query
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import os
# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

app = Flask(__name__)
UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'pdf'}

app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

index = ConstructIndex(
    from_path="docs",
    model_name="gpt-3.5-turbo"
)
index.construct_index()

prompt_setup = GetPrompt()
prompt_setup.load_template()

query = Query(dir_path="index")


@app.route("/", methods=['POST'])
def main():
    body = request.get_json()

    promptText = body.get("prompt")

    chat_histories = prompt_setup.append_history(body.get("history"))
    sys_prompt, train_chat = prompt_setup.text_qa_tmpl()

    query.setup_query(sys_prompt=sys_prompt,
                      train_chat=train_chat, chat_histories=chat_histories)

    if body.get("fileName") and body.get("fileName") != "":
        promptText = "please review this contract base on knowlage you have, provide response in Indonesian languange:"
        reader = PdfReader('upload/'+body.get("fileName"))
        for page in range(len(reader.pages)):
            text = reader.pages[page].extract_text()
            promptText += text

    return query.generate_chat(promptText)


@app.route("/upload", methods=['POST'])
def upload():
    if 'file' not in request.files:
        return ('No file part')

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return filename


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, threaded=True)
