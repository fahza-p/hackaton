from flask import (Flask, request)
from ai_bot import ConstructIndex, GetPrompt, Query
# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

app = Flask(__name__)
app.config["DEBUG"] = True

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

    chat_histories = prompt_setup.append_history(body.get("history"))
    sys_prompt, train_chat = prompt_setup.text_qa_tmpl()

    query.setup_query(sys_prompt=sys_prompt,
                      train_chat=train_chat, chat_histories=chat_histories)

    return query.generate_chat(body.get("prompt"))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, threaded=True)
