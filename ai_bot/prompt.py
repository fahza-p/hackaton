from llama_index.llms.base import (ChatMessage, MessageRole)
import json


class GetPrompt():
    train_template_path = "template/index.json"
    train_template: list[dict[str, str]]

    def __init__(self, train_template_path: str = "") -> None:
        if train_template_path != "":
            self.train_template_path = train_template_path

    def load_template(self):
        file = open(self.train_template_path)
        self.train_template = json.load(file)

    def text_qa_tmpl(self):
        sys_prompt = "Develop a system to analyze legal documents and provide a comprehensive legal checker tool. The system should be capable of reviewing and verifying various legal documents, contracts, and agreements. It should offer an automated process to identify potential issues, inconsistencies, and discrepancies within the documents"

        train_chat: list[ChatMessage] = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=sys_prompt
            ),
        ]

        for item in self.train_template:
            train_chat.extend((
                ChatMessage(
                    role=MessageRole.USER,
                    content=item["context_str"]
                ),
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=item["query_str"]
                )
            ))

        return sys_prompt, train_chat

    def append_history(self, histories: list[dict[str, str]]):
        chat_histories: list[ChatMessage] = []
        for history in histories:
            chat_histories.extend((
                ChatMessage(
                    role=MessageRole.USER,
                    content=history["context_str"]
                ),
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=history["query_str"]
                )
            ))

        return chat_histories
