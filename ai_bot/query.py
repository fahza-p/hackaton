from llama_index.indices.query.base import BaseQueryEngine
from llama_index import (
    StorageContext,
    load_index_from_storage
)
from llama_index.vector_stores.simple import SimpleVectorStore
from llama_index.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.storage.index_store.simple_index_store import SimpleIndexStore


class Query():
    dir_path: str = "index"
    query_engine: BaseQueryEngine

    def __init__(self, dir_path: str = "") -> None:
        if dir_path != "":
            self.dir_path = dir_path

    def setup_query(self, sys_prompt, train_chat, chat_histories):
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(
                persist_dir="index"),
            vector_store=SimpleVectorStore.from_persist_dir(
                persist_dir="index"),
            index_store=SimpleIndexStore.from_persist_dir(
                persist_dir="index"),
        )
        storage_context = StorageContext.from_defaults(
            persist_dir=self.dir_path)

        index = load_index_from_storage(storage_context)

        query_engine = index.as_chat_engine(
            chat_mode="context",
            verbose=True,
            chat_history=chat_histories,
            # system_prompt=sys_prompt,
            prefix_messages=train_chat
        )

        self.query_engine = query_engine

    def generate_chat(self, prompt: str):
        response = self.query_engine.chat(
            prompt
        )
        return response.response
