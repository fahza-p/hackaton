from llama_index.llms import OpenAI
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import set_global_service_context
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)


class ConstructIndex():
    from_path: str = ""
    dir_path: str = "index"
    model_name: str = "gpt-3.5-turbo"

    def __init__(self, from_path: str, dir_path: str = "", model_name: str = "") -> None:
        self.from_path = from_path
        if dir_path != "":
            self.dir_path = dir_path
        if model_name != "":
            self.model_name = model_name

    def construct_index(self):
        # Define llm
        llm = OpenAI(
            temperature=1,
            model=self.model_name
        )

        # Load documents
        docs = SimpleDirectoryReader(self.from_path).load_data()

        # Parse the docs into nodes
        parser = SimpleNodeParser(text_splitter=TokenTextSplitter(
            chunk_size=1024, chunk_overlap=20))

        # Create service context
        service_context = ServiceContext.from_defaults(
            llm=llm,
            node_parser=parser
        )

        set_global_service_context(service_context)

        # Build an index
        index = VectorStoreIndex.from_documents(
            docs, service_context=service_context)

        # Store index
        index.storage_context.persist(persist_dir=self.dir_path)
