from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from config.ai_models import get_model

class ChatBot:
    def __init__(self, vector_db, model_name):
        self.db = vector_db
        model_config = get_model(model_name)
        
        if model_name == "OpenAI":
            self.llm = ChatOpenAI(**model_config)
        else:
            self.llm = ChatOllama(**model_config)
        
        self.prompt_template = """
            You are an AI assistant tasked with answering questions based solely
            on the provided context. Your goal is to generate a comprehensive ans
            for the given question using only the information available in the context.

            context: {context}
            question: {question}

            <response> Your answer in Markdown format. </response>
        """

        self.chain = self.build_chain()
    
    def build_chain(self):
        prompt = PromptTemplate(template = self.prompt_template, 
                                input_variables = ["context", "question"])
        retriever = self.db.as_retriever(search_kwargs = {"k": 5})

        chain = RetrievalQA.from_chain_type(
            llm = self.llm,
            chain_type = "stuff",
            retriever = retriever,
            return_source_documents = True,
            chain_type_kwargs = {"prompt": prompt},
            verbose = True
        )

        return chain

    def qa(self, question):        
        response = self.chain.invoke(question)
        return response["result"]
    