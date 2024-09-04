import logging
logging.basicConfig(level=logging.INFO)
from configs import SYSTEM_PROMPT, TEMPERATURE

class ChatEngine:
    def __init__(self, retriever):
        """
        Initializes the ChatEngine with a retriever and a language model.

        Args:
            retriever (HybridRetriever): An instance of a retriever to fetch relevant documents.
        """

        self.retriever = retriever
        
    def ask_question(self, question):
        """
        Asks a question to the language model, using the retriever to fetch relevant documents.

        Args:
            question (str): The question to be asked.

        Returns:
            str: The response from the language model in markdown format.
        """

        question = "[INST]" + question + "[/INST]"

        results = self.retriever.best_docs(question)
        document = [doc.text for doc, sc in results]
        logging.info(f"Created Document - len docs:{len(document)}")
        chat_history = SYSTEM_PROMPT + "\n\n" + f"Question: {question}\n\nDocument: {document}"
        logging.info("Created Chat History")
        return chat_history
