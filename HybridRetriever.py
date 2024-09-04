from llama_index.retrievers.bm25 import BM25Retriever 
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import Document

class HybridRetriever:
    def __init__(self, bm25_retriever: BM25Retriever, vector_retriever: VectorIndexRetriever):
        """
        Inıtializes a Hybrid Retriever with BM25Retriever and VectorIndexRetriever.

        Args:
            bm25_retriever (BM25Retriever): An instance of BM25Retriever for keyword-based retrieval.
            vector_retriever (VectorIndexRetriever): An instance of VectorIndexRetriever for vector-based retrieval.
        """
        
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.top_k = vector_retriever._similarity_top_k + bm25_retriever._similarity_top_k

    def retrieve(self, query: str):
        """
        Retrieves documents relevant to the query using both BM25 and vector retrieval methods.
        
        Args:
            query (str): The query string for which relevant documents are to be retrieved.
        
        Returns:
            list: A list of tuples, each containing the document text and its combined score.
        """
        query = "[INST] " + " [/INST]"
        # Perform keyword search using BM25 retriever
        bm25_results = self.bm25_retriever.retrieve(query)
        
        # Perform vector search using VectorIndexRetriever
        vector_results = self.vector_retriever.retrieve(query)
        
        # Combine results, filter duplicates, and calculate combined scores
        combined_results = {}
        for result in bm25_results:
            combined_results[result.node.text] = {'score': result.score}
        
        for result in vector_results:
            if result.node.text in combined_results:
                combined_results[result.node.text]['score'] += result.score
            else:
                combined_results[result.node.text] = {'score': result.score}

        # Convert combined results to a list of tuples and sort by score
        combined_results_list = sorted(combined_results.items(), key=lambda item: item[1]['score'], reverse=True)
        return combined_results_list # {score, document}
        
    def best_docs(self, query: str):
        """
        Retrieves the most relevant documents to the query as Document objects with their scores.

        Args:
            query (str): The query string for which the most relevant documents are to be retrieved.

        Returns:
            list: A list of tuples, each containing a Document object and its score.
        """
        top_results = self.retrieve(query)
        return [(Document(text=text), score) for text, score in top_results]
