from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings


class EmbeddingSearchLayer:
    def __init__(self, out_results, llms):
        self.vector_db = Chroma("langchain_store", llms["embedding"][1])
        self.out_results = out_results

    def search(self, query, input_results):
        all_ids = list(input_results.keys())
        self.vector_db.add_texts(
            texts=[input_results[x]["text"] for x in all_ids],
            metadatas=[{"id": x} for x in all_ids],
            ids=[str(x) for x in all_ids],
        )
        output_documents = self.vector_db.similarity_search_with_relevance_scores(
            query, k=self.out_results
        )
        output_ids = [doc[0].metadata["id"] for doc in output_documents]
        output_results = {x: input_results[x] for x in output_ids}
        self.vector_db.delete([str(x) for x in all_ids])
        return output_results
