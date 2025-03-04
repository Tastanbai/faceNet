import faiss
import numpy as np

class FaissIndexWrapper:
    def __init__(self):
        self.index = None  # Faiss индекс
        self.metadata = []  # Список кортежей (patient_id, hospital_id)

    def build_index(self, embeddings_list):
        """
        embeddings_list: список кортежей (patient_id, hospital_id, embedding, emb_path)
        """
        if not embeddings_list:
            self.index = None
            self.metadata = []
            return

        normalized_embeddings = []
        self.metadata = []
        for patient_id, hospital_id, embedding, emb_path in embeddings_list:
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue
            normalized_embeddings.append(embedding / norm)
            self.metadata.append((patient_id, hospital_id))
        if normalized_embeddings:
            emb_np = np.vstack(normalized_embeddings).astype('float32')
            self.index = faiss.IndexFlatIP(emb_np.shape[1])
            self.index.add(emb_np)

    def add_embedding(self, patient_id: str, hospital_id: str, embedding: np.ndarray):
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return
        normalized = (embedding / norm).astype('float32')
        normalized = normalized.reshape(1, -1)
        if self.index is None:
            self.index = faiss.IndexFlatIP(normalized.shape[1])
        self.index.add(normalized)
        self.metadata.append((patient_id, hospital_id))

    def search(self, query_embedding: np.ndarray, k: int = 1):
        norm = np.linalg.norm(query_embedding)
        if norm == 0:
            raise ValueError("Ошибка нормализации эмбеддинга.")
        query_normalized = (query_embedding / norm).astype('float32')
        query_normalized = query_normalized.reshape(1, -1)
        distances, indices = self.index.search(query_normalized, k)
        return distances, indices
