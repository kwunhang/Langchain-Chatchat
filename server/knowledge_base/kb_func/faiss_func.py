from langchain_community.vectorstores.faiss import FAISS, dependable_faiss_import
from typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
    
VST = TypeVar("VST", bound="FAISS")


class Custom_FAISS(FAISS):
    @classmethod
    def from_documents(
        self,
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> FAISS:
        """Return VectorStore initialized from documents and embeddings."""
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        embeddings = embedding.embed_documents(texts)
        return self.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            **kwargs,
        )
    
    @classmethod
    def __from(
        cls,
        texts: Iterable[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[Iterable[dict]] = None,
        ids: Optional[List[str]] = None,
        normalize_L2: bool = False,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        index: Any = None,
        **kwargs: Any,
    ) -> FAISS:
        faiss = dependable_faiss_import()
        if 'index' in kwargs:
            index = kwargs.get('index')
        elif distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            index = faiss.IndexFlatIP(len(embeddings[0]))
        else:
            # Default to L2, currently other metric types not initialized.
            index = faiss.IndexFlatL2(len(embeddings[0]))
        docstore = kwargs.pop("docstore", InMemoryDocstore())
        index_to_docstore_id = kwargs.pop("index_to_docstore_id", {})
        vecstore = cls(
            embedding,
            index,
            docstore,
            index_to_docstore_id,
            normalize_L2=normalize_L2,
            distance_strategy=distance_strategy,
            **kwargs,
        )
        vecstore._FAISS__add(texts, embeddings, metadatas=metadatas, ids=ids)
        return vecstore