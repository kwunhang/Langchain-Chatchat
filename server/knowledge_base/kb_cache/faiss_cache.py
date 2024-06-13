from configs import CACHED_VS_NUM, CACHED_MEMO_VS_NUM, logger
from server.knowledge_base.kb_cache.base import *
from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter
from server.utils import load_local_embeddings
from server.knowledge_base.utils import get_vs_path
from server.knowledge_base.kb_func.faiss_func import Custom_FAISS
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import os
from langchain.schema import Document


# patch FAISS to include doc id in Document.metadata
def _new_ds_search(self, search: str) -> Union[str, Document]:
    if search not in self._dict:
        return f"ID {search} not found."
    else:
        doc = self._dict[search]
        if isinstance(doc, Document):
            doc.metadata["id"] = search
        return doc
InMemoryDocstore.search = _new_ds_search

class SupportedIndexing:
    INDEXHNSWFLAT = 'IndexHNSWFlat'
    INDEXIVFPQ = 'IndexIVFPQ'
    # INDEXIVFSCALARQUANTIZER = 'IndexIVFScalarQuantizer'
    # INDEXIVFPQ = 'IndexIVFPQ'
    

class ThreadSafeFaiss(ThreadSafeObject):
    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}, docs_count: {self.docs_count()}>"

    def docs_count(self) -> int:
        return len(self._obj.docstore._dict)

    def save(self, path: str, create_path: bool = True):
        with self.acquire():
            if not os.path.isdir(path) and create_path:
                os.makedirs(path)
            ret = self._obj.save_local(path)
            logger.info(f"已将向量库 {self.key} 保存到磁盘")
        return ret

    def clear(self):
        ret = []
        with self.acquire():
            ids = list(self._obj.docstore._dict.keys())
            if ids:
                ret = self._obj.delete(ids)
                assert len(self._obj.docstore._dict) == 0
            logger.info(f"已将向量库 {self.key} 清空")
        return ret


class _FaissPool(CachePool):
    def new_vector_store(
        self,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
        index: str = None,
    ) -> FAISS:
        # logger.info(f"new vector store")
        embedding = EmbeddingsFunAdapter(embed_model)
        doc = Document(page_content="init", metadata={})
        index = self.get_indexing([doc],embedding, index)
        if index:
            vector_store = Custom_FAISS.from_documents(documents = [doc], embedding = embedding, distance_strategy="METRIC_INNER_PRODUCT", index = index)
        else:
            vector_store = Custom_FAISS.from_documents(documents = [doc], embedding = embedding, distance_strategy="METRIC_INNER_PRODUCT")
        ids = list(vector_store.docstore._dict.keys())
        vector_store.delete(ids)
        return vector_store

    def save_vector_store(self, kb_name: str, path: str=None):
        if cache := self.get(kb_name):
            return cache.save(path)

    def unload_vector_store(self, kb_name: str):
        if cache := self.get(kb_name):
            self.pop(kb_name)
            logger.info(f"成功释放向量库：{kb_name}")
    
    def get_indexing(
            self, 
            documents: List[Document],
            embedding: Embeddings, 
            index: str = None
        ) -> Any:
        logger.info(f"get_indexing")
        if index:
            import faiss
            indexing = getattr(SupportedIndexing, index.upper())
            logger.info(f"if index")
            texts = [d.page_content for d in documents]
            embeddings = embedding.embed_documents(texts[0])
            if SupportedIndexing.INDEXHNSWFLAT == indexing:
                index = faiss.index_factory(len(embeddings[0]),"HNSW32,Flat")
                logger.info(f"INDEXHNSWFLAT")
            elif SupportedIndexing.INDEXIVFPQ == indexing:
                index = faiss.index_factory(len(embeddings[0]),"IVF4096,PQ16x8")
                logger.info(f"INDEXIVFPQ")
            else:
                logger.info(f"None")
                index = None
        return index


class KBFaissPool(_FaissPool):
    def load_vector_store(
            self,
            kb_name: str,
            vector_name: str = None,
            create: bool = True,
            embed_model: str = EMBEDDING_MODEL,
            embed_device: str = embedding_device(),
            index: str = None,
    ) -> ThreadSafeFaiss:
        self.atomic.acquire()
        vector_name = vector_name or embed_model
        cache = self.get((kb_name, vector_name)) # 用元组比拼接字符串好一些
        if cache is None:
            item = ThreadSafeFaiss((kb_name, vector_name), pool=self)
            self.set((kb_name, vector_name), item)
            with item.acquire(msg="初始化"):
                self.atomic.release()
                logger.info(f"loading vector store in '{kb_name}/vector_store/{vector_name}' from disk.")
                vs_path = get_vs_path(kb_name, vector_name)

                if os.path.isfile(os.path.join(vs_path, "index.faiss")):
                    embeddings = self.load_kb_embeddings(kb_name=kb_name, embed_device=embed_device, default_embed_model=embed_model)
                    if index:
                        vector_store = FAISS.load_local(vs_path, embeddings, distance_strategy="METRIC_INNER_PRODUCT")
                    else:
                        vector_store = FAISS.load_local(vs_path, embeddings, distance_strategy="METRIC_INNER_PRODUCT")
                elif create:
                    # create an empty vector store
                    if not os.path.exists(vs_path):
                        os.makedirs(vs_path)
                    vector_store = self.new_vector_store(embed_model=embed_model, embed_device=embed_device, index=index)
                    vector_store.save_local(vs_path)
                else:
                    raise RuntimeError(f"knowledge base {kb_name} not exist.")
                item.obj = vector_store
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get((kb_name, vector_name))


class MemoFaissPool(_FaissPool):
    def load_vector_store(
        self,
        kb_name: str,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
        index: str = None,
    ) -> ThreadSafeFaiss:
        self.atomic.acquire()
        cache = self.get(kb_name)
        if cache is None:
            item = ThreadSafeFaiss(kb_name, pool=self)
            self.set(kb_name, item)
            with item.acquire(msg="初始化"):
                self.atomic.release()
                logger.info(f"loading vector store in '{kb_name}' to memory.")
                # create an empty vector store
                vector_store = self.new_vector_store(embed_model=embed_model, embed_device=embed_device, index=index)
                item.obj = vector_store
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get(kb_name)


kb_faiss_pool = KBFaissPool(cache_num=CACHED_VS_NUM)
memo_faiss_pool = MemoFaissPool(cache_num=CACHED_MEMO_VS_NUM)


if __name__ == "__main__":
    import time, random
    from pprint import pprint

    kb_names = ["vs1", "vs2", "vs3"]
    # for name in kb_names:
    #     memo_faiss_pool.load_vector_store(name)

    def worker(vs_name: str, name: str):
        vs_name = "samples"
        time.sleep(random.randint(1, 5))
        embeddings = load_local_embeddings()
        r = random.randint(1, 3)

        with kb_faiss_pool.load_vector_store(vs_name).acquire(name) as vs:
            if r == 1: # add docs
                ids = vs.add_texts([f"text added by {name}"], embeddings=embeddings)
                pprint(ids)
            elif r == 2: # search docs
                docs = vs.similarity_search_with_score(f"{name}", k=3, score_threshold=1.0)
                pprint(docs)
        if r == 3: # delete docs
            logger.warning(f"清除 {vs_name} by {name}")
            kb_faiss_pool.get(vs_name).clear()

    threads = []
    for n in range(1, 30):
        t = threading.Thread(target=worker,
                             kwargs={"vs_name": random.choice(kb_names), "name": f"worker {n}"},
                             daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
