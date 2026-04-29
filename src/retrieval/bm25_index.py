"""
BM25 Index - Build and search inverted index for keyword retrieval.

Uses rank-bm25 library for efficient BM25 scoring.
"""

from typing import List, Dict, Any, Optional
import pickle
from pathlib import Path
import re
from rank_bm25 import BM25Okapi

from src.models.chunk import Chunk
from src.storage.chroma_store import ChromaVectorStore

from src.utils.logger import setup_logger
from src.utils.exceptions import AgenticRAGException
from src.utils.retrieval_debug import format_ranked_chunk_line


class BM25IndexError(AgenticRAGException):
    """Error during BM25 index operations."""
    pass


class BM25Index:
    """
    BM25 inverted index for keyword search.
    
    Features:
    - Build index from ChromaDB chunks
    - Fast keyword-based retrieval
    - BM25 scoring (Okapi BM25 variant)
    - Persistent storage
    
    Example:
        >>> # Build index
        >>> index = BM25Index()
        >>> index.build_from_vector_store()
        
        >>> # Search
        >>> results = index.search("python programming", top_k=5)
        >>> for result in results:
        ...     print(result['chunk_id'], result['score'])
    """
    
    def __init__(self, index_path: str = "data/bm25_index.pkl"):
        """
        Initialize BM25 index.
        
        Args:
            index_path: Path to save/load index
        """
        self.logger = setup_logger("bm25_index")
        self.index_path = Path(index_path)
        
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.chunk_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Try to load existing index
        if self.index_path.exists():
            self.load()
    
    def build_from_vector_store(
        self,
        vector_store: Optional[ChromaVectorStore] = None  # ← CHANGE type
    ) -> None:
        """
        Build BM25 index from chunks in vector store.
        
        Args:
            vector_store: ChromaVectorStore instance (creates new if None)
        
        Raises:
            BM25IndexError: If index building fails
        
        Example:
            >>> index = BM25Index()
            >>> index.build_from_vector_store()
            >>> index.save()
        """
        self.logger.info("Building BM25 index from vector store...")
        
        try:
            # Get vector store
            if vector_store is None:
                vector_store = ChromaVectorStore()
            
            collection = vector_store.collection
            total_chunks = collection.count()
            
            if total_chunks == 0:
                raise BM25IndexError(
                    message="No chunks in vector store to index",
                    details={"collection": "chunks"}
                )
            
            self.logger.info(f"Indexing {total_chunks} chunks...")
            
            # Fetch all chunks from the flat collection
            results = collection.get(
                include=["documents", "metadatas"],
                limit=total_chunks
            )
            
            if not results or not results['ids']:
                raise BM25IndexError(
                    message="Failed to fetch chunks from vector store",
                    details={}
                )
            
            # Prepare data for BM25
            documents = []
            chunk_ids = []
            chunk_metadata = {}
            
            for i, chunk_id in enumerate(results['ids']):
                text = results['documents'][i]
                metadata = results['metadatas'][i]
                
                # Tokenize (simple word splitting)
                tokens = self._tokenize(text)
                documents.append(tokens)
                
                chunk_ids.append(chunk_id)
                chunk_metadata[chunk_id] = {
                    'text': text,
                    'metadata': metadata
                }
            
            # Build BM25 index
            self.bm25 = BM25Okapi(documents)
            self.chunk_ids = chunk_ids
            self.chunk_metadata = chunk_metadata
            
            self.logger.info(
                f"✅ Built BM25 index with {len(chunk_ids)} chunks"
            )
            
        except BM25IndexError:
            raise
        except Exception as e:
            raise BM25IndexError(
                message=f"Failed to build BM25 index: {str(e)}",
                details={"error": str(e)}
            ) from e
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        
        try:
            # 对查询文本进行分词，得到 BM25 可用的分词列表
            query_tokens = self._tokenize(query)
            print(f"\nBM25 Query Tokens: {query_tokens}")

            # 计算每个文档块的 BM25 分数
            scores = self.bm25.get_scores(query_tokens)

            # 按分数从高到低排序，并取前 top_k 个索引
            top_indices = scores.argsort()[::-1][:top_k]

          
            results = []

            # 遍历 top_k 文档块索引，构造返回结果
            for idx in top_indices:
                # 获取当前文档块 ID
                chunk_id = self.chunk_ids[idx]

                # 根据 chunk_id 获取文档块内容和元数据
                chunk_data = self.chunk_metadata[chunk_id]

                # 将当前 BM25 分数转换为普通浮点数
                score = float(scores[idx])

                # 只返回分数大于 0 的文档块
                if score > 0:
                    # 复制元数据，避免直接修改原始数据
                    metadata = dict(chunk_data['metadata'] or {})

                    # 标记当前结果来源为关键词检索
                    metadata.setdefault("source", "keyword")

                    # 将当前文档块整理为标准结果格式
                    results.append({
                        'chunk_id': chunk_id,
                        'text': chunk_data['text'],
                        'score': score,
                        'metadata': metadata
                    })

            # 记录 BM25 检索返回的结果数量
            self.logger.debug(f"BM25 search returned {len(results)} results")

            # 打印 BM25 检索概要信息
            print(f"\nBM25 Search (top_k={top_k})")
            print(f"   Found {len(results)} results")

            # 逐条打印排序后的检索结果摘要
            for rank, result in enumerate(results, start=1):
                print(format_ranked_chunk_line(rank, result))

            # 返回检索结果列表
            return results

        # 捕获检索过程中的异常，并封装为 BM25IndexError
        except Exception as e:
            raise BM25IndexError(
                message=f"BM25 search failed: {str(e)}",
                details={"query": query}
            ) from e
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Optimized for English RAG:
        - Lowercase
        - Preserve useful technical terms
        - Split punctuation
        - Remove common stopwords
        - Keep short technical tokens like AI, ML, RAG
        """
        text = text.lower()

        # 把连字符词拆开，同时保留原词的一种近似形式
        # fine-tuning -> fine tuning finetuning
        hyphen_words = re.findall(r'\b[a-z0-9]+(?:-[a-z0-9]+)+\b', text)
        extra_terms = []
        for w in hyphen_words:
            extra_terms.append(w.replace("-", ""))      # finetuning
            extra_terms.extend(w.split("-"))            # fine, tuning

        # 保留字母、数字、下划线，其余符号变空格
        text = re.sub(r"[^\w\s]", " ", text)

        tokens = text.split()

        # 英文停用词，别放太多，BM25 里停用词过多会影响精确匹配
        stopwords = {
            "a", "an", "the",
            "is", "are", "was", "were", "be", "been", "being",
            "of", "in", "on", "at", "to", "for", "from", "by", "with",
            "and", "or", "but",
            "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
            "does", "do", "did",
            "can", "could", "should", "would", "may", "might",
            "this", "that", "these", "those",
            "it", "its", "as", "than", "then",
            "between", "into", "about"
        }

        tokens = [
            t for t in tokens
            if t not in stopwords and len(t) > 1
        ]

        tokens.extend(extra_terms)

        return tokens

    def tokenize_query(self, query: str) -> List[str]:
        """Expose BM25 query tokenization for debugging and tests."""
        return self._tokenize(query)
    
    def save(self) -> None:
        """
        Save BM25 index to disk.
        
        Example:
            >>> index.build_from_vector_store()
            >>> index.save()
        """
        if self.bm25 is None:
            self.logger.warning("No index to save")
            return
        
        try:
            # Create directory if needed
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save index data
            data = {
                'bm25': self.bm25,
                'chunk_ids': self.chunk_ids,
                'chunk_metadata': self.chunk_metadata
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"💾 Saved BM25 index to {self.index_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {str(e)}")
            raise BM25IndexError(
                message=f"Failed to save index: {str(e)}",
                details={"path": str(self.index_path)}
            ) from e
    
    def load(self) -> bool:
        """
        Load BM25 index from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        
        Example:
            >>> index = BM25Index()
            >>> if index.load():
            ...     results = index.search("query")
        """
        if not self.index_path.exists():
            self.logger.debug(f"Index file not found: {self.index_path}")
            return False
        
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
            
            self.bm25 = data['bm25']
            self.chunk_ids = data['chunk_ids']
            self.chunk_metadata = data['chunk_metadata']
            
            self.logger.info(
                f"📂 Loaded BM25 index with {len(self.chunk_ids)} chunks"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index stats
        
        Example:
            >>> stats = index.get_stats()
            >>> print(stats['total_chunks'])
        """
        if self.bm25 is None:
            return {
                'built': False,
                'total_chunks': 0
            }
        
        return {
            'built': True,
            'total_chunks': len(self.chunk_ids),
            'index_path': str(self.index_path),
            'index_exists': self.index_path.exists()
        }
    
    def rebuild(self) -> None:
        """
        Rebuild index from vector store.
        
        Convenience method to rebuild and save index.
        
        Example:
            >>> index = BM25Index()
            >>> index.rebuild()  # Build and save in one step
        """
        self.build_from_vector_store()
        self.save()
        self.logger.info("✅ Index rebuilt and saved")
