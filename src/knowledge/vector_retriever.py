"""
向量知識庫 - 使用 FAISS 進行語意檢索
將操作手冊轉換為向量，並支援語意相似度搜尋
"""

import json
import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np


class VectorKnowledgeBase:
    """
    基於向量相似度的知識檢索系統
    使用 Sentence Transformers 進行文本嵌入，FAISS 進行快速檢索
    """
    
    def __init__(
        self, 
        manual_path: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        cache_dir: str = None
    ):
        """
        初始化向量知識庫
        
        Args:
            manual_path: 操作手冊 JSON 路徑
            embedding_model: Sentence Transformer 模型名稱
            cache_dir: 向量快取目錄
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.manual_path = manual_path or os.path.join(self.base_dir, "wafer_maintenance_manual.json")
        self.cache_dir = cache_dir or os.path.join(self.base_dir, ".vector_cache")
        self.embedding_model_name = embedding_model
        
        # 延遲載入的組件
        self._embedding_model = None
        self._index = None
        self._documents = []
        self._metadata = []
        
        # 確保快取目錄存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 嘗試載入或建立索引
        self._initialize()
    
    def _initialize(self):
        """初始化知識庫"""
        cache_file = os.path.join(self.cache_dir, "vector_index.pkl")
        
        if os.path.exists(cache_file):
            # 載入快取
            try:
                self._load_cache(cache_file)
                print(f"✅ 向量知識庫已從快取載入 ({len(self._documents)} 文件)")
                return
            except Exception as e:
                print(f"⚠️ 快取載入失敗: {e}，重新建立索引...")
        
        # 建立新索引
        self._build_index()
        self._save_cache(cache_file)

    def _rebuild_cache(self):
        """使用目前可用的 embedding backend 重建索引快取。"""
        cache_file = os.path.join(self.cache_dir, "vector_index.pkl")
        self._embedding_model = None
        self._index = None
        self._build_index()
        self._save_cache(cache_file)
    
    def _get_embedding_model(self):
        """延遲載入 Embedding 模型"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    local_files_only=True,
                )
                print(f"✅ Embedding 模型已載入: {self.embedding_model_name}")
            except Exception as e:
                print(f"⚠️ Embedding 模型載入失敗 ({e})，使用簡易 TF-IDF 替代")
                self._embedding_model = SimpleTfidfEncoder()
        return self._embedding_model
    
    def _parse_manual(self) -> List[Dict[str, Any]]:
        """
        解析操作手冊 JSON，將每個章節段落轉換為文件
        """
        documents = []
        
        try:
            with open(self.manual_path, 'r', encoding='utf-8') as f:
                manual = json.load(f)
        except FileNotFoundError:
            print(f"⚠️ 找不到操作手冊: {self.manual_path}")
            return documents
        
        # 遍歷所有章節
        for chapter in manual.get("chapters", []):
            chapter_id = chapter.get("chapter_id", "")
            chapter_title = chapter.get("title", "")
            
            for section in chapter.get("sections", []):
                section_id = section.get("section_id", "")
                section_title = section.get("title", "")
                content = section.get("content", "")
                keywords = section.get("keywords", [])
                parameters = section.get("parameters", {})
                priority = section.get("priority", "NORMAL")
                
                # 組合完整文件
                doc = {
                    "id": section_id,
                    "chapter": chapter_title,
                    "title": section_title,
                    "content": content,
                    "keywords": keywords,
                    "parameters": parameters,
                    "priority": priority,
                    # 用於向量化的完整文本
                    "full_text": f"{chapter_title} - {section_title}\n{content}\n關鍵字: {', '.join(keywords)}"
                }
                documents.append(doc)
        
        # [新增] 處理附錄 (Appendix)
        appendix = manual.get("appendix", {})
        if appendix:
            # 1. Common Parameters
            if "common_parameters" in appendix:
                params = appendix["common_parameters"]
                content_str = "\n".join([f"{k}: {', '.join(v)}" for k, v in params.items()])
                documents.append({
                    "id": "APP-01",
                    "chapter": "附錄",
                    "title": "常用參數單位 (Common Parameters)",
                    "content": content_str,
                    "keywords": ["單位", "Unit", "參數", "轉換"],
                    "parameters": {},
                    "priority": "NORMAL",
                    "full_text": f"附錄 - 常用參數單位\n{content_str}\n關鍵字: 單位, Unit"
                })

            # 2. Emergency Contacts
            if "emergency_contacts" in appendix:
                contacts = appendix["emergency_contacts"]
                content_str = "\n".join([f"{k}: {v}" for k, v in contacts.items()])
                documents.append({
                    "id": "APP-02",
                    "chapter": "附錄",
                    "title": "緊急聯絡人 (Emergency Contacts)",
                    "content": content_str,
                    "keywords": ["緊急聯絡", "電話", "分機", "Contact"],
                    "parameters": {},
                    "priority": "NORMAL",
                    "full_text": f"附錄 - 緊急聯絡人\n{content_str}\n關鍵字: 緊急, 聯絡人"
                })

            # 3. Reference Documents
            if "reference_documents" in appendix:
                refs = appendix["reference_documents"]
                content_str = "\n".join(refs)
                documents.append({
                    "id": "APP-03",
                    "chapter": "附錄",
                    "title": "參考文件 (Reference Documents)",
                    "content": content_str,
                    "keywords": ["參考文件", "SOP", "相關文獻"],
                    "parameters": {},
                    "priority": "NORMAL",
                    "full_text": f"附錄 - 參考文件\n{content_str}\n關鍵字: 參考文件, SOP"
                })
        
        return documents
    
    def _build_index(self):
        """建立 FAISS 向量索引"""
        print("正在建立向量索引...")
        
        # 解析文件
        self._documents = self._parse_manual()
        
        if not self._documents:
            print("⚠️ 沒有文件可供索引")
            return
        
        # 取得 Embedding
        model = self._get_embedding_model()
        texts = [doc["full_text"] for doc in self._documents]
        
        # 計算向量
        embeddings = model.encode(texts, show_progress_bar=True)
        self._embeddings = np.array(embeddings).astype('float32')
        
        # 建立 FAISS 索引
        try:
            import faiss
            dimension = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)  # 內積相似度
            # 正規化向量以使用 cosine similarity
            faiss.normalize_L2(self._embeddings)
            self._index.add(self._embeddings)
            print(f"✅ FAISS 索引建立完成 (維度: {dimension}, 文件數: {len(self._documents)})")
        except ImportError:
            print("⚠️ FAISS 未安裝，使用 NumPy 進行檢索")
            self._index = None
        
        # 儲存 metadata
        self._metadata = [
            {
                "id": doc["id"],
                "chapter": doc["chapter"],
                "title": doc["title"],
                "keywords": doc["keywords"],
                "priority": doc["priority"]
            }
            for doc in self._documents
        ]
    
    def _save_cache(self, cache_file: str):
        """儲存快取"""
        cache_data = {
            "documents": self._documents,
            "metadata": self._metadata,
            "embeddings": self._embeddings
        }
        
        # 若是簡易編碼器，需儲存狀態
        if isinstance(self._embedding_model, SimpleTfidfEncoder):
            cache_data["simple_encoder_state"] = {
                "vocabulary": self._embedding_model.vocabulary,
                "idf": self._embedding_model.idf,
                "fitted": self._embedding_model.fitted
            }
            
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"✅ 向量快取已儲存: {cache_file}")
    
    def _load_cache(self, cache_file: str):
        """載入快取"""
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self._documents = cache_data["documents"]
        self._metadata = cache_data["metadata"]
        self._embeddings = cache_data["embeddings"]
        
        # 恢復簡易編碼器狀態 (若有)
        if "simple_encoder_state" in cache_data:
            state = cache_data["simple_encoder_state"]
            # 確保使用 TF-IDF 編碼器
            model = self._get_embedding_model()
            if isinstance(model, SimpleTfidfEncoder):
                model.vocabulary = state["vocabulary"]
                model.idf = state["idf"]
                model.fitted = state["fitted"]
                print("✅ 簡易 TF-IDF 模型狀態已恢復")
        
        # 重建 FAISS 索引
        try:
            import faiss
            dimension = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)
            self._index.add(self._embeddings)
        except ImportError:
            self._index = None
    
    def search(
        self, 
        query: str, 
        top_k: int = 3,
        defect_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        語意檢索相關文件
        
        Args:
            query: 查詢文字
            top_k: 返回前 k 個結果
            defect_type: 可選的瑕疵類型過濾
            
        Returns:
            相關文件列表，包含分數
        """
        if not self._documents:
            return []
        
        # 如果指定了瑕疵類型，加入查詢
        if defect_type:
            query = f"{defect_type} 瑕疵 {query}"
        
        # 計算查詢向量
        model = self._get_embedding_model()
        query_embedding = model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        if (
            hasattr(self, "_embeddings")
            and len(query_embedding.shape) == 2
            and len(self._embeddings.shape) == 2
            and query_embedding.shape[1] != self._embeddings.shape[1]
        ):
            print(
                "⚠️ 向量快取維度與目前 embedding backend 不一致，重新建立索引..."
            )
            self._rebuild_cache()
            model = self._get_embedding_model()
            query_embedding = np.array(model.encode([query])).astype('float32')
        
        if self._index is not None:
            # 使用 FAISS 檢索
            import faiss
            if np.linalg.norm(query_embedding) == 0:
                scores = np.zeros((1, min(top_k, len(self._documents))), dtype="float32")
                indices = np.arange(min(top_k, len(self._documents))).reshape(1, -1)
            else:
                faiss.normalize_L2(query_embedding)
                scores, indices = self._index.search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self._documents):
                    doc = self._documents[idx].copy()
                    doc["relevance_score"] = float(score)
                    results.append(doc)
        else:
            # 使用 NumPy cosine similarity
            query_magnitude = np.linalg.norm(query_embedding)
            if query_magnitude == 0:
                scores = np.zeros(len(self._documents), dtype="float32")
            else:
                query_norm = query_embedding / query_magnitude
                scores = np.dot(self._embeddings, query_norm.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                doc = self._documents[idx].copy()
                doc["relevance_score"] = float(scores[idx])
                results.append(doc)
        
        return results
    
    def get_solution_by_defect(self, defect_type: str, top_k: int = 5) -> Dict[str, Any]:
        """
        根據瑕疵類型取得完整的診斷建議
        優先嘗試「章節標題匹配」的確定性檢索，若無結果則降級為「向量語意搜尋」
        
        Args:
            defect_type: 瑕疵類型 (Center, Donut, Edge-Ring, etc.)
            top_k: 檢索的相關文件數量
            
        Returns:
            包含診斷建議的結構化資料
        """
        # 策略 1: 確定性章節匹配 (Deterministic Chapter Lookup)
        # 檢查章節標題是否包含瑕疵類別 (IgnoreCase)
        matched_docs = []
        if self._documents:
            target_lower = defect_type.lower()
            for doc in self._documents:
                chapter_lower = doc['chapter'].lower()
                is_match = False
                
                # 特殊規則：防止 "Loc" 誤匹配到 "Edge-Loc"
                # (因為 "Loc" 字串包含於 "Edge-Loc" 中)
                if target_lower == "loc" and "edge-loc" in chapter_lower:
                    is_match = False
                elif target_lower in chapter_lower:
                    is_match = True
                    
                if is_match:
                    # 複製並標記為最高相關度
                    d = doc.copy()
                    d["relevance_score"] = 1.0
                    matched_docs.append(d)
        
        if matched_docs:
            print(f"✅ 命中確定性規則：找到 {len(matched_docs)} 個 {defect_type} 相關章節")
            results = matched_docs
        else:
            # 策略 2: 向量語意搜尋 (Vector Semantic Search)
            print(f"⚠️ 無法命中章節，降級為語意搜尋: {defect_type}")
            query = f"{defect_type} 瑕疵的成因分析、診斷方法與參數調整建議"
            results = self.search(query, top_k=top_k, defect_type=defect_type)
        
        if not results:
            return {
                "defect_type": defect_type,
                "found": False,
                "message": "知識庫中無此瑕疵類別的相關資訊"
            }
        
        # 組織結果
        response = {
            "defect_type": defect_type,
            "found": True,
            "sections": [],
            "all_parameters": {},
            "priority_level": "NORMAL",
            "source_references": []
        }
        
        for doc in results:
            section_info = {
                "chapter": doc.get("chapter", "Unknown Chapter"),
                "title": doc["title"],
                "content": doc["content"],
                "keywords": doc["keywords"],
                "relevance_score": doc["relevance_score"]
            }
            response["sections"].append(section_info)
            
            # 收集所有參數
            if doc.get("parameters"):
                response["all_parameters"].update(doc["parameters"])
            
            # 檢查優先級
            if doc.get("priority") in ["CRITICAL", "EMERGENCY"]:
                response["priority_level"] = doc["priority"]
            
            # 記錄來源
            response["source_references"].append(f"{doc['chapter']} > {doc['title']}")
        
        return response


class SimpleTfidfEncoder:
    """
    簡易 TF-IDF 編碼器（當 sentence-transformers 未安裝時使用）
    """
    
    def __init__(self):
        self.vocabulary = {}
        self.idf = None
        self.fitted = False
    
    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """將文本編碼為向量"""
        from collections import Counter
        import math
        
        # 簡易分詞
        def tokenize(text):
            # 移除標點並分詞
            import re
            tokens = re.findall(r'\w+', text.lower())
            return tokens
        
        # 建立詞彙表
        if not self.fitted:
            all_tokens = []
            for text in texts:
                all_tokens.extend(set(tokenize(text)))
            
            token_counts = Counter(all_tokens)
            self.vocabulary = {token: idx for idx, (token, _) in enumerate(token_counts.most_common(1000))}
            
            # 計算 IDF
            n_docs = len(texts)
            doc_freq = Counter()
            for text in texts:
                doc_freq.update(set(tokenize(text)))
            
            self.idf = np.zeros(len(self.vocabulary))
            for token, idx in self.vocabulary.items():
                self.idf[idx] = math.log((n_docs + 1) / (doc_freq[token] + 1)) + 1
            
            self.fitted = True
        
        # 計算 TF-IDF 向量
        vectors = []
        for text in texts:
            tokens = tokenize(text)
            tf = Counter(tokens)
            vec = np.zeros(len(self.vocabulary))
            for token, count in tf.items():
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    vec[idx] = count * self.idf[idx]
            # 正規化
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        
        return np.array(vectors)


# 向後相容的 KnowledgeBase 類別
class KnowledgeBase:
    """
    整合型知識庫 - 結合簡單查詢與向量檢索
    """
    
    def __init__(self, json_path: str = None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 載入簡易對照表
        simple_path = os.path.join(self.base_dir, "solutions.json")
        try:
            with open(simple_path, 'r', encoding='utf-8') as f:
                self.simple_solutions = json.load(f)
        except FileNotFoundError:
            self.simple_solutions = {}
        
        # 初始化向量知識庫
        try:
            self.vector_kb = VectorKnowledgeBase()
            self.use_vector = True
            print("✅ 向量知識庫已啟用")
        except Exception as e:
            self.vector_kb = None
            self.use_vector = False
            print(f"⚠️ 向量知識庫初始化失敗: {e}，使用簡易模式")
    
    def get_solution(self, defect_label: str) -> str:
        """
        取得瑕疵的維修建議（簡易版本，向後相容）
        """
        return self.simple_solutions.get(
            defect_label, 
            "資料庫中無此瑕疵類別的特定維修建議，建議人工排查。"
        )
    
    def get_detailed_solution(self, defect_label: str) -> Dict[str, Any]:
        """
        取得詳細的診斷建議（使用向量檢索）
        """
        if self.use_vector and self.vector_kb:
            return self.vector_kb.get_solution_by_defect(defect_label)
        else:
            # 降級到簡易模式
            simple = self.get_solution(defect_label)
            return {
                "defect_type": defect_label,
                "found": True,
                "sections": [{"title": "基本建議", "content": simple, "keywords": []}],
                "all_parameters": {},
                "source_references": ["solutions.json"]
            }
    
    def search_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        語意搜尋知識庫
        """
        if self.use_vector and self.vector_kb:
            return self.vector_kb.search(query, top_k=top_k)
        else:
            return []
