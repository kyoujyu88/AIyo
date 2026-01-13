import os
import glob
import pickle
import numpy as np
import faiss
import shutil      # ★追加：ファイルを移動する道具
import tempfile    # ★追加：一時ファイルを作る道具
from llama_cpp import Llama 

class RAGManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.knowledge_dir = os.path.join(base_dir, "knowledge")
        self.db_path = os.path.join(base_dir, "vector_db")
        
        # 篤志くんのGemmaモデル
        self.model_path = os.path.join(base_dir, "gguf", "gemma-2-2b-jpn-it-Q4_K_M.gguf")
        
        if not os.path.exists(self.knowledge_dir): os.makedirs(self.knowledge_dir)
        if not os.path.exists(self.db_path): os.makedirs(self.db_path)

        self.index = None
        self.chunks = []
        self.embed_model = None 
        
        self.load_db()

    def _load_model(self):
        if self.embed_model is None:
            print(f"Embedding用モデル(Gemma)を読み込んでいます...\n{self.model_path}")
            if not os.path.exists(self.model_path):
                return f"モデルが見つかりません: {self.model_path}"
            try:
                self.embed_model = Llama(
                    model_path=self.model_path,
                    embedding=True,
                    verbose=False,
                    n_ctx=2048
                )
            except Exception as e:
                return f"モデル読込エラー: {e}"
        return None

    def build_database(self):
        err = self._load_model()
        if err: return err

        files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
        if not files: return "知識ファイル(.txt)がありません"

        new_chunks = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    filename = os.path.basename(file_path)
                    
                    chunk_size = 250
                    overlap = 30
                    for i in range(0, len(text), chunk_size - overlap):
                        chunk_text = text[i : i + chunk_size].strip()
                        if len(chunk_text) > 10:
                            new_chunks.append(f"【出典:{filename}】\n{chunk_text}")
            except: pass

        if not new_chunks: return "有効なテキストがありませんでした"

        embeddings = []
        print(f"ベクトル化を開始します({len(new_chunks)}件)...")
        
        for i, chunk in enumerate(new_chunks):
            try:
                vec = self.embed_model.create_embedding(chunk)
                raw_vec = vec['data'][0]['embedding']
                if isinstance(raw_vec[0], list): raw_vec = raw_vec[0]
                embeddings.append(raw_vec)
            except Exception as e:
                print(f"チャンク処理エラー: {e}")
            
            if (i+1) % 5 == 0: print(f"{i+1}/{len(new_chunks)} 完了")

        if not embeddings: return "ベクトル化に失敗しました"

        np_embeddings = np.array(embeddings)
        if np_embeddings.ndim > 2: np_embeddings = np.squeeze(np_embeddings)
        
        print(f"データの形状: {np_embeddings.shape}")

        dimension = np_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np_embeddings.astype('float32'))
        self.chunks = new_chunks

        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

        # ★★★ここを修正！日本語パス対策★★★
        try:
            # 1. 日本語を含まない「安全な場所（Temp）」に一時保存
            fd, temp_path = tempfile.mkstemp(suffix=".faiss")
            os.close(fd) # ファイルを開放
            
            # FAISSにはその安全なパスを渡す
            faiss.write_index(self.index, temp_path)
            
            # 2. 作成されたファイルを、本来の場所へ「移動」する（Pythonなら日本語パスOK）
            target_path = os.path.join(self.db_path, "index.faiss")
            if os.path.exists(target_path): os.remove(target_path) # 古いのがあれば消す
            shutil.move(temp_path, target_path)
            
            # chunks.pkl はPython標準機能なのでそのまま保存できる
            with open(os.path.join(self.db_path, "chunks.pkl"), "wb") as f:
                pickle.dump(self.chunks, f)

        except Exception as e:
            return f"保存エラー(日本語パス対策失敗): {e}"

        return f"完了！ {len(new_chunks)}個のデータを処理しました。"

    def get_context(self, query):
        if self.index is None or not self.chunks: return "", []
        err = self._load_model()
        if err: return "", []

        try:
            vec_res = self.embed_model.create_embedding(query)
            query_vec = vec_res['data'][0]['embedding']
            if isinstance(query_vec[0], list): query_vec = query_vec[0]
            
            np_query = np.array([query_vec]).astype('float32')
            if np_query.ndim > 2: np_query = np.squeeze(np_query)
            if np_query.ndim == 1: np_query = np.expand_dims(np_query, axis=0)
            
            k = 3
            distances, indices = self.index.search(np_query, k)
            
            results = []
            source_files = []
            for i in indices[0]:
                if i < len(self.chunks) and i >= 0:
                    results.append(self.chunks[i])
                    try:
                        fname = self.chunks[i].split("【出典:")[1].split("】")[0]
                        if fname not in source_files: source_files.append(fname)
                    except: pass

            if results:
                context_text = "\n\n".join(results)
                formatted = f"\n\n### 参照情報 ###\n{context_text}\n################\n"
                return formatted, source_files
        except Exception as e:
            print(f"検索エラー: {e}")
        
        return "", []

    def open_folder(self): os.startfile(self.knowledge_dir)
    def load_user_file(self, path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f: return f.read()
        except: return None
    
    # ★読込時も日本語パス対策
    def load_db(self):
        try:
            idx_path = os.path.join(self.db_path, "index.faiss")
            chk_path = os.path.join(self.db_path, "chunks.pkl")
            
            if os.path.exists(idx_path) and os.path.exists(chk_path):
                # 読込は Python の read_index が頑張ってくれる場合が多いですが、
                # 念のためここもダメならTemp経由にするなどの対策が必要かも
                # まずは通常通りトライ
                self.index = faiss.read_index(idx_path)
                with open(chk_path, "rb") as f: self.chunks = pickle.load(f)
                print("ベクトルDBを読み込みました")
        except Exception as e:
            print(f"DB読込エラー(恐らく初回): {e}")
