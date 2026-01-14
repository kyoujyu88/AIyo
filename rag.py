import os
import glob
import pickle
import numpy as np
import faiss
import shutil
import tempfile
from llama_cpp import Llama 

class RAGManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.knowledge_dir = os.path.join(base_dir, "knowledge")
        self.db_path = os.path.join(base_dir, "vector_db")
        
        # ★Gemmaモデルのパス（ここが合っているか確認！）
        self.model_path = os.path.join(base_dir, "gguf", "gemma-2-2b-jpn-it-Q4_K_M.gguf")
        
        if not os.path.exists(self.knowledge_dir): os.makedirs(self.knowledge_dir)
        if not os.path.exists(self.db_path): os.makedirs(self.db_path)

        self.index = None
        self.chunks = []
        self.embed_model = None 
        
        self.load_db()

    def _load_model(self):
        if self.embed_model is None:
            print(f"Embeddingモデル読込中...\n{self.model_path}")
            if not os.path.exists(self.model_path):
                return f"モデルが見つかりません: {self.model_path}"
            try:
                self.embed_model = Llama(
                    model_path=self.model_path,
                    embedding=True,
                    verbose=False,
                    n_ctx=2048,
                    n_threads=6,    # ★ここも6スレッドに制限
                    n_gpu_layers=0  # ★GPUオフ
                )
            except Exception as e:
                return f"モデル読込エラー: {e}"
        return None

    def build_database(self):
        err = self._load_model()
        if err: return err

        files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
        if not files: return "知識ファイル(.txt)がありません"

        print(f"\n【検出ファイル一覧】")
        for f in files: print(f" - {os.path.basename(f)}")
        print("-" * 20)

        new_chunks = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    filename = os.path.basename(file_path)
                    
                    # チャンク分割
                    chunk_size = 300
                    overlap = 50
                    for i in range(0, len(text), chunk_size - overlap):
                        chunk_text = text[i : i + chunk_size].strip()
                        if len(chunk_text) > 10:
                            new_chunks.append(f"【出典:{filename}】\n{chunk_text}")
            except: pass

        if not new_chunks: return "有効なテキストがありませんでした"

        embeddings = []
        print(f"ベクトル化開始 ({len(new_chunks)}件)...")
        
        for i, chunk in enumerate(new_chunks):
            try:
                vec = self.embed_model.create_embedding(chunk)
                raw_vec = vec['data'][0]['embedding']
                if isinstance(raw_vec[0], list): raw_vec = raw_vec[0]
                embeddings.append(raw_vec)
            except Exception as e:
                print(f"Error chunk {i}: {e}")
            if (i+1) % 10 == 0: print(f"{i+1}/{len(new_chunks)} 完了")

        if not embeddings: return "ベクトル化失敗"

        # 整形
        np_embeddings = np.array(embeddings)
        if np_embeddings.ndim > 2: np_embeddings = np.squeeze(np_embeddings)
        
        # FAISSインデックス作成
        dimension = np_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np_embeddings.astype('float32'))
        self.chunks = new_chunks

        # ★最強の保存処理（日本語パス対策）
        if not os.path.exists(self.db_path): os.makedirs(self.db_path)
        try:
            fd, temp_path = tempfile.mkstemp(suffix=".faiss")
            os.close(fd)
            faiss.write_index(self.index, temp_path)
            
            target_path = os.path.join(self.db_path, "index.faiss")
            if os.path.exists(target_path): os.remove(target_path)
            shutil.move(temp_path, target_path)
            
            with open(os.path.join(self.db_path, "chunks.pkl"), "wb") as f:
                pickle.dump(self.chunks, f)
        except Exception as e:
            return f"保存エラー: {e}"

        return f"完了！ {len(new_chunks)}件処理しました。"

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
            
            # ★100件取得して重複を除外する（埋もれたファイル救出作戦）
            k = 100
            total = len(self.chunks)
            if k > total: k = total
            
            distances, indices = self.index.search(np_query, k)
            
            results = []
            source_files = []
            file_counts = {}
            
            print("\n--- 検索ヒット状況 (Top選抜) ---")
            for i in indices[0]:
                if i < len(self.chunks) and i >= 0:
                    chunk = self.chunks[i]
                    try:
                        fname = chunk.split("【出典:")[1].split("】")[0]
                        
                        # 公平フィルター：1ファイルにつき最大2件まで
                        count = file_counts.get(fname, 0)
                        if count >= 2: continue
                        
                        results.append(chunk)
                        if fname not in source_files: source_files.append(fname)
                        file_counts[fname] = count + 1
                        print(f"・採用: {fname}")
                        
                        if len(results) >= 5: break
                    except: pass
            print("--------------------------------\n")

            if results:
                context_text = "\n\n".join(results)
                formatted = f"\n\n### 🧠 知識データベース参照 ###\n{context_text}\n#############################\n"
                return formatted, source_files
        except Exception as e:
            print(f"検索エラー: {e}")
        
        return "", []

    def open_folder(self): os.startfile(self.knowledge_dir)
    def load_user_file(self, path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f: return f.read()
        except: return None
    def load_db(self):
        try:
            idx = os.path.join(self.db_path, "index.faiss")
            chk = os.path.join(self.db_path, "chunks.pkl")
            if os.path.exists(idx) and os.path.exists(chk):
                self.index = faiss.read_index(idx)
                with open(chk, "rb") as f: self.chunks = pickle.load(f)
                print("DB読込完了")
        except: pass
