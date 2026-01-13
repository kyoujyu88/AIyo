import os
import glob
import pickle
import numpy as np
import faiss
from llama_cpp import Llama 

class RAGManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.knowledge_dir = os.path.join(base_dir, "knowledge")
        self.db_path = os.path.join(base_dir, "vector_db")
        
        # Gemmaモデルのパス（ここはそのままでOK）
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

        raw_chunks = []
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
                            raw_chunks.append(f"【出典:{filename}】\n{chunk_text}")
            except: pass

        if not raw_chunks: return "有効なテキストがありませんでした"

        # ★ここを修正！安全装置付きのベクトル化ループ
        embeddings = []
        valid_chunks = [] # 成功したテキストだけを入れるリスト
        expected_dim = None # 最初のデータのサイズを基準にする

        print(f"ベクトル化を開始します({len(raw_chunks)}件)...")
        print("※Gemmaを使っているため、少し時間がかかります")
        
        for i, chunk in enumerate(raw_chunks):
            try:
                vec = self.embed_model.create_embedding(chunk)
                current_vec = vec['data'][0]['embedding']
                current_dim = len(current_vec)
                
                # 最初の1個目で「基準サイズ」を決める
                if expected_dim is None:
                    expected_dim = current_dim
                
                # 基準サイズと同じなら採用、違ったら捨てる
                if current_dim == expected_dim:
                    embeddings.append(current_vec)
                    valid_chunks.append(chunk)
                else:
                    print(f"スキップ: チャンク{i}のサイズ異常 ({current_dim} vs {expected_dim})")

            except Exception as e:
                print(f"Error at chunk {i}: {e}")
            
            if (i+1) % 5 == 0: print(f"{i+1}/{len(raw_chunks)} 完了")

        if not embeddings: return "ベクトル化に失敗しました"

        # FAISSへ登録
        self.index = faiss.IndexFlatL2(expected_dim)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # ★成功したものだけを正解データとして保存
        self.chunks = valid_chunks

        faiss.write_index(self.index, os.path.join(self.db_path, "index.faiss"))
        with open(os.path.join(self.db_path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

        return f"完了！ {len(valid_chunks)}個のデータを処理しました。"

    def get_context(self, query):
        if self.index is None or not self.chunks: return "", []
        err = self._load_model()
        if err: return "", []

        try:
            # 質問もベクトル化
            query_vec = self.embed_model.create_embedding(query)['data'][0]['embedding']
            
            # 検索
            k = 3
            distances, indices = self.index.search(np.array([query_vec]).astype('float32'), k)
            
            results = []
            source_files = []
            for i in indices[0]:
                if i < len(self.chunks) and i >= 0:
                    results.append(self.chunks[i])
                    try:
                        # ファイル名の抽出
                        if "【出典:" in self.chunks[i]:
                            fname = self.chunks[i].split("【出典:")[1].split("】")[0]
                            if fname not in source_files: source_files.append(fname)
                    except: pass

            if results:
                context_text = "\n\n".join(results)
                formatted = f"\n\n### 参照情報 ###\n{context_text}\n################\n"
                return formatted, source_files
        except: pass
        
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
                print("ベクトルDBを読み込みました")
        except: pass
