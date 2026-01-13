import os
import glob
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class RAGManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.knowledge_dir = os.path.join(base_dir, "knowledge")
        self.db_path = os.path.join(base_dir, "vector_db") # DB保存用フォルダ
        
        # ★モデルの場所（ここを自分のフォルダ名に合わせてください！）
        # 例: models/multilingual-e5-small
        self.model_path = os.path.join(base_dir, "models", "multilingual-e5-small")
        
        if not os.path.exists(self.knowledge_dir): os.makedirs(self.knowledge_dir)
        if not os.path.exists(self.db_path): os.makedirs(self.db_path)

        self.index = None
        self.chunks = []
        self.model = None
        
        # 起動時にDBがあれば読み込む
        self.load_db()

    def _load_model(self):
        # モデルをまだ読み込んでいなければ、ここで読み込む（初回のみ）
        if self.model is None:
            print("Embeddingモデルを読み込んでいます...")
            try:
                self.model = SentenceTransformer(self.model_path)
            except Exception as e:
                print(f"モデル読込エラー: {e}")
                return False
        return True

    def build_database(self):
        """知識フォルダのテキストを全部読んで、ベクトル化して保存する"""
        if not self._load_model(): return "モデルが見つかりません"

        files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
        if not files: return "知識ファイル(.txt)がありません"

        new_chunks = []
        
        # 1. テキストを読み込んで「チャンク（切れ端）」にする
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    filename = os.path.basename(file_path)
                    
                    # 300文字ごとに区切る（重なり50文字）
                    chunk_size = 300
                    overlap = 50
                    for i in range(0, len(text), chunk_size - overlap):
                        chunk_text = text[i : i + chunk_size].strip()
                        if len(chunk_text) > 10: # 短すぎるゴミは捨てる
                            # 「ファイル名」と「本文」をセットで保存
                            new_chunks.append(f"【出典:{filename}】\n{chunk_text}")
            except: pass

        if not new_chunks: return "有効なテキストがありませんでした"

        # 2. ベクトル化（ここが重い処理！）
        embeddings = self.model.encode(new_chunks, show_progress_bar=True)
        
        # 3. FAISSインデックス作成
        dimension = embeddings.shape[1] # ベクトルの次元数
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        self.chunks = new_chunks

        # 4. 保存
        faiss.write_index(self.index, os.path.join(self.db_path, "index.faiss"))
        with open(os.path.join(self.db_path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

        return f"完了！ {len(new_chunks)}個のデータをベクトル化しました。"

    def load_db(self):
        """保存されたDBを読み込む"""
        try:
            idx_file = os.path.join(self.db_path, "index.faiss")
            chk_file = os.path.join(self.db_path, "chunks.pkl")
            
            if os.path.exists(idx_file) and os.path.exists(chk_file):
                self.index = faiss.read_index(idx_file)
                with open(chk_file, "rb") as f:
                    self.chunks = pickle.load(f)
                print("ベクトルDBを読み込みました")
        except: pass

    def get_context(self, query):
        """質問(query)に近い情報を検索して返す"""
        if self.index is None or not self.chunks:
            return "", []
            
        if not self._load_model(): return "", []

        # 質問をベクトル化
        query_vector = self.model.encode([query])
        
        # 検索（上位3件を取得）
        k = 3
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        source_files = []
        
        for i in indices[0]:
            if i < len(self.chunks):
                results.append(self.chunks[i])
                # ファイル名を抽出（簡易的）
                try:
                    fname = self.chunks[i].split("【出典:")[1].split("】")[0]
                    if fname not in source_files: source_files.append(fname)
                except: pass

        if results:
            context_text = "\n\n".join(results)
            formatted = f"\n\n### 参照情報（ベクトル検索結果） ###\n{context_text}\n#############################\n"
            return formatted, source_files
        
        return "", []

    def open_folder(self):
        os.startfile(self.knowledge_dir)
        
    def load_user_file(self, path):
        # (ファイル読込機能はそのまま残す)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f: return f.read()
        except: return None
