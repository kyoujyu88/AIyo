import os
import glob
import pandas as pd

class RAGManager:
    def __init__(self, base_dir):
        self.knowledge_dir = os.path.join(base_dir, "knowledge")
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)

    def open_folder(self):
        os.startfile(self.knowledge_dir)

    def get_context(self):
        context_text = ""
        files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
        loaded_files = []
        
        if not files: return "", []

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                    if content:
                        filename = os.path.basename(file_path)
                        context_text += f"\n【参照資料: {filename}】\n{content}\n"
                        loaded_files.append(filename)
            except: pass
            
        if context_text:
            formatted = f"\n\n### 重要な参照情報 ###\n以下の資料を知識として持ち、これを踏まえて回答してください。\n{context_text}\n######################\n"
            return formatted, loaded_files
        return "", []

    def load_user_file(self, path):
        try:
            if path.endswith('.xlsx'): return pd.read_excel(path).to_string()
            elif path.endswith('.csv'): return pd.read_csv(path).to_string()
            else: 
                with open(path, "r", encoding="utf-8", errors="ignore") as f: return f.read()
        except: return None
