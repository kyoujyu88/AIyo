from llama_cpp import Llama
import os

class AIEngine:
    def __init__(self, config):
        self.llm = None
        self.config = config
        self.stop_flag = False

    def load_model(self, path):
        if not path or not os.path.exists(path):
            return False, "モデルファイルが見つかりません"
        
        try:
            # 設定からスレッド数(6)を取得
            threads = self.config.params.get("n_threads", 6)
            
            self.llm = Llama(
                model_path=path,
                n_ctx=self.config.params["n_ctx"],
                n_threads=threads,
                n_gpu_layers=0, # ★i3の内蔵グラフィックスなのでGPUオフ
                verbose=False
            )
            return True, os.path.basename(path)
        except Exception as e:
            return False, f"読込エラー: {e}"

    def generate(self, prompt):
        if not self.llm: return None
        self.stop_flag = False
        
        try:
            return self.llm(
                prompt,
                max_tokens=self.config.params["max_tokens"],
                temperature=self.config.params["temperature"],
                top_k=self.config.params["top_k"],
                repeat_penalty=self.config.params["repeat_penalty"],
                stream=True
            )
        except:
            return None

    def stop(self):
        self.stop_flag = True
