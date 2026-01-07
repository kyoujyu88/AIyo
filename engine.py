from llama_cpp import Llama
import os

class AIEngine:
    def __init__(self, config_manager):
        self.config = config_manager
        self.llm = None
        self.stop_flag = False

    def load_model(self, path):
        try:
            self.llm = Llama(
                model_path=path,
                n_ctx=self.config.params["n_ctx"],
                n_threads=self.config.params["n_threads"],
                n_batch=512, verbose=False
            )
            return True, os.path.basename(path)
        except Exception as e:
            return False, str(e)

    def generate(self, prompt):
        if not self.llm: return None
        
        # 暴走防止ストッパー
        stop_words = ["ユーザー:", "システム:", "\nユーザー:", "\nシステム:", "User:", "System:"]
        
        return self.llm(
            prompt,
            max_tokens=2048,
            temperature=self.config.params["temperature"],
            stop=stop_words,
            stream=True
        )

    def stop(self):
        self.stop_flag = True
