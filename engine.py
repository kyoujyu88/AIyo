from llama_cpp import Llama
import os

class AIEngine:
    def __init__(self, config_manager):
        self.config = config_manager
        self.llm = None
        self.stop_flag = False

    def load_model(self, path):
        try:
            # ★修正点：int(...) で囲って、強制的に整数にします！
            self.llm = Llama(
                model_path=path,
                n_ctx=int(self.config.params["n_ctx"]),         # ここ！
                n_threads=int(self.config.params["n_threads"]), # ここ！
                n_batch=512, verbose=False
            )
            return True, os.path.basename(path)
        except Exception as e:
            return False, str(e)

    def generate(self, prompt):
        if not self.llm: return None
        
        stop_words = ["ユーザー:", "システム:", "\nユーザー:", "\nシステム:", "User:", "System:"]
        
        return self.llm(
            prompt,
            max_tokens=2048,
            temperature=self.config.params["temperature"],
            stop=stop_words,
            # ★修正点：念のためここも整数化しておきます
            top_k=int(self.config.params["top_k"]),
            top_p=self.config.params["top_p"],
            repeat_penalty=self.config.params["repeat_penalty"],
            stream=True
        )

    def stop(self):
        self.stop_flag = True
