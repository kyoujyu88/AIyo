from llama_cpp import Llama
import os
import sys

class AIEngine:
    def __init__(self, config):
        self.llm = None
        self.config = config
        self.stop_flag = False

    def load_model(self, path):
        if not path or not os.path.exists(path):
            return False, "モデルファイルが見つかりません"
        
        try:
            threads = self.config.params.get("n_threads", 6)
            n_ctx = self.config.params.get("n_ctx", 8192)
            
            print(f"DEBUG: Load Model (Threads={threads}, ctx={n_ctx})")
            
            self.llm = Llama(
                model_path=path,
                n_ctx=n_ctx,
                n_threads=threads,
                n_gpu_layers=0, # GPUオフ
                verbose=False 
            )
            return True, os.path.basename(path)
        except Exception as e:
            return False, f"読込エラー: {e}"

    def generate(self, prompt):
        if not self.llm: return None
        self.stop_flag = False
        
        try:
            # ★一人芝居を止めるためのブレーキ
            stop_words = [
                "User:", "ユーザー:", 
                "System:", "システム:",
                "<start_of_turn>", "<end_of_turn>",
                "<|start_header_id|>", "<|eot_id|>",
                "\n\n\n"
            ]

            stream = self.llm(
                prompt,
                max_tokens=self.config.params["max_tokens"],
                temperature=self.config.params["temperature"],
                top_k=self.config.params["top_k"],
                repeat_penalty=self.config.params["repeat_penalty"],
                stop=stop_words,
                stream=True
            )
            
            # 受信した文字をコンソールに表示するデバッグ関数
            def debug_stream():
                print("DEBUG: Stream Start")
                for output in stream:
                    token = output['choices'][0]['text']
                    # コンソールにもリアルタイム表示
                    print(f"{token}", end="", flush=True)
                    yield output
                print("\nDEBUG: Stream End")
            
            return debug_stream()
            
        except Exception as e:
            print(f"Generate Error: {e}")
            return None

    def stop(self):
        self.stop_flag = True
