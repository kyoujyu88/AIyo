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
            n_ctx = self.config.params.get("n_ctx", 2048)
            
            print(f"DEBUG: モデル読込開始 (Threads={threads}, ctx={n_ctx})")
            
            self.llm = Llama(
                model_path=path,
                n_ctx=n_ctx,
                n_threads=threads,
                n_gpu_layers=0,
                verbose=True # ★詳細なログを出すように変更
            )
            return True, os.path.basename(path)
        except Exception as e:
            print(f"DEBUG: モデル読込エラー: {e}")
            return False, f"読込エラー: {e}"

    def generate(self, prompt):
        if not self.llm:
            print("DEBUG: モデルが読み込まれていません")
            return None
        
        self.stop_flag = False
        
        # ★プロンプトの長さをチェック
        # 日本語はざっくり文字数くらいがトークン数の目安
        print(f"DEBUG: 生成リクエスト受信。文字数: {len(prompt)}")
        
        try:
            # 生成開始
            stream = self.llm(
                prompt,
                max_tokens=self.config.params["max_tokens"],
                temperature=self.config.params["temperature"],
                top_k=self.config.params["top_k"],
                repeat_penalty=self.config.params["repeat_penalty"],
                stream=True
            )
            print("DEBUG: ストリーム生成を開始しました")
            return stream
            
        except Exception as e:
            # ★ここでエラーの正体を暴く！
            print(f"\n!!!!!!!! エラー発生 !!!!!!!!\n{e}\n!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            
            # コンテキスト長あふれの場合のヒント
            if "exceeds context window" in str(e):
                print("ヒント: n_ctx のサイズが足りていません。")
                
            return None

    def stop(self):
        self.stop_flag = True
