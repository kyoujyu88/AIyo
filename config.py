import json
import os

class ConfigManager:
    def __init__(self, base_dir):
        self.config_path = os.path.join(base_dir, "config.json")
        self.normal_temperature = 0.7
        self.params = {
            "max_tokens": 1024,
            
            # ★Gemma対応：記憶容量を大きく確保（8192）
            "n_ctx": 8192,
            
            "temperature": 0.7,
            "top_k": 40,
            "repeat_penalty": 1.1,
            
            # ★事務所PC(i3-1315U)最適化：6スレッド
            "n_threads": 6,      
            
            "last_model": "",
            "last_mode": "normal"
        }
        
        # プロンプトファイルの場所定義
        self.prompt_dir = os.path.join(base_dir, "prompts")
        if not os.path.exists(self.prompt_dir): os.makedirs(self.prompt_dir)
        
        self.prompt_files = {
            "normal": os.path.join(self.prompt_dir, "system_prompt.txt"),
            "proofread": os.path.join(self.prompt_dir, "proofread_prompt.txt")
        }
        
        # 初回起動時に空のプロンプトファイルを作成しておく（便利機能）
        for path in self.prompt_files.values():
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("あなたは優秀なアシスタントです。")

        self.load_settings()

    def load_settings(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.params.update(loaded)
            except: pass

    def save_settings(self, mode):
        self.params["last_mode"] = mode
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.params, f, indent=4)
        except: pass

    def get_system_prompt(self, mode):
        path = self.prompt_files.get(mode)
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except: pass
        return "あなたは優秀なアシスタントです。"
