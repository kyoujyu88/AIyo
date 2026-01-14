import json
import os

class ConfigManager:
    def __init__(self, base_dir):
        self.config_path = os.path.join(base_dir, "config.json")
        self.normal_temperature = 0.7
        self.params = {
            "max_tokens": 1024,   # 回答の長さ
            "n_ctx": 2048,        # 記憶量（16GBメモリならこれくらいが安全）
            "temperature": 0.7,   # 創造性
            "top_k": 40,
            "repeat_penalty": 1.1,
            
            # ★事務所PC(i3-1315U)向けの最適値
            # 8スレッド中、6スレッドをAIに使用（残り2つはOS用）
            "n_threads": 6,      
            
            "last_model": "",
            "last_mode": "normal"
        }
        self.prompt_files = {
            "normal": os.path.join(base_dir, "prompts", "system_prompt.txt"),
            "proofread": os.path.join(base_dir, "prompts", "proofread_prompt.txt")
        }
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
