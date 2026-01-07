import os
import json

class ConfigManager:
    def __init__(self, base_dir):
        self.config_file = os.path.join(base_dir, "config.json")
        self.prompt_files = {
            "normal": os.path.join(base_dir, "prompt_normal.txt"),
            "proofread": os.path.join(base_dir, "prompt_proofread.txt")
        }
        
        # デフォルト設定
        self.default_params = {
            "n_ctx": 8192, "n_threads": 4, "max_tokens": 4096,
            "temperature": 0.6, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.2,
            "last_model": "", "last_mode": "normal"
        }
        self.params = self.load_settings()
        
        # 「いつもの温度」を記憶
        self.normal_temperature = self.params["temperature"]
        
        # プロンプトファイルがなければ作る
        self._create_defaults()

    def load_settings(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    loaded = json.load(f)
                    for k, v in self.default_params.items():
                        if k not in loaded: loaded[k] = v
                    return loaded
            except: pass
        return self.default_params.copy()

    def save_settings(self, current_mode="normal"):
        save_data = self.params.copy()
        # 校正モード中でも、保存時は「いつもの温度」を記録
        if current_mode == "proofread":
            save_data["temperature"] = self.normal_temperature
            
        try:
            with open(self.config_file, "w") as f: json.dump(save_data, f, indent=4)
        except: pass

    def get_system_prompt(self, mode):
        path = self.prompt_files.get(mode)
        try:
            with open(path, "r", encoding="utf-8") as f: return f.read().strip()
        except: return "システム: エラーが発生しました。"

    def _create_defaults(self):
        if not os.path.exists(self.prompt_files["normal"]):
            with open(self.prompt_files["normal"], "w", encoding="utf-8") as f: f.write("システム: 優秀なAIです。")
        if not os.path.exists(self.prompt_files["proofread"]):
            with open(self.prompt_files["proofread"], "w", encoding="utf-8") as f: f.write("システム: 校正AIです。")
