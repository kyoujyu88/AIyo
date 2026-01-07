import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
import threading
import os
import glob

from config import ConfigManager
from rag import RAGManager
from engine import AIEngine

class AIChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI分析アシスタント (分割構成・修正版)")
        self.root.geometry("900x800")
        self.root.resizable(True, True)
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = ConfigManager(base_dir)
        self.rag = RAGManager(base_dir)
        self.engine = AIEngine(self.config)
        
        self.current_mode = tk.StringVar(value=self.config.params["last_mode"])
        self.model_map = {}
        self.history = ""
        self.system_prompt = ""

        self._setup_top_area()
        self._setup_mode_area()
        self._setup_log_area()
        self._setup_input_area()
        
        self.reload_model_list()
        self.load_model()
        self.on_mode_change()

    def _setup_top_area(self):
        f = tk.Frame(self.root, bg="#e0e0e0", pady=5); f.pack(side=tk.TOP, fill=tk.X)
        tk.Label(f, text="モデル:", bg="#e0e0e0").pack(side=tk.LEFT, padx=5)
        
        self.model_combo = ttk.Combobox(f, width=40, state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Button(f, text="読込", command=self.load_model, bg="#98fb98").pack(side=tk.LEFT, padx=5)
        tk.Button(f, text="📚 知識", command=self.rag.open_folder, bg="#ffd700").pack(side=tk.RIGHT, padx=2)
        tk.Button(f, text="📝 プロンプト", command=self.open_prompt, bg="#fffacd").pack(side=tk.RIGHT, padx=2)
        tk.Button(f, text="⚙ 設定", command=self.open_settings, bg="#dcdcdc").pack(side=tk.RIGHT, padx=2)

    def _setup_mode_area(self):
        f = tk.Frame(self.root, bg="#f8f8ff", pady=5); f.pack(side=tk.TOP, fill=tk.X)
        tk.Label(f, text="モード:", bg="#f8f8ff").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(f, text="通常", variable=self.current_mode, value="normal", command=self.on_mode_change, bg="#f8f8ff").pack(side=tk.LEFT)
        tk.Radiobutton(f, text="校正", variable=self.current_mode, value="proofread", command=self.on_mode_change, bg="#f8f8ff").pack(side=tk.LEFT)

    def _setup_log_area(self):
        self.log = scrolledtext.ScrolledText(self.root, font=("Meiryo", 11), state='disabled', padx=10, pady=10)
        self.log.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.log.tag_config("user", foreground="#0000cd", font=("Meiryo", 11, "bold"))
        self.log.tag_config("ai", foreground="#228b22", font=("Meiryo", 11, "bold"))
        self.log.tag_config("sys", foreground="#808080", font=("Meiryo", 9))
        self.log.tag_config("rag", foreground="#ff8c00", font=("Meiryo", 9))
        self.log.tag_config("sep", foreground="#d3d3d3")

    def _setup_input_area(self):
        f = tk.Frame(self.root, bg="#f0f0f0", pady=5); f.pack(side=tk.BOTTOM, fill=tk.X)
        bf = tk.Frame(f, bg="#f0f0f0"); bf.pack(side=tk.RIGHT, padx=5)
        
        tk.Button(bf, text="送信", command=self.send, bg="#ffb6c1", width=10, height=2).pack(pady=2)
        self.stop_btn = tk.Button(bf, text="停止", command=self.engine.stop, state="disabled", width=10); self.stop_btn.pack(pady=2)
        tk.Button(bf, text="📂 読込", command=self.load_file, bg="#87ceeb", width=10).pack(pady=2)

        self.input_text = scrolledtext.ScrolledText(f, font=("Meiryo", 11), height=4)
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.input_text.bind("<Return>", lambda e: (self.send(), "break")[1])

    # --- アクション ---
    def reload_model_list(self):
        gguf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf")
        files = glob.glob(os.path.join(gguf_dir, "*.gguf"))
        self.model_map = {os.path.basename(p): p for p in files}
        names = list(self.model_map.keys()) or ["モデルなし"]
        self.model_combo['values'] = names
        
        last = self.config.params.get("last_model", "")
        if last in names: self.model_combo.set(last)
        elif names: self.model_combo.current(0)

    def load_model(self):
        name = self.model_combo.get()
        if not name or "モデルなし" in name: return
        self.config.params["last_model"] = name
        self.config.save_settings(self.current_mode.get())
        
        path = self.model_map.get(name)
        threading.Thread(target=self._run_load, args=(path,), daemon=True).start()

    def _run_load(self, path):
        ok, msg = self.engine.load_model(path)
        if ok: self.root.after(0, lambda: self.post_load(msg))
        else: print(msg)

    def post_load(self, name):
        self.root.title(f"AI Assistant - {name}")
        self.append_log("システム", f"準備完了: {name}", "sys")
        self.on_mode_change()

    def on_mode_change(self):
        mode = self.current_mode.get()
        self.system_prompt = self.config.get_system_prompt(mode)
        
        if mode == "normal":
            self.config.params["temperature"] = self.config.normal_temperature
        else:
            self.config.params["temperature"] = 0.0
            
        self.append_log("システム", f"モード変更: {mode} (Temp:{self.config.params['temperature']})", "sys")
        self.history = self.system_prompt + "\n"
        self.config.save_settings(mode)

    def send(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text: return
        self.input_text.delete("1.0", tk.END)
        self.append_sep()
        self.append_log("あなた", text, "user")
        
        rag_text, files = self.rag.get_context()
        if files: self.append_log("システム", f"参照: {','.join(files)}", "rag")
        
        full_prompt = f"{self.history}{rag_text}ユーザー: {text}\nシステム:"
        self.history += f"ユーザー: {text}\nシステム:"
        
        self.stop_btn.config(state="normal", bg="#ff4500")
        self.engine.stop_flag = False
        threading.Thread(target=self._run_gen, args=(full_prompt,), daemon=True).start()

    def _run_gen(self, prompt):
        stream = self.engine.generate(prompt)
        if stream:
            self.root.after(0, lambda: self.append_log("AI", "", "ai"))
            full = ""
            for out in stream:
                if self.engine.stop_flag: break
                chunk = out['choices'][0]['text']
                full += chunk
                self.root.after(0, lambda: self.append_chunk(chunk))
            self.history += f" {full}\n"
        self.root.after(0, lambda: self.stop_btn.config(state="disabled", bg="#f0f0f0"))

    def load_file(self):
        path = filedialog.askopenfilename()
        if not path: return
        text = self.rag.load_user_file(path)
        if text:
            self.append_log("システム", f"読込: {os.path.basename(path)}", "sys")
            self.history += f"ユーザー: 以下のデータを読んで。\n\n{text[:2000]}\nシステム: 了解。\n"

    # --- 補助画面 ---
    def open_settings(self):
        sw = tk.Toplevel(self.root); sw.title("設定")
        entries = {}
        for k in ["n_ctx", "temperature", "max_tokens", "repeat_penalty"]:
            f = tk.Frame(sw); f.pack()
            tk.Label(f, text=k, width=15).pack(side=tk.LEFT)
            e = tk.Entry(f); e.insert(0, self.config.params[k]); e.pack(side=tk.LEFT)
            entries[k] = e
        
        def save():
            for k,e in entries.items():
                val = float(e.get())
                # ★修正点：整数であるべき項目は、ここでintに変換して保存します
                if k in ["n_ctx", "n_threads", "max_tokens", "top_k"]:
                    self.config.params[k] = int(val)
                else:
                    self.config.params[k] = val
                    
            if self.current_mode.get() == "normal":
                self.config.normal_temperature = self.config.params["temperature"]
            self.config.save_settings(self.current_mode.get())
            sw.destroy()
        
        tk.Button(sw, text="保存", command=save, bg="#98fb98").pack(pady=10)

    def open_prompt(self):
        os.startfile(self.config.prompt_files[self.current_mode.get()])

    # --- ログ描画 ---
    def append_sep(self):
        self.log.config(state='normal'); self.log.insert(tk.END, "\n" + "-"*40 + "\n", "sep"); self.log.config(state='disabled')
    def append_log(self, sender, text, tag):
        self.log.config(state='normal')
        if sender=="AI": self.log.insert(tk.END, f"\n【{sender}】\n", tag)
        elif sender=="システム": self.log.insert(tk.END, f"[{sender}] {text}\n", tag)
        else: self.log.insert(tk.END, f"【{sender}】\n{text}\n", tag)
        self.log.see(tk.END); self.log.config(state='disabled')
    def append_chunk(self, t):
        self.log.config(state='normal'); self.log.insert(tk.END, t); self.log.see(tk.END); self.log.config(state='disabled')
