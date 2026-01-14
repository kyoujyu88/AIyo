import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
import threading
import os
import glob
import psutil
from config import ConfigManager
from rag import RAGManager
from engine import AIEngine

# インデント削除用
import textwrap 

class AIChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Assistant (Office PC Final)")
        self.root.geometry("950x850")
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = ConfigManager(base_dir)
        self.rag = RAGManager(base_dir)
        self.engine = AIEngine(self.config)
        
        self.current_mode = tk.StringVar(value=self.config.params.get("last_mode", "normal"))
        self.model_map = {}
        self.history = ""
        self.system_prompt = ""

        self._setup_ui()
        self.reload_model_list()
        self.load_model()
        self.on_mode_change()
        self.update_system_stats()

    def _setup_ui(self):
        # 1. 上部エリア
        top = tk.Frame(self.root, bg="#e0e0e0", pady=5); top.pack(side=tk.TOP, fill=tk.X)
        tk.Label(top, text="モデル:", bg="#e0e0e0").pack(side=tk.LEFT, padx=5)
        self.model_combo = ttk.Combobox(top, width=30, state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="読込", command=self.load_model, bg="#98fb98").pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="📊 CPU詳細", command=self.open_cpu_monitor, bg="#dda0dd").pack(side=tk.LEFT, padx=5)
        
        tk.Button(top, text="🔄 DB更新", command=self.build_vector_db, bg="#ff7f50").pack(side=tk.RIGHT, padx=2)
        tk.Button(top, text="📚 知識", command=self.rag.open_folder, bg="#ffd700").pack(side=tk.RIGHT, padx=2)
        tk.Button(top, text="📝 プロンプト", command=self.open_prompt, bg="#fffacd").pack(side=tk.RIGHT, padx=2)
        tk.Button(top, text="⚙ 設定", command=self.open_settings, bg="#dcdcdc").pack(side=tk.RIGHT, padx=2)

        # 2. モードエリア
        mode_f = tk.Frame(self.root, bg="#f8f8ff", pady=5); mode_f.pack(side=tk.TOP, fill=tk.X)
        tk.Label(mode_f, text="モード:", bg="#f8f8ff").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(mode_f, text="通常", variable=self.current_mode, value="normal", command=self.on_mode_change, bg="#f8f8ff").pack(side=tk.LEFT)
        tk.Radiobutton(mode_f, text="校正", variable=self.current_mode, value="proofread", command=self.on_mode_change, bg="#f8f8ff").pack(side=tk.LEFT)

        # 3. ステータスバー
        self.status_label = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.E)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # 4. メインエリア
        self.paned = tk.PanedWindow(self.root, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=6)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log = scrolledtext.ScrolledText(self.paned, font=("Meiryo", 11), state='disabled')
        self.log.tag_config("user", foreground="#0000cd", font=("Meiryo", 11, "bold"))
        self.log.tag_config("ai", foreground="#228b22", font=("Meiryo", 11, "bold"))
        self.log.tag_config("sys", foreground="#808080", font=("Meiryo", 9))
        self.log.tag_config("rag", foreground="#ff8c00", font=("Meiryo", 9))
        self.paned.add(self.log, stretch="always", height=500) 

        input_frame = tk.Frame(self.paned, bg="#f0f0f0")
        self.paned.add(input_frame, stretch="never", height=150)

        bf = tk.Frame(input_frame, bg="#f0f0f0"); bf.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        tk.Button(bf, text="送信", command=self.send, bg="#ffb6c1", width=10, height=2).pack(pady=2)
        self.stop_btn = tk.Button(bf, text="停止", command=self.engine.stop, state="disabled", width=10); self.stop_btn.pack(pady=2)
        tk.Button(bf, text="📂 読込", command=self.load_file, bg="#87ceeb", width=10).pack(pady=2)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, font=("Meiryo", 11))
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.input_text.bind("<Return>", lambda e: (self.send(), "break")[1])

    # --- CPU Monitor ---
    def open_cpu_monitor(self):
        win = tk.Toplevel(self.root)
        count = psutil.cpu_count()
        win.title(f"CPU Monitor ({count} Threads)")
        win.geometry("650x400")
        win.configure(bg="#1a1a1a")
        
        canvas = tk.Canvas(win, bg="#1a1a1a"); canvas.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(win, orient="vertical", command=canvas.yview); sb.pack(side="right", fill="y")
        sf = tk.Frame(canvas, bg="#1a1a1a")
        canvas.create_window((0,0), window=sf, anchor="nw")
        sf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.configure(yscrollcommand=sb.set)

        bars = []
        labels = []
        for i in range(count):
            r, c = i // 2, i % 2
            f = tk.Frame(sf, bg="#1a1a1a", pady=5, padx=10); f.grid(row=r, column=c, sticky="ew")
            l = tk.Label(f, text=f"CPU {i:02}: 0.0%", fg="#00ff00", bg="#1a1a1a", font=("Consolas", 10), width=12)
            l.pack(side=tk.LEFT)
            p = ttk.Progressbar(f, length=180, maximum=100, mode='determinate'); p.pack(side=tk.LEFT, padx=5)
            bars.append(p); labels.append(l)

        def update():
            if not win.winfo_exists(): return
            try:
                percents = psutil.cpu_percent(interval=None, percpu=True)
                for i, v in enumerate(percents):
                    if i < len(bars):
                        bars[i]['value'] = v
                        labels[i].config(text=f"CPU {i:02}: {v:>4.1f}%", fg="#ff4500" if v>80 else "#00ff00")
            except: pass
            win.after(500, update)
        update()

    # --- Actions ---
    def update_system_stats(self):
        try:
            c = psutil.cpu_percent(interval=None)
            m = psutil.virtual_memory()
            self.status_label.config(text=f"CPU: {c}% | MEM: {m.percent}%")
        except: pass
        self.root.after(1000, self.update_system_stats)

    def build_vector_db(self):
        if messagebox.askyesno("確認", "DBを更新しますか？\n（既存の知識は上書きされます）"):
            threading.Thread(target=self._run_build, daemon=True).start()

    # ★修正箇所：リアルタイム報告を受け取って表示
    def _run_build(self):
        self.append_log("システム", "ベクトル化開始...時間がかかります...", "sys")
        
        # 画面にメッセージを出すための関数
        def update_ui(msg):
             self.root.after(0, lambda: self.append_log("システム", msg, "sys"))
        
        # callback引数に渡す
        msg = self.rag.build_database(callback=update_ui)
        self.root.after(0, lambda: messagebox.showinfo("完了", msg))

    def reload_model_list(self):
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf")
        self.model_map = {os.path.basename(p): p for p in glob.glob(os.path.join(d, "*.gguf"))}
        self.model_combo['values'] = list(self.model_map.keys())
        last = self.config.params.get("last_model", "")
        if last in self.model_map: self.model_combo.set(last)
        elif self.model_map: self.model_combo.current(0)

    def load_model(self):
        name = self.model_combo.get()
        if not name: return
        self.config.params["last_model"] = name
        self.config.save_settings(self.current_mode.get())
        threading.Thread(target=self._load_th, args=(self.model_map[name],), daemon=True).start()

    def _load_th(self, path):
        ok, msg = self.engine.load_model(path)
        if ok: self.root.after(0, lambda: self._post_load(msg))

    def _post_load(self, name):
        self.root.title(f"AI Assistant (Office PC) - {name}")
        self.append_log("システム", f"モデル読込完了: {name}", "sys")
        self.on_mode_change()

    def on_mode_change(self):
        m = self.current_mode.get()
        self.system_prompt = self.config.get_system_prompt(m)
        self.config.params["temperature"] = self.config.normal_temperature if m=="normal" else 0.0
        self.history = self.system_prompt + "\n"
        self.config.save_settings(m)
        self.append_log("システム", f"モード: {m}", "sys")

    def send(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text: return
        self.input_text.delete("1.0", tk.END)
        self.append_log("あなた", text, "user")
        
        ctx, files = self.rag.get_context(text)
        
        if files:
            self.append_log("システム", f"参照: {', '.join(files)}", "rag")
            rag_instruction = f"以下の【参照情報】を事実として、ユーザーの質問に答えてください。\n\n【参照情報】\n{ctx}"
        else:
            rag_instruction = "ユーザーの質問に親切に答えてください。"

        sys_msg = self.system_prompt
        if not sys_msg: sys_msg = "あなたは優秀なアシスタントです。"

        current_model_name = self.config.params.get("last_model", "").lower()
        
        if "gemma" in current_model_name:
            prompt = f"""<start_of_turn>user
{sys_msg}

{rag_instruction}

【ユーザーの質問】
{text}<end_of_turn>
<start_of_turn>model
"""
        elif "llama-3" in current_model_name or "elyza" in current_model_name:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{sys_msg}
{rag_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        else:
            prompt = f"""
{sys_msg}

{rag_instruction}

ユーザー: {text}
システム:
"""
        
        prompt = prompt.strip()

        self.stop_btn.config(state="normal", bg="#ff4500")
        print(f"DEBUG: Model={current_model_name}, PromptLen={len(prompt)}")
        threading.Thread(target=self._gen_th, args=(prompt,), daemon=True).start()

    def _gen_th(self, prompt):
        res = self.engine.generate(prompt)
        if res:
            self.root.after(0, lambda: self.append_log("AI", "", "ai"))
            full = ""
            try:
                for x in res:
                    if self.engine.stop_flag: break
                    t = x['choices'][0]['text']
                    full += t
                    self.root.after(0, lambda chunk=t: self._insert_chunk(chunk))
            except Exception as e:
                print(f"Gen Error: {e}")
            
            self.history += f" {full}\n"
        self.root.after(0, lambda: self.stop_btn.config(state="disabled", bg="#f0f0f0"))

    def _insert_chunk(self, text):
        self.log.config(state='normal')
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.config(state='disabled')

    def load_file(self):
        path = filedialog.askopenfilename()
        if path:
            t = self.rag.load_user_file(path)
            if t:
                self.append_log("システム", f"読込: {os.path.basename(path)}", "sys")
                self.history += f"ユーザー: 以下を読んで。\n{t[:1000]}\nシステム: はい。\n"

    def open_prompt(self):
        path = self.config.prompt_files.get(self.current_mode.get())
        if path and os.path.exists(path):
            os.startfile(path)
        else:
            messagebox.showerror("エラー", "プロンプトファイルが見つかりません")

    def open_settings(self):
        sw = tk.Toplevel(self.root); sw.title("設定")
        ents = {}
        for k in ["n_ctx", "temperature", "max_tokens", "n_threads"]:
            f = tk.Frame(sw); f.pack()
            tk.Label(f, text=k, width=15).pack(side=tk.LEFT)
            e = tk.Entry(f); e.insert(0, self.config.params.get(k, "")); e.pack(side=tk.LEFT)
            ents[k] = e
        def save():
            for k,e in ents.items():
                v = float(e.get())
                self.config.params[k] = int(v) if k in ["n_ctx", "max_tokens", "n_threads"] else v
            self.config.save_settings(self.current_mode.get())
            sw.destroy()
        tk.Button(sw, text="保存", command=save, bg="#98fb98").pack(pady=10)

    def append_log(self, sender, text, tag):
        self.log.config(state='normal')
        self.log.insert(tk.END, f"\n【{sender}】\n{text}\n" if text else f"\n【{sender}】\n", tag)
        self.log.see(tk.END); self.log.config(state='disabled')
