import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
import threading
import os
import glob
import psutil  # CPU監視用

# 各担当モジュールの読み込み
from config import ConfigManager
from rag import RAGManager
from engine import AIEngine

class AIChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI分析アシスタント (Ryzen 9 Edition)")
        self.root.geometry("950x850") # 少し横幅を広げました
        self.root.resizable(True, True)
        
        # --- チーム結成 ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = ConfigManager(base_dir)
        self.rag = RAGManager(base_dir)
        self.engine = AIEngine(self.config)
        
        # 変数初期化
        self.current_mode = tk.StringVar(value=self.config.params.get("last_mode", "normal"))
        self.model_map = {}
        self.history = ""
        self.system_prompt = ""

        # UI構築
        self._setup_top_area()
        self._setup_mode_area()
        self._setup_log_area()
        self._setup_input_area()
        self._setup_status_bar()
        
        # 起動時処理
        self.reload_model_list()
        self.load_model()
        self.on_mode_change()
        
        # 全体監視スタート
        self.update_system_stats()

    def _setup_top_area(self):
        f = tk.Frame(self.root, bg="#e0e0e0", pady=5); f.pack(side=tk.TOP, fill=tk.X)
        tk.Label(f, text="モデル:", bg="#e0e0e0").pack(side=tk.LEFT, padx=5)
        
        self.model_combo = ttk.Combobox(f, width=35, state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Button(f, text="読込", command=self.load_model, bg="#98fb98").pack(side=tk.LEFT, padx=5)
        
        # ★新機能：Ryzenモニター起動ボタン
        tk.Button(f, text="📊 CPU詳細", command=self.open_cpu_monitor, bg="#dda0dd").pack(side=tk.LEFT, padx=5)
        
        # ベクトルDB更新ボタン
        tk.Button(f, text="🔄 DB更新", command=self.build_vector_db, bg="#ff7f50").pack(side=tk.RIGHT, padx=2)
        
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
        
        # 色分けタグ設定
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
        # Shift+Enterで改行、Enterで送信
        self.input_text.bind("<Return>", lambda e: (self.send(), "break")[1])

    def _setup_status_bar(self):
        self.status_bar = tk.Frame(self.root, bg="#333333", height=25)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = tk.Label(self.status_bar, text="System Ready", bg="#333333", fg="white", font=("Consolas", 10))
        self.status_label.pack(side=tk.RIGHT, padx=10)

    # --- ★ここが新機能！Ryzenモニター ---
    def open_cpu_monitor(self):
        monitor_win = tk.Toplevel(self.root)
        # スレッド数を取得 (5900Xなら24)
        count = psutil.cpu_count()
        monitor_win.title(f"Ryzen Monitor - {count} Threads")
        monitor_win.geometry("650x500")
        monitor_win.configure(bg="#1a1a1a")

        # スクロール対応のキャンバス（24個並ぶと縦に長いので）
        canvas = tk.Canvas(monitor_win, bg="#1a1a1a", highlightthickness=0)
        scrollbar = ttk.Scrollbar(monitor_win, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="#1a1a1a")

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # グラフ（プログレスバー）の準備
        bars = []
        labels = []
        cols = 2 # 2列で表示
        
        for i in range(count):
            row = i // cols
            col = i % cols
            
            f = tk.Frame(scroll_frame, bg="#1a1a1a", pady=5, padx=10)
            f.grid(row=row, column=col, sticky="ew")
            
            lbl = tk.Label(f, text=f"CPU {i:02}: 0.0%", fg="#00ff00", bg="#1a1a1a", font=("Consolas", 10), width=12, anchor="w")
            lbl.pack(side=tk.LEFT)
            
            # 緑色のバー
            style = ttk.Style()
            style.theme_use('default')
            style.configure("green.Horizontal.TProgressbar", background='#00ff00', troughcolor='#333333', thickness=10)
            
            pb = ttk.Progressbar(f, length=180, maximum=100, mode='determinate', style="green.Horizontal.TProgressbar")
            pb.pack(side=tk.LEFT, padx=5)
            
            bars.append(pb)
            labels.append(lbl)

        # 更新ループ関数
        def update_loop():
            if not monitor_win.winfo_exists(): return
            try:
                # 個別の負荷を取得
                percents = psutil.cpu_percent(interval=None, percpu=True)
                for i, p in enumerate(percents):
                    if i < len(bars):
                        bars[i]['value'] = p
                        labels[i].config(text=f"CPU {i:02}: {p:>4.1f}%")
                        # 80%超えで赤文字警告
                        if p > 80: labels[i].config(fg="#ff4500")
                        else: labels[i].config(fg="#00ff00")
            except: pass
            
            # 0.5秒ごとに更新
            monitor_win.after(500, update_loop)

        # 最初のキック
        update_loop()

    # --- アクション ---
    
    # 全体ステータス監視
    def update_system_stats(self):
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            mem_col = "white"
            if mem.percent > 90: mem_col = "#ff4500"
            elif mem.percent > 80: mem_col = "#ffff00"
            
            used_gb = mem.used / (1024**3)
            total_gb = mem.total / (1024**3)
            
            text = f"Total CPU: {cpu:>4.1f}% | MEM: {mem.percent:>4.1f}% ({used_gb:.1f}GB / {total_gb:.1f}GB)"
            self.status_label.config(text=text, fg=mem_col)
        except: pass
        self.root.after(1000, self.update_system_stats)

    # ベクトルDB作成
    def build_vector_db(self):
        if messagebox.askyesno("確認", "知識フォルダの内容をベクトル化してDBを更新しますか？\n（データの量によっては時間がかかります）"):
            threading.Thread(target=self._run_build_db, daemon=True).start()

    def _run_build_db(self):
        self.append_log("システム", "ベクトル化を開始しました...お待ちください...", "sys")
        msg = self.rag.build_database()
        self.root.after(0, lambda: messagebox.showinfo("完了", msg))
        self.root.after(0, lambda: self.append_log("システム", f"DB更新: {msg}", "sys"))

    # モデル操作
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
        self.root.title(f"AI Assistant (Ryzen Edition) - {name}")
        self.append_log("システム", f"準備完了: {name}", "sys")
        self.on_mode_change()

    def on_mode_change(self):
        mode = self.current_mode.get()
        self.system_prompt = self.config.get_system_prompt(mode)
        
        if mode == "normal":
            self.config.params["temperature"] = self.config.normal_temperature
        else:
            self.config.params["temperature"] = 0.0
            
        self.append_log("システム", f"モード変更: {mode}", "sys")
        self.history = self.system_prompt + "\n"
        self.config.save_settings(mode)

    # 送信
    def send(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text: return
        self.input_text.delete("1.0", tk.END)
        self.append_sep()
        self.append_log("あなた", text, "user")
        
        # RAG検索（GGUFモデルでベクトル検索）
        rag_text, files = self.rag.get_context(text)
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

    def open_settings(self):
        sw = tk.Toplevel(self.root); sw.title("設定")
        entries = {}
        keys = ["n_ctx", "temperature", "max_tokens", "repeat_penalty", "top_k"]
        for k in keys:
            if k in self.config.params:
                f = tk.Frame(sw); f.pack()
                tk.Label(f, text=k, width=15).pack(side=tk.LEFT)
                e = tk.Entry(f); e.insert(0, self.config.params[k]); e.pack(side=tk.LEFT)
                entries[k] = e
        def save():
            for k,e in entries.items():
                val = float(e.get())
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
