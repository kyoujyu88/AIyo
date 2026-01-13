import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
import threading
import os
import glob
import psutil  # CPUモニター用

# 分割したチームメンバーを読み込み
from config import ConfigManager
from rag import RAGManager
from engine import AIEngine

class AIChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI分析アシスタント (完全版)")
        self.root.geometry("900x850")
        self.root.resizable(True, True)
        
        # --- チーム結成 ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = ConfigManager(base_dir) # 設定管理
        self.rag = RAGManager(base_dir)       # 知識管理
        self.engine = AIEngine(self.config)   # AI脳
        
        # 変数初期化
        self.current_mode = tk.StringVar(value=self.config.params["last_mode"])
        self.
