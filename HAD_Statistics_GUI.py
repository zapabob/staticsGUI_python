import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json
import pickle
import os
import signal
import time
from datetime import datetime
from tqdm import tqdm
import threading
import queue
import sys

# 統計分析とプロット機能をインポート
from HAD_Statistics_Functions import HADStatisticalAnalysis
from HAD_Plotting_Functions import HADPlottingFunctions

class HADStatisticsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HAD Statistics - Python Edition")
        self.root.geometry("1200x800")
        
        # セッション管理
        self.session_id = f"session_{int(time.time())}"
        self.backup_dir = "HAD_backups"
        self.auto_save_interval = 300  # 5分間隔
        self.backup_count = 10
        
        # データ管理
        self.data = pd.DataFrame()
        self.results_queue = queue.Queue()
        
        # 統計分析エンジン初期化
        self.stats_engine = HADStatisticalAnalysis()
        self.plot_engine = HADPlottingFunctions()
        
        # 電源断保護の初期化
        self.setup_signal_handlers()
        self.create_backup_directory()
        self.auto_save_timer = None
        
        # GUI初期化
        self.create_widgets()
        self.start_auto_save()
        
        # 復旧チェック
        self.check_recovery()

    def setup_signal_handlers(self):
        """シグナルハンドラーの設定"""
        try:
            signal.signal(signal.SIGINT, self.emergency_save)
            signal.signal(signal.SIGTERM, self.emergency_save)
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, self.emergency_save)
        except Exception as e:
            print(f"Signal handler setup warning: {e}")

    def create_backup_directory(self):
        """バックアップディレクトリ作成"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def emergency_save(self, signum=None, frame=None):
        """緊急保存機能"""
        try:
            emergency_file = os.path.join(self.backup_dir, f"emergency_{self.session_id}.pkl")
            self.save_session(emergency_file)
            print(f"Emergency save completed: {emergency_file}")
        except Exception as e:
            print(f"Emergency save failed: {e}")
        
        if signum:
            exit(0)

    def save_session(self, filename=None):
        """セッション保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.backup_dir, f"session_{timestamp}.pkl")
        
        session_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'data': self.data,
            'backup_metadata': {
                'version': '1.0',
                'python_version': str(sys.version),
                'data_shape': self.data.shape if not self.data.empty else (0, 0)
            }
        }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(session_data, f)
            
            # JSONバックアップも作成
            json_filename = filename.replace('.pkl', '.json')
            json_data = {
                'session_id': self.session_id,
                'timestamp': session_data['timestamp'],
                'data_info': {
                    'shape': session_data['backup_metadata']['data_shape'],
                    'columns': list(self.data.columns) if not self.data.empty else []
                }
            }
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
                
            self.manage_backup_rotation()
            return True
        except Exception as e:
            print(f"Session save error: {e}")
            return False

    def manage_backup_rotation(self):
        """バックアップローテーション管理"""
        try:
            backup_files = [f for f in os.listdir(self.backup_dir) if f.endswith('.pkl')]
            backup_files.sort(key=lambda x: os.path.getctime(os.path.join(self.backup_dir, x)))
            
            while len(backup_files) > self.backup_count:
                old_file = os.path.join(self.backup_dir, backup_files.pop(0))
                os.remove(old_file)
                # 対応するJSONも削除
                json_file = old_file.replace('.pkl', '.json')
                if os.path.exists(json_file):
                    os.remove(json_file)
        except Exception as e:
            print(f"Backup rotation error: {e}")

    def start_auto_save(self):
        """自動保存開始"""
        def auto_save():
            if not self.data.empty:
                self.save_session()
            self.auto_save_timer = threading.Timer(self.auto_save_interval, auto_save)
            self.auto_save_timer.daemon = True
            self.auto_save_timer.start()
        
        auto_save()

    def check_recovery(self):
        """復旧チェック"""
        try:
            backup_files = [f for f in os.listdir(self.backup_dir) if f.endswith('.pkl')]
            if backup_files:
                backup_files.sort(key=lambda x: os.path.getctime(os.path.join(self.backup_dir, x)), reverse=True)
                latest_backup = backup_files[0]
                
                response = messagebox.askyesno(
                    "復旧確認",
                    f"前回のセッション ({latest_backup}) が見つかりました。\n復旧しますか？"
                )
                
                if response:
                    self.load_session(os.path.join(self.backup_dir, latest_backup))
        except Exception as e:
            print(f"Recovery check error: {e}")

    def load_session(self, filename):
        """セッション読み込み"""
        try:
            with open(filename, 'rb') as f:
                session_data = pickle.load(f)
            
            self.data = session_data.get('data', pd.DataFrame())
            self.update_data_display()
            messagebox.showinfo("復旧完了", "セッションが正常に復旧されました。")
        except Exception as e:
            messagebox.showerror("復旧エラー", f"セッションの復旧に失敗しました: {e}")

    def create_widgets(self):
        """メインGUI作成"""
        # メインフレーム
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ノートブック（タブ）
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # データ入力タブ
        self.create_data_tab()
        
        # 記述統計タブ
        self.create_descriptive_tab()
        
        # 推定統計タブ
        self.create_inferential_tab()
        
        # グラフタブ
        self.create_plot_tab()
        
        # 結果表示タブ
        self.create_results_tab()

    def create_data_tab(self):
        """データ入力タブ作成"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="データ入力")
        
        # データ入力コントロール
        control_frame = ttk.Frame(data_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="CSVファイル読み込み", 
                  command=self.load_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="手動データ入力", 
                  command=self.manual_data_input).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="データクリア", 
                  command=self.clear_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="データ保存", 
                  command=self.save_data).pack(side=tk.LEFT, padx=5)
        
        # データ表示フレーム
        display_frame = ttk.Frame(data_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # TreeView for data display
        columns = ['Index'] + [f'Variable_{i+1}' for i in range(10)]
        self.data_tree = ttk.Treeview(display_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100, anchor=tk.CENTER)
        
        # スクロールバー
        data_scrollbar_v = ttk.Scrollbar(display_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        data_scrollbar_h = ttk.Scrollbar(display_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=data_scrollbar_v.set, xscrollcommand=data_scrollbar_h.set)
        
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        data_scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        data_scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)

    def create_descriptive_tab(self):
        """記述統計タブ作成"""
        desc_frame = ttk.Frame(self.notebook)
        self.notebook.add(desc_frame, text="記述統計")
        
        # コントロールフレーム
        control_frame = ttk.LabelFrame(desc_frame, text="統計分析選択")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="基本統計量", 
                  command=self.basic_statistics).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="度数分布", 
                  command=self.frequency_distribution).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="相関分析", 
                  command=self.correlation_analysis).pack(side=tk.LEFT, padx=5, pady=5)
        
        # 結果表示フレーム
        self.desc_results = tk.Text(desc_frame, height=25, wrap=tk.WORD)
        desc_scrollbar = ttk.Scrollbar(desc_frame, command=self.desc_results.yview)
        self.desc_results.configure(yscrollcommand=desc_scrollbar.set)
        
        self.desc_results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        desc_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

    def create_inferential_tab(self):
        """推定統計タブ作成"""
        inf_frame = ttk.Frame(self.notebook)
        self.notebook.add(inf_frame, text="推定統計")
        
        # コントロールフレーム
        control_frame = ttk.LabelFrame(inf_frame, text="検定選択")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="t検定", 
                  command=self.t_test).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="ANOVA", 
                  command=self.anova_test).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="回帰分析", 
                  command=self.regression_analysis).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="χ²検定", 
                  command=self.chi_square_test).pack(side=tk.LEFT, padx=5, pady=5)
        
        # 結果表示フレーム
        self.inf_results = tk.Text(inf_frame, height=25, wrap=tk.WORD)
        inf_scrollbar = ttk.Scrollbar(inf_frame, command=self.inf_results.yview)
        self.inf_results.configure(yscrollcommand=inf_scrollbar.set)
        
        self.inf_results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        inf_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

    def create_plot_tab(self):
        """グラフタブ作成"""
        plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(plot_frame, text="グラフ")
        
        # コントロールフレーム
        control_frame = ttk.LabelFrame(plot_frame, text="グラフ選択")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="ヒストグラム", 
                  command=self.create_histogram).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="散布図", 
                  command=self.create_scatter).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="箱ひげ図", 
                  command=self.create_boxplot).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="相関行列", 
                  command=self.create_correlation_matrix).pack(side=tk.LEFT, padx=5, pady=5)
        
        # プロット表示フレーム
        self.plot_frame = ttk.Frame(plot_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_results_tab(self):
        """結果表示タブ作成"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="分析結果")
        
        # 結果表示
        self.results_text = tk.Text(results_frame, height=30, wrap=tk.WORD, font=('Consolas', 10))
        results_scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # 結果保存ボタン
        ttk.Button(results_frame, text="結果をファイルに保存", 
                  command=self.save_results).pack(pady=5)

    # データ操作メソッド
    def load_csv(self):
        """CSVファイル読み込み"""
        try:
            file_path = filedialog.askopenfilename(
                title="CSVファイルを選択",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if file_path:
                self.data = pd.read_csv(file_path, encoding='utf-8')
                self.update_data_display()
                self.save_session()  # 自動保存
                messagebox.showinfo("成功", f"データを読み込みました。\n形状: {self.data.shape}")
        except Exception as e:
            messagebox.showerror("エラー", f"ファイルの読み込みに失敗しました: {e}")

    def manual_data_input(self):
        """手動データ入力ダイアログ"""
        input_window = tk.Toplevel(self.root)
        input_window.title("手動データ入力")
        input_window.geometry("600x400")
        
        # 入力フレーム
        input_frame = ttk.LabelFrame(input_window, text="データ入力 (カンマ区切りで入力)")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # テキストエリア
        text_area = tk.Text(input_frame, height=15, wrap=tk.WORD)
        text_scrollbar = ttk.Scrollbar(input_frame, command=text_area.yview)
        text_area.configure(yscrollcommand=text_scrollbar.set)
        
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # サンプルデータ表示
        sample_text = """# サンプルデータ（この行は削除してください）
Variable1,Variable2,Variable3
1.5,2.3,3.1
2.1,1.9,2.8
3.2,3.5,4.2
1.8,2.1,2.9
2.9,3.1,3.7"""
        text_area.insert(tk.END, sample_text)
        
        def process_input():
            try:
                content = text_area.get("1.0", tk.END).strip()
                lines = [line for line in content.split('\n') if line.strip() and not line.startswith('#')]
                
                if len(lines) < 2:
                    messagebox.showerror("エラー", "ヘッダーとデータが必要です")
                    return
                
                # CSVとして処理
                from io import StringIO
                csv_data = '\n'.join(lines)
                self.data = pd.read_csv(StringIO(csv_data))
                
                self.update_data_display()
                self.save_session()  # 自動保存
                input_window.destroy()
                messagebox.showinfo("成功", f"データを入力しました。\n形状: {self.data.shape}")
                
            except Exception as e:
                messagebox.showerror("エラー", f"データの処理に失敗しました: {e}")
        
        ttk.Button(input_frame, text="データを登録", command=process_input).pack(pady=5)

    def update_data_display(self):
        """データ表示更新"""
        # 既存のアイテムをクリア
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        if not self.data.empty:
            # 統計エンジンにデータを設定
            self.stats_engine.set_data(self.data)
            self.plot_engine.set_data(self.data)
            
            # 列名を更新
            columns = ['Index'] + list(self.data.columns)
            self.data_tree['columns'] = columns
            
            for col in columns:
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=100, anchor=tk.CENTER)
            
            # データを挿入
            for idx, row in self.data.iterrows():
                values = [idx] + [str(val) for val in row.values]
                self.data_tree.insert('', tk.END, values=values)

    def clear_data(self):
        """データクリア"""
        self.data = pd.DataFrame()
        self.update_data_display()
        self.save_session()
        messagebox.showinfo("完了", "データがクリアされました。")

    def save_data(self):
        """データ保存"""
        if self.data.empty:
            messagebox.showwarning("警告", "保存するデータがありません。")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                title="データを保存",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
            )
            
            if file_path:
                if file_path.endswith('.xlsx'):
                    self.data.to_excel(file_path, index=False)
                else:
                    self.data.to_csv(file_path, index=False, encoding='utf-8')
                
                messagebox.showinfo("完了", f"データが保存されました: {file_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"データの保存に失敗しました: {e}")

    # 統計分析メソッド
    def basic_statistics(self):
        """基本統計量"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが設定されていません。")
            return
        
        try:
            result = self.stats_engine.basic_statistics()
            self.desc_results.delete(1.0, tk.END)
            self.desc_results.insert(tk.END, result)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("エラー", f"基本統計量の計算でエラーが発生しました: {e}")

    def frequency_distribution(self):
        """度数分布"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが設定されていません。")
            return
        
        # 列選択ダイアログ
        columns = list(self.data.columns)
        if not columns:
            messagebox.showwarning("警告", "分析する列がありません。")
            return
        
        # 簡単な列選択ダイアログ
        def select_column():
            dialog = tk.Toplevel(self.root)
            dialog.title("列選択")
            dialog.geometry("300x200")
            
            ttk.Label(dialog, text="分析する列を選択してください:").pack(pady=10)
            
            column_var = tk.StringVar(value=columns[0])
            column_combo = ttk.Combobox(dialog, textvariable=column_var, values=columns, state="readonly")
            column_combo.pack(pady=5)
            
            bins_label = ttk.Label(dialog, text="ビン数 (数値データの場合):")
            bins_label.pack(pady=5)
            bins_var = tk.IntVar(value=10)
            bins_entry = ttk.Entry(dialog, textvariable=bins_var)
            bins_entry.pack(pady=5)
            
            def ok_clicked():
                try:
                    result = self.stats_engine.frequency_distribution(column_var.get(), bins_var.get())
                    self.desc_results.delete(1.0, tk.END)
                    self.desc_results.insert(tk.END, result)
                    self.results_text.delete(1.0, tk.END)
                    self.results_text.insert(tk.END, result)
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("エラー", f"度数分布の計算でエラーが発生しました: {e}")
            
            ttk.Button(dialog, text="実行", command=ok_clicked).pack(pady=10)
            ttk.Button(dialog, text="キャンセル", command=dialog.destroy).pack()
        
        select_column()

    def correlation_analysis(self):
        """相関分析"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが設定されていません。")
            return
        
        try:
            result = self.stats_engine.correlation_analysis()
            self.desc_results.delete(1.0, tk.END)
            self.desc_results.insert(tk.END, result)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("エラー", f"相関分析でエラーが発生しました: {e}")

    def t_test(self):
        """t検定"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが設定されていません。")
            return
        
        numeric_columns = list(self.data.select_dtypes(include=[np.number]).columns)
        if len(numeric_columns) < 1:
            messagebox.showwarning("警告", "数値データが見つかりません。")
            return
        
        def t_test_dialog():
            dialog = tk.Toplevel(self.root)
            dialog.title("t検定設定")
            dialog.geometry("400x300")
            
            # 検定タイプ
            ttk.Label(dialog, text="検定タイプ:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            test_type = tk.StringVar(value="one_sample")
            ttk.Radiobutton(dialog, text="1標本t検定", variable=test_type, value="one_sample").grid(row=1, column=0, sticky="w", padx=20)
            ttk.Radiobutton(dialog, text="対応なし2標本t検定", variable=test_type, value="independent").grid(row=2, column=0, sticky="w", padx=20)
            ttk.Radiobutton(dialog, text="対応あり2標本t検定", variable=test_type, value="paired").grid(row=3, column=0, sticky="w", padx=20)
            
            # 変数選択
            ttk.Label(dialog, text="変数1:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
            var1 = tk.StringVar(value=numeric_columns[0])
            var1_combo = ttk.Combobox(dialog, textvariable=var1, values=numeric_columns, state="readonly")
            var1_combo.grid(row=4, column=1, padx=5, pady=5)
            
            ttk.Label(dialog, text="変数2 (2標本の場合):").grid(row=5, column=0, sticky="w", padx=5, pady=5)
            var2 = tk.StringVar()
            var2_combo = ttk.Combobox(dialog, textvariable=var2, values=numeric_columns, state="readonly")
            var2_combo.grid(row=5, column=1, padx=5, pady=5)
            
            # 検定値（1標本の場合）
            ttk.Label(dialog, text="検定値 μ (1標本の場合):").grid(row=6, column=0, sticky="w", padx=5, pady=5)
            mu_var = tk.DoubleVar(value=0)
            mu_entry = ttk.Entry(dialog, textvariable=mu_var)
            mu_entry.grid(row=6, column=1, padx=5, pady=5)
            
            def execute_test():
                try:
                    test_type_val = test_type.get()
                    if test_type_val == "one_sample":
                        result = self.stats_engine.t_test(var1.get(), mu=mu_var.get())
                    elif test_type_val == "independent":
                        if not var2.get():
                            messagebox.showerror("エラー", "変数2を選択してください。")
                            return
                        result = self.stats_engine.t_test(var1.get(), var2.get(), paired=False)
                    else:  # paired
                        if not var2.get():
                            messagebox.showerror("エラー", "変数2を選択してください。")
                            return
                        result = self.stats_engine.t_test(var1.get(), var2.get(), paired=True)
                    
                    self.inf_results.delete(1.0, tk.END)
                    self.inf_results.insert(tk.END, result)
                    self.results_text.delete(1.0, tk.END)
                    self.results_text.insert(tk.END, result)
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("エラー", f"t検定でエラーが発生しました: {e}")
            
            ttk.Button(dialog, text="実行", command=execute_test).grid(row=7, column=0, pady=10)
            ttk.Button(dialog, text="キャンセル", command=dialog.destroy).grid(row=7, column=1, pady=10)
        
        t_test_dialog()

    def anova_test(self):
        """分散分析"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが設定されていません。")
            return
        
        numeric_columns = list(self.data.select_dtypes(include=[np.number]).columns)
        all_columns = list(self.data.columns)
        
        if len(numeric_columns) < 1:
            messagebox.showwarning("警告", "数値データが見つかりません。")
            return
        
        def anova_dialog():
            dialog = tk.Toplevel(self.root)
            dialog.title("分散分析設定")
            dialog.geometry("350x200")
            
            ttk.Label(dialog, text="従属変数:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            dep_var = tk.StringVar(value=numeric_columns[0])
            dep_combo = ttk.Combobox(dialog, textvariable=dep_var, values=numeric_columns, state="readonly")
            dep_combo.grid(row=0, column=1, padx=5, pady=5)
            
            ttk.Label(dialog, text="独立変数 (グループ変数):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            ind_var = tk.StringVar()
            ind_combo = ttk.Combobox(dialog, textvariable=ind_var, values=all_columns, state="readonly")
            ind_combo.grid(row=1, column=1, padx=5, pady=5)
            
            post_hoc_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(dialog, text="多重比較を実行", variable=post_hoc_var).grid(row=2, columnspan=2, pady=10)
            
            def execute_anova():
                try:
                    if not ind_var.get():
                        messagebox.showerror("エラー", "独立変数を選択してください。")
                        return
                    
                    result = self.stats_engine.anova_test(dep_var.get(), ind_var.get(), post_hoc_var.get())
                    self.inf_results.delete(1.0, tk.END)
                    self.inf_results.insert(tk.END, result)
                    self.results_text.delete(1.0, tk.END)
                    self.results_text.insert(tk.END, result)
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("エラー", f"分散分析でエラーが発生しました: {e}")
            
            ttk.Button(dialog, text="実行", command=execute_anova).grid(row=3, column=0, pady=10)
            ttk.Button(dialog, text="キャンセル", command=dialog.destroy).grid(row=3, column=1, pady=10)
        
        anova_dialog()

    def regression_analysis(self):
        """回帰分析"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが設定されていません。")
            return
        
        numeric_columns = list(self.data.select_dtypes(include=[np.number]).columns)
        if len(numeric_columns) < 2:
            messagebox.showwarning("警告", "回帰分析には少なくとも2つの数値変数が必要です。")
            return
        
        def regression_dialog():
            dialog = tk.Toplevel(self.root)
            dialog.title("回帰分析設定")
            dialog.geometry("400x300")
            
            ttk.Label(dialog, text="従属変数 (目的変数):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            dep_var = tk.StringVar(value=numeric_columns[0])
            dep_combo = ttk.Combobox(dialog, textvariable=dep_var, values=numeric_columns, state="readonly")
            dep_combo.grid(row=0, column=1, padx=5, pady=5)
            
            ttk.Label(dialog, text="独立変数 (説明変数):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            
            # 独立変数選択用のリストボックス
            ind_vars_frame = ttk.Frame(dialog)
            ind_vars_frame.grid(row=2, columnspan=2, padx=5, pady=5, sticky="ew")
            
            ind_vars_listbox = tk.Listbox(ind_vars_frame, selectmode=tk.MULTIPLE, height=6)
            ind_vars_scrollbar = ttk.Scrollbar(ind_vars_frame, orient=tk.VERTICAL, command=ind_vars_listbox.yview)
            ind_vars_listbox.configure(yscrollcommand=ind_vars_scrollbar.set)
            
            for col in numeric_columns:
                if col != dep_var.get():
                    ind_vars_listbox.insert(tk.END, col)
            
            ind_vars_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            ind_vars_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            standardize_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(dialog, text="標準化回帰分析", variable=standardize_var).grid(row=3, columnspan=2, pady=5)
            
            def execute_regression():
                try:
                    selected_indices = ind_vars_listbox.curselection()
                    if not selected_indices:
                        messagebox.showerror("エラー", "独立変数を選択してください。")
                        return
                    
                    selected_vars = [ind_vars_listbox.get(i) for i in selected_indices]
                    result = self.stats_engine.regression_analysis(dep_var.get(), selected_vars, standardize_var.get())
                    
                    self.inf_results.delete(1.0, tk.END)
                    self.inf_results.insert(tk.END, result)
                    self.results_text.delete(1.0, tk.END)
                    self.results_text.insert(tk.END, result)
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("エラー", f"回帰分析でエラーが発生しました: {e}")
            
            ttk.Button(dialog, text="実行", command=execute_regression).grid(row=4, column=0, pady=10)
            ttk.Button(dialog, text="キャンセル", command=dialog.destroy).grid(row=4, column=1, pady=10)
        
        regression_dialog()

    def chi_square_test(self):
        """カイ二乗検定"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが設定されていません。")
            return
        
        all_columns = list(self.data.columns)
        
        def chi_square_dialog():
            dialog = tk.Toplevel(self.root)
            dialog.title("カイ二乗検定設定")
            dialog.geometry("350x250")
            
            # 検定タイプ
            ttk.Label(dialog, text="検定タイプ:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            test_type = tk.StringVar(value="independence")
            ttk.Radiobutton(dialog, text="独立性検定", variable=test_type, value="independence").grid(row=1, column=0, sticky="w", padx=20)
            ttk.Radiobutton(dialog, text="適合度検定", variable=test_type, value="goodness").grid(row=2, column=0, sticky="w", padx=20)
            
            ttk.Label(dialog, text="変数1:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
            var1 = tk.StringVar()
            var1_combo = ttk.Combobox(dialog, textvariable=var1, values=all_columns, state="readonly")
            var1_combo.grid(row=3, column=1, padx=5, pady=5)
            
            ttk.Label(dialog, text="変数2 (独立性検定の場合):").grid(row=4, column=0, sticky="w", padx=5, pady=5)
            var2 = tk.StringVar()
            var2_combo = ttk.Combobox(dialog, textvariable=var2, values=all_columns, state="readonly")
            var2_combo.grid(row=4, column=1, padx=5, pady=5)
            
            def execute_chi_square():
                try:
                    if not var1.get():
                        messagebox.showerror("エラー", "変数1を選択してください。")
                        return
                    
                    if test_type.get() == "independence":
                        if not var2.get():
                            messagebox.showerror("エラー", "独立性検定には変数2も必要です。")
                            return
                        result = self.stats_engine.chi_square_test(var1.get(), var2.get())
                    else:
                        result = self.stats_engine.chi_square_test(var1.get())
                    
                    self.inf_results.delete(1.0, tk.END)
                    self.inf_results.insert(tk.END, result)
                    self.results_text.delete(1.0, tk.END)
                    self.results_text.insert(tk.END, result)
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("エラー", f"カイ二乗検定でエラーが発生しました: {e}")
            
            ttk.Button(dialog, text="実行", command=execute_chi_square).grid(row=5, column=0, pady=10)
            ttk.Button(dialog, text="キャンセル", command=dialog.destroy).grid(row=5, column=1, pady=10)
        
        chi_square_dialog()

    # グラフ作成メソッド
    def create_histogram(self):
        """ヒストグラム作成"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが設定されていません。")
            return
        
        numeric_columns = list(self.data.select_dtypes(include=[np.number]).columns)
        if not numeric_columns:
            messagebox.showwarning("警告", "数値データが見つかりません。")
            return
        
        def histogram_dialog():
            dialog = tk.Toplevel(self.root)
            dialog.title("ヒストグラム設定")
            dialog.geometry("300x200")
            
            ttk.Label(dialog, text="変数選択:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            var = tk.StringVar(value=numeric_columns[0])
            var_combo = ttk.Combobox(dialog, textvariable=var, values=numeric_columns, state="readonly")
            var_combo.grid(row=0, column=1, padx=5, pady=5)
            
            ttk.Label(dialog, text="ビン数:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            bins_var = tk.IntVar(value=30)
            bins_entry = ttk.Entry(dialog, textvariable=bins_var)
            bins_entry.grid(row=1, column=1, padx=5, pady=5)
            
            kde_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(dialog, text="KDE曲線を表示", variable=kde_var).grid(row=2, columnspan=2, pady=5)
            
            def create_plot():
                try:
                    fig, message = self.plot_engine.create_histogram(var.get(), bins_var.get(), kde=kde_var.get())
                    if fig:
                        self.display_plot(fig)
                        messagebox.showinfo("完了", message)
                    else:
                        messagebox.showerror("エラー", message)
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("エラー", f"ヒストグラム作成でエラーが発生しました: {e}")
            
            ttk.Button(dialog, text="作成", command=create_plot).grid(row=3, column=0, pady=10)
            ttk.Button(dialog, text="キャンセル", command=dialog.destroy).grid(row=3, column=1, pady=10)
        
        histogram_dialog()

    def create_scatter(self):
        """散布図作成"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが設定されていません。")
            return
        
        numeric_columns = list(self.data.select_dtypes(include=[np.number]).columns)
        if len(numeric_columns) < 2:
            messagebox.showwarning("警告", "散布図には少なくとも2つの数値変数が必要です。")
            return
        
        def scatter_dialog():
            dialog = tk.Toplevel(self.root)
            dialog.title("散布図設定")
            dialog.geometry("350x250")
            
            ttk.Label(dialog, text="X軸変数:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            x_var = tk.StringVar(value=numeric_columns[0])
            x_combo = ttk.Combobox(dialog, textvariable=x_var, values=numeric_columns, state="readonly")
            x_combo.grid(row=0, column=1, padx=5, pady=5)
            
            ttk.Label(dialog, text="Y軸変数:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            y_var = tk.StringVar(value=numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0])
            y_combo = ttk.Combobox(dialog, textvariable=y_var, values=numeric_columns, state="readonly")
            y_combo.grid(row=1, column=1, padx=5, pady=5)
            
            all_columns = list(self.data.columns)
            ttk.Label(dialog, text="色分け変数 (オプション):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
            hue_var = tk.StringVar()
            hue_combo = ttk.Combobox(dialog, textvariable=hue_var, values=[''] + all_columns, state="readonly")
            hue_combo.grid(row=2, column=1, padx=5, pady=5)
            
            def create_plot():
                try:
                    hue_col = hue_var.get() if hue_var.get() else None
                    fig, message = self.plot_engine.create_scatter(x_var.get(), y_var.get(), hue_column=hue_col)
                    if fig:
                        self.display_plot(fig)
                        messagebox.showinfo("完了", message)
                    else:
                        messagebox.showerror("エラー", message)
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("エラー", f"散布図作成でエラーが発生しました: {e}")
            
            ttk.Button(dialog, text="作成", command=create_plot).grid(row=3, column=0, pady=10)
            ttk.Button(dialog, text="キャンセル", command=dialog.destroy).grid(row=3, column=1, pady=10)
        
        scatter_dialog()

    def create_boxplot(self):
        """箱ひげ図作成"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが設定されていません。")
            return
        
        numeric_columns = list(self.data.select_dtypes(include=[np.number]).columns)
        if not numeric_columns:
            messagebox.showwarning("警告", "数値データが見つかりません。")
            return
        
        try:
            fig, message = self.plot_engine.create_boxplot()
            if fig:
                self.display_plot(fig)
                messagebox.showinfo("完了", message)
            else:
                messagebox.showerror("エラー", message)
        except Exception as e:
            messagebox.showerror("エラー", f"箱ひげ図作成でエラーが発生しました: {e}")

    def create_correlation_matrix(self):
        """相関行列ヒートマップ作成"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが設定されていません。")
            return
        
        try:
            fig, message = self.plot_engine.create_correlation_matrix()
            if fig:
                self.display_plot(fig)
                messagebox.showinfo("完了", message)
            else:
                messagebox.showerror("エラー", message)
        except Exception as e:
            messagebox.showerror("エラー", f"相関行列ヒートマップ作成でエラーが発生しました: {e}")

    def display_plot(self, fig):
        """プロット表示"""
        # 既存のプロットをクリア
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # 新しいプロットを表示
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # ツールバー追加
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()

    def save_results(self):
        """結果保存"""
        try:
            file_path = filedialog.asksaveasfilename(
                title="結果を保存",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.get("1.0", tk.END))
                messagebox.showinfo("完了", f"結果が保存されました: {file_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"結果の保存に失敗しました: {e}")

if __name__ == "__main__":
    import sys
    
    root = tk.Tk()
    app = HADStatisticsGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.emergency_save()
    finally:
        if app.auto_save_timer:
            app.auto_save_timer.cancel() 