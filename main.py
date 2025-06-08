#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Professional Python GUI Statistical Analysis Software
プロフェッショナル統計解析ソフトウェア

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
GitHub: @zapabob
License: MIT
"""

import sys
import os
import json
import pickle
import signal
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# GUI フレームワーク
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk

# 数値計算・データ処理
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import signal as scipy_signal

# 統計解析
import statsmodels.api as sm
import statsmodels.stats.api as sms
import pingouin as pg
from lifelines import KaplanMeierFitter, CoxPHFitter

# 機械学習
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

# 可視化
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ユーティリティ
from tqdm import tqdm
import psutil
import memory_profiler

# プロフェッショナル機能モジュール
try:
    from professional_utils import (
        professional_logger, performance_monitor, exception_handler,
        security_manager, database_manager
    )
    from professional_reports import report_generator
    PROFESSIONAL_FEATURES_AVAILABLE = True
    print("✅ プロフェッショナル機能が利用可能です")
except ImportError as e:
    PROFESSIONAL_FEATURES_AVAILABLE = False
    print(f"⚠️ プロフェッショナル機能が利用できません: {e}")

# AI統合モジュール
try:
    from ai_integration import ai_analyzer
    AI_INTEGRATION_AVAILABLE = True
except ImportError:
    AI_INTEGRATION_AVAILABLE = False
    print("⚠️ AI統合機能が利用できません（オプションライブラリが不足）")

# CUDA/GPU 検出
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    CUDA_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "None"
    DEVICE = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
except ImportError:
    CUDA_AVAILABLE = False
    GPU_NAME = "None"
    DEVICE = "cpu"

# 追加の機械学習ライブラリ
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# テーマ設定
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SessionManager:
    """セッション管理・電源断保護システム"""
    
    def __init__(self, app_instance):
        self.app = app_instance
        self.session_id = str(uuid.uuid4())
        self.backup_dir = Path("backup")
        self.checkpoint_dir = Path("checkpoints")
        self.auto_save_interval = 300  # 5分間隔
        self.backup_count = 10
        
        # ディレクトリ作成
        self.backup_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._emergency_save)
        signal.signal(signal.SIGTERM, self._emergency_save)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._emergency_save)
        
        # 自動保存タイマー開始
        self._start_auto_save()
        
        print(f"🛡️ セッション開始: {self.session_id}")
        print(f"💾 バックアップ: {self.backup_dir.absolute()}")
    
    def _start_auto_save(self):
        """自動保存タイマー開始"""
        self.auto_save_timer = threading.Timer(self.auto_save_interval, self._auto_save)
        self.auto_save_timer.daemon = True
        self.auto_save_timer.start()
    
    def _auto_save(self):
        """定期自動保存"""
        try:
            self.save_session()
            print(f"✅ 自動保存完了: {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"❌ 自動保存エラー: {e}")
        finally:
            self._start_auto_save()  # 次の自動保存を予約
    
    def _emergency_save(self, signum=None, frame=None):
        """緊急保存（異常終了時）"""
        print("\n🚨 緊急保存を実行中...")
        try:
            self.save_session(emergency=True)
            print("✅ 緊急保存完了")
        except Exception as e:
            print(f"❌ 緊急保存失敗: {e}")
        finally:
            sys.exit(0)
    
    def save_session(self, emergency=False):
        """セッション保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "emergency_" if emergency else "auto_"
        
        session_data = {
            'session_id': self.session_id,
            'timestamp': timestamp,
            'data': getattr(self.app, 'current_data', None),
            'analysis_results': getattr(self.app, 'analysis_results', {}),
            'settings': getattr(self.app, 'user_settings', {}),
            'gpu_info': {'available': CUDA_AVAILABLE, 'name': GPU_NAME}
        }
        
        # JSON保存
        json_file = self.checkpoint_dir / f"{prefix}session_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2, default=str)
        
        # Pickle保存（大容量データ用）
        pkl_file = self.checkpoint_dir / f"{prefix}session_{timestamp}.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(session_data, f)
        
        # バックアップローテーション
        self._rotate_backups()
        
        return json_file
    
    def _rotate_backups(self):
        """バックアップローテーション管理"""
        backup_files = sorted(self.checkpoint_dir.glob("auto_session_*.json"))
        if len(backup_files) > self.backup_count:
            for old_file in backup_files[:-self.backup_count]:
                old_file.unlink()
                pkl_file = old_file.with_suffix('.pkl')
                if pkl_file.exists():
                    pkl_file.unlink()
    
    def load_latest_session(self):
        """最新セッションの復旧"""
        backup_files = sorted(self.checkpoint_dir.glob("*.json"))
        if not backup_files:
            return None
        
        latest_file = backup_files[-1]
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"セッション復旧エラー: {e}")
            return None

class MLAnalysisWindow:
    """機械学習分析ウィンドウ"""
    
    def __init__(self, parent, data, variable_selection=None):
        self.parent = parent
        self.data = data
        self.variable_selection = variable_selection or {}
        self.window = None
        self.models = {}
        self.results = {}
    
    def show(self):
        """ウィンドウ表示"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("🤖 Machine Learning Analysis")
        self.window.geometry("800x600")
        
        # メインフレーム
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # タスク選択
        task_frame = ctk.CTkFrame(main_frame)
        task_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(task_frame, text="🎯 Machine Learning Task", font=("Arial", 16, "bold")).pack(pady=5)
        
        self.task_var = tk.StringVar(value="classification")
        task_radio1 = ctk.CTkRadioButton(task_frame, text="Classification", variable=self.task_var, value="classification")
        task_radio2 = ctk.CTkRadioButton(task_frame, text="Regression", variable=self.task_var, value="regression")
        task_radio1.pack(side="left", padx=20)
        task_radio2.pack(side="left", padx=20)
        
        # 特徴量・ターゲット選択
        feature_frame = ctk.CTkFrame(main_frame)
        feature_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(feature_frame, text="📊 Feature & Target Selection").pack(pady=5)
        
        # ターゲット選択
        target_frame = ctk.CTkFrame(feature_frame)
        target_frame.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(target_frame, text="Target Variable:").pack(side="left", padx=5)
        self.target_var = ctk.CTkComboBox(target_frame, values=list(self.data.columns))
        self.target_var.pack(side="left", padx=5)
        
        # アルゴリズム選択
        algo_frame = ctk.CTkFrame(main_frame)
        algo_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(algo_frame, text="🔧 Algorithm Selection").pack(pady=5)
        
        self.algo_var = ctk.CTkComboBox(algo_frame, values=[
            "Random Forest", "XGBoost", "LightGBM", "SVM", 
            "Gradient Boosting", "Decision Tree", "Neural Network"
        ])
        self.algo_var.pack(pady=5)
        self.algo_var.set("Random Forest")
        
        # GPU使用オプション
        if CUDA_AVAILABLE:
            gpu_frame = ctk.CTkFrame(main_frame)
            gpu_frame.pack(fill="x", padx=5, pady=5)
            
            self.use_gpu = tk.BooleanVar(value=True)
            gpu_check = ctk.CTkCheckBox(gpu_frame, text=f"⚡ Use GPU ({GPU_NAME})", variable=self.use_gpu)
            gpu_check.pack(pady=5)
        
        # 実行ボタン
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", padx=5, pady=10)
        
        run_btn = ctk.CTkButton(button_frame, text="🚀 Run Analysis", command=self.run_analysis)
        run_btn.pack(side="left", padx=5)
        
        optimize_btn = ctk.CTkButton(button_frame, text="⚙️ Hyperparameter Optimization", command=self.optimize_hyperparameters)
        optimize_btn.pack(side="left", padx=5)
        
        # 結果表示エリア
        self.result_text = ctk.CTkTextbox(main_frame, height=300)
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def run_analysis(self):
        """機械学習分析実行"""
        try:
            target_col = self.target_var.get()
            if not target_col:
                messagebox.showwarning("Warning", "Please select target variable")
                return
            
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", "🤖 Starting Machine Learning Analysis...\n\n")
            
            # 変数選択の活用
            if self.variable_selection.get('control_variables'):
                # 統制変数が選択されている場合はそれを使用
                feature_cols = self.variable_selection['control_variables']
                self.result_text.insert("end", f"🎯 Using selected control variables: {', '.join(feature_cols)}\n\n")
            else:
                # 従来の方法（数値型列を自動選択）
                feature_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
                if target_col in feature_cols:
                    feature_cols.remove(target_col)
                self.result_text.insert("end", f"📊 Using all numeric variables as features\n\n")
            
            if len(feature_cols) == 0:
                messagebox.showerror("Error", "No feature variables found")
                return
            
            X = self.data[feature_cols].fillna(0)
            y = self.data[target_col].fillna(0)
            
            # タスクタイプに応じた処理
            task_type = self.task_var.get()
            
            if task_type == "classification":
                # 分類タスク
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                results = self._run_classification(X, y)
            else:
                # 回帰タスク
                results = self._run_regression(X, y)
            
            # 結果表示
            self._display_results(results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.result_text.insert("end", f"\n❌ Error: {str(e)}")
    
    def _run_classification(self, X, y):
        """分類タスク実行"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # データ正規化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # アルゴリズム選択
        algo = self.algo_var.get()
        
        if algo == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif algo == "XGBoost":
            model = xgb.XGBClassifier(random_state=42)
        elif algo == "LightGBM":
            model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        elif algo == "SVM":
            model = SVC(random_state=42)
        elif algo == "Gradient Boosting":
            model = GradientBoostingClassifier(random_state=42)
        elif algo == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        else:  # Neural Network
            return self._run_neural_network_classification(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # 学習
        with tqdm(total=1, desc="Training") as pbar:
            model.fit(X_train_scaled, y_train)
            pbar.update(1)
        
        # 予測
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        # 評価指標計算
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # 特徴量重要度
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
        else:
            feature_importance = None
        
        return {
            'task': 'classification',
            'algorithm': algo,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': feature_importance,
            'model': model,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def _run_regression(self, X, y):
        """回帰タスク実行"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # データ正規化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # アルゴリズム選択
        algo = self.algo_var.get()
        
        if algo == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif algo == "XGBoost":
            model = xgb.XGBRegressor(random_state=42)
        elif algo == "LightGBM":
            model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        elif algo == "SVM":
            model = SVR()
        elif algo == "Gradient Boosting":
            model = GradientBoostingRegressor(random_state=42)
        elif algo == "Decision Tree":
            model = DecisionTreeRegressor(random_state=42)
        else:  # Neural Network
            return self._run_neural_network_regression(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # 学習
        with tqdm(total=1, desc="Training") as pbar:
            model.fit(X_train_scaled, y_train)
            pbar.update(1)
        
        # 予測
        y_pred = model.predict(X_test_scaled)
        
        # 評価指標計算
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 特徴量重要度
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
        else:
            feature_importance = None
        
        return {
            'task': 'regression',
            'algorithm': algo,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2_score': r2,
            'feature_importance': feature_importance,
            'model': model
        }
    
    def _run_neural_network_classification(self, X_train, X_test, y_train, y_test):
        """ニューラルネットワーク分類"""
        # PyTorchでニューラルネットワーク実装
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        
        class SimpleNN(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size//2)
                self.fc3 = nn.Linear(hidden_size//2, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        # データをTensorに変換
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        y_test_tensor = torch.LongTensor(y_test).to(DEVICE)
        
        # モデル作成
        model = SimpleNN(n_features, 128, n_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 学習
        model.train()
        epochs = 100
        with tqdm(total=epochs, desc="Neural Network Training") as pbar:
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                pbar.update(1)
        
        # 評価
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs.data, 1)
            
        y_pred = predicted.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()
        
        accuracy = accuracy_score(y_test_np, y_pred)
        precision = precision_score(y_test_np, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_np, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_np, y_pred, average='weighted', zero_division=0)
        
        return {
            'task': 'classification',
            'algorithm': 'Neural Network (PyTorch)',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': None,
            'model': model,
            'device': str(DEVICE),
            'confusion_matrix': confusion_matrix(y_test_np, y_pred)
        }
    
    def _run_neural_network_regression(self, X_train, X_test, y_train, y_test):
        """ニューラルネットワーク回帰"""
        # 実装は同様の構造で回帰用に調整
        n_features = X_train.shape[1]
        
        class SimpleRegressionNN(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(SimpleRegressionNN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size//2)
                self.fc3 = nn.Linear(hidden_size//2, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        # 同様の学習プロセス
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_train_tensor = torch.FloatTensor(y_train.values.reshape(-1, 1)).to(DEVICE)
        y_test_tensor = torch.FloatTensor(y_test.values.reshape(-1, 1)).to(DEVICE)
        
        model = SimpleRegressionNN(n_features, 128).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 学習
        model.train()
        epochs = 100
        with tqdm(total=epochs, desc="Neural Network Training") as pbar:
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                pbar.update(1)
        
        # 評価
        model.eval()
        with torch.no_grad():
            y_pred_tensor = model(X_test_tensor)
            
        y_pred = y_pred_tensor.cpu().numpy().flatten()
        y_test_np = y_test_tensor.cpu().numpy().flatten()
        
        mse = mean_squared_error(y_test_np, y_pred)
        mae = mean_absolute_error(y_test_np, y_pred)
        r2 = r2_score(y_test_np, y_pred)
        
        return {
            'task': 'regression',
            'algorithm': 'Neural Network (PyTorch)',
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2_score': r2,
            'feature_importance': None,
            'model': model,
            'device': str(DEVICE)
        }
    
    def _display_results(self, results):
        """結果表示"""
        result_text = f"\n🎯 Machine Learning Results\n"
        result_text += "=" * 50 + "\n\n"
        result_text += f"Algorithm: {results['algorithm']}\n"
        result_text += f"Task: {results['task'].capitalize()}\n"
        
        if 'device' in results:
            result_text += f"Device: {results['device']}\n"
        
        result_text += "\n📊 Performance Metrics:\n"
        
        if results['task'] == 'classification':
            result_text += f"• Accuracy: {results['accuracy']:.4f}\n"
            result_text += f"• Precision: {results['precision']:.4f}\n"
            result_text += f"• Recall: {results['recall']:.4f}\n"
            result_text += f"• F1-Score: {results['f1_score']:.4f}\n"
        else:
            result_text += f"• R² Score: {results['r2_score']:.4f}\n"
            result_text += f"• MSE: {results['mse']:.4f}\n"
            result_text += f"• RMSE: {results['rmse']:.4f}\n"
            result_text += f"• MAE: {results['mae']:.4f}\n"
        
        if results['feature_importance']:
            result_text += "\n🔍 Feature Importance (Top 5):\n"
            sorted_features = sorted(results['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in sorted_features:
                result_text += f"• {feature}: {importance:.4f}\n"
        
        self.result_text.insert("end", result_text)
        
        # モデル保存
        self.models[results['algorithm']] = results['model']
        self.results[results['algorithm']] = results
    
    def optimize_hyperparameters(self):
        """ハイパーパラメータ最適化"""
        messagebox.showinfo("Info", "⚙️ Hyperparameter optimization starting...")
        # Optuna等を使った最適化実装は次のステップで行います

class DeepLearningWindow:
    """深層学習分析ウィンドウ"""
    
    def __init__(self, parent, data, variable_selection=None):
        self.parent = parent
        self.data = data
        self.variable_selection = variable_selection or {}
        self.window = None
    
    def show(self):
        """ウィンドウ表示"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("🧠 Deep Learning Analysis")
        self.window.geometry("800x600")
        
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 深層学習メニュー
        ctk.CTkLabel(main_frame, text="🧠 Deep Learning Models", font=("Arial", 16, "bold")).pack(pady=10)
        
        # 各種深層学習アルゴリズムボタン
        models = [
            ("🔍 Autoencoder", self.run_autoencoder),
            ("📈 LSTM Time Series", self.run_lstm),
            ("🎯 Advanced CNN", self.run_cnn),
            ("🔮 Transformer", self.run_transformer)
        ]
        
        for name, command in models:
            btn = ctk.CTkButton(main_frame, text=name, command=command, width=200, height=40)
            btn.pack(pady=5)
        
        # 結果表示エリア
        self.result_text = ctk.CTkTextbox(main_frame, height=300)
        self.result_text.pack(fill="both", expand=True, padx=5, pady=10)
    
    def run_autoencoder(self):
        messagebox.showinfo("Info", "🔍 Autoencoder implementation coming soon!")
    
    def run_lstm(self):
        messagebox.showinfo("Info", "📈 LSTM implementation coming soon!")
    
    def run_cnn(self):
        messagebox.showinfo("Info", "🎯 CNN implementation coming soon!")
    
    def run_transformer(self):
        messagebox.showinfo("Info", "🔮 Transformer implementation coming soon!")

class VariableSelectionWindow:
    """変数選択ウィンドウ（統制変数・目的変数・剰余変数）"""
    
    def __init__(self, parent, data, callback=None):
        self.parent = parent
        self.data = data
        self.callback = callback
        self.columns = list(data.columns)
        
        # 変数分類
        self.control_variables = []  # 統制変数
        self.target_variables = []   # 目的変数
        self.residual_variables = [] # 剰余変数
        
        self.window = None
        
    def show(self):
        """変数選択ウィンドウ表示"""
        self.window = ctk.CTkToplevel(self.parent)
        self.window.title("🎯 Variable Selection - 変数選択")
        self.window.geometry("800x700")
        self.window.grab_set()  # モーダルウィンドウ化
        
        # メインフレーム
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # タイトル
        title_label = ctk.CTkLabel(main_frame, text="🎯 変数分類設定", font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 説明文
        info_frame = ctk.CTkFrame(main_frame)
        info_frame.pack(fill="x", pady=(0, 20))
        
        info_text = """
📊 変数分類について:
• 統制変数: 分析で制御したい説明変数（独立変数）
• 目的変数: 予測・説明したい従属変数（アウトカム）
• 剰余変数: 分析に含めない変数（除外対象）

⚠️ 注意: 機械学習では統制変数を特徴量、目的変数をターゲットとして使用します
        """
        info_label = ctk.CTkLabel(info_frame, text=info_text, justify="left")
        info_label.pack(padx=10, pady=10)
        
        # 3列レイアウト
        columns_frame = ctk.CTkFrame(main_frame)
        columns_frame.pack(fill="both", expand=True, pady=(0, 20))
        
        # 利用可能な変数リスト
        available_frame = ctk.CTkFrame(columns_frame)
        available_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ctk.CTkLabel(available_frame, text="📋 Available Variables", font=("Arial", 14, "bold")).pack(pady=5)
        self.available_listbox = tk.Listbox(available_frame, selectmode=tk.MULTIPLE, height=15)
        self.available_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        for col in self.columns:
            self.available_listbox.insert(tk.END, col)
        
        # 統制変数リスト
        control_frame = ctk.CTkFrame(columns_frame)
        control_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        ctk.CTkLabel(control_frame, text="🎛️ Control Variables\n(統制変数)", font=("Arial", 14, "bold")).pack(pady=5)
        self.control_listbox = tk.Listbox(control_frame, height=15)
        self.control_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 目的変数リスト
        target_frame = ctk.CTkFrame(columns_frame)
        target_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        ctk.CTkLabel(target_frame, text="🎯 Target Variables\n(目的変数)", font=("Arial", 14, "bold")).pack(pady=5)
        self.target_listbox = tk.Listbox(target_frame, height=15)
        self.target_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 剰余変数リスト
        residual_frame = ctk.CTkFrame(columns_frame)
        residual_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        ctk.CTkLabel(residual_frame, text="📦 Residual Variables\n(剰余変数)", font=("Arial", 14, "bold")).pack(pady=5)
        self.residual_listbox = tk.Listbox(residual_frame, height=15)
        self.residual_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 操作ボタン
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", pady=(0, 20))
        
        # 変数移動ボタン
        move_frame = ctk.CTkFrame(button_frame)
        move_frame.pack(pady=10)
        
        ctk.CTkButton(move_frame, text="→ Control", command=self._move_to_control, width=100).pack(side="left", padx=2)
        ctk.CTkButton(move_frame, text="→ Target", command=self._move_to_target, width=100).pack(side="left", padx=2)
        ctk.CTkButton(move_frame, text="→ Residual", command=self._move_to_residual, width=100).pack(side="left", padx=2)
        ctk.CTkButton(move_frame, text="← Remove", command=self._remove_from_lists, width=100).pack(side="left", padx=2)
        
        # リセット・自動分類ボタン
        auto_frame = ctk.CTkFrame(button_frame)
        auto_frame.pack(pady=5)
        
        ctk.CTkButton(auto_frame, text="🔄 Reset All", command=self._reset_all, width=120).pack(side="left", padx=5)
        ctk.CTkButton(auto_frame, text="🤖 Auto Classify", command=self._auto_classify, width=120).pack(side="left", padx=5)
        ctk.CTkButton(auto_frame, text="📊 Show Summary", command=self._show_summary, width=120).pack(side="left", padx=5)
        
        # 確定・キャンセルボタン
        action_frame = ctk.CTkFrame(main_frame)
        action_frame.pack(fill="x")
        
        ctk.CTkButton(action_frame, text="✅ Apply Selection", command=self._apply_selection, 
                     fg_color="green", width=150, height=40).pack(side="left", padx=10)
        ctk.CTkButton(action_frame, text="❌ Cancel", command=self._cancel, 
                     fg_color="red", width=100, height=40).pack(side="right", padx=10)
    
    def _move_to_control(self):
        """統制変数に移動"""
        selected = self.available_listbox.curselection()
        for i in reversed(selected):
            var = self.available_listbox.get(i)
            self.control_listbox.insert(tk.END, var)
            self.available_listbox.delete(i)
    
    def _move_to_target(self):
        """目的変数に移動"""
        selected = self.available_listbox.curselection()
        for i in reversed(selected):
            var = self.available_listbox.get(i)
            self.target_listbox.insert(tk.END, var)
            self.available_listbox.delete(i)
    
    def _move_to_residual(self):
        """剰余変数に移動"""
        selected = self.available_listbox.curselection()
        for i in reversed(selected):
            var = self.available_listbox.get(i)
            self.residual_listbox.insert(tk.END, var)
            self.available_listbox.delete(i)
    
    def _remove_from_lists(self):
        """リストから変数を削除して利用可能リストに戻す"""
        # 統制変数リストから削除
        selected = self.control_listbox.curselection()
        for i in reversed(selected):
            var = self.control_listbox.get(i)
            self.available_listbox.insert(tk.END, var)
            self.control_listbox.delete(i)
        
        # 目的変数リストから削除
        selected = self.target_listbox.curselection()
        for i in reversed(selected):
            var = self.target_listbox.get(i)
            self.available_listbox.insert(tk.END, var)
            self.target_listbox.delete(i)
        
        # 剰余変数リストから削除
        selected = self.residual_listbox.curselection()
        for i in reversed(selected):
            var = self.residual_listbox.get(i)
            self.available_listbox.insert(tk.END, var)
            self.residual_listbox.delete(i)
    
    def _reset_all(self):
        """全てリセット"""
        self.available_listbox.delete(0, tk.END)
        self.control_listbox.delete(0, tk.END)
        self.target_listbox.delete(0, tk.END)
        self.residual_listbox.delete(0, tk.END)
        
        for col in self.columns:
            self.available_listbox.insert(tk.END, col)
    
    def _auto_classify(self):
        """自動分類（ヒューリスティック）"""
        self._reset_all()
        
        # 数値型変数を統制変数、文字列・カテゴリ型を剰余変数に分類
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 'id', 'index' などは剰余変数に
        id_patterns = ['id', 'index', 'idx', 'key', '番号', 'no', 'num']
        
        self.available_listbox.delete(0, tk.END)
        
        for col in self.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in id_patterns):
                self.residual_listbox.insert(tk.END, col)
            elif col in numeric_cols and col not in categorical_cols:
                if 'target' in col_lower or 'outcome' in col_lower or 'result' in col_lower:
                    self.target_listbox.insert(tk.END, col)
                else:
                    self.control_listbox.insert(tk.END, col)
            else:
                self.residual_listbox.insert(tk.END, col)
    
    def _show_summary(self):
        """変数分類サマリー表示"""
        control_vars = [self.control_listbox.get(i) for i in range(self.control_listbox.size())]
        target_vars = [self.target_listbox.get(i) for i in range(self.target_listbox.size())]
        residual_vars = [self.residual_listbox.get(i) for i in range(self.residual_listbox.size())]
        
        summary = f"""
📊 Variable Classification Summary
================================

🎛️ Control Variables ({len(control_vars)}):
{', '.join(control_vars) if control_vars else 'None'}

🎯 Target Variables ({len(target_vars)}):
{', '.join(target_vars) if target_vars else 'None'}

📦 Residual Variables ({len(residual_vars)}):
{', '.join(residual_vars) if residual_vars else 'None'}

📈 Data Types:
{self.data.dtypes.to_string()}
        """
        
        messagebox.showinfo("Variable Summary", summary)
    
    def _apply_selection(self):
        """選択を適用"""
        self.control_variables = [self.control_listbox.get(i) for i in range(self.control_listbox.size())]
        self.target_variables = [self.target_listbox.get(i) for i in range(self.target_listbox.size())]
        self.residual_variables = [self.residual_listbox.get(i) for i in range(self.residual_listbox.size())]
        
        # バリデーション
        if not self.control_variables and not self.target_variables:
            messagebox.showerror("Error", "Please select at least one control or target variable!")
            return
        
        # コールバック実行
        if self.callback:
            result = {
                'control_variables': self.control_variables,
                'target_variables': self.target_variables,
                'residual_variables': self.residual_variables,
                'data_subset': self._create_data_subset()
            }
            self.callback(result)
        
        self.window.destroy()
    
    def _create_data_subset(self):
        """選択された変数でデータサブセットを作成"""
        selected_vars = self.control_variables + self.target_variables
        if selected_vars:
            return self.data[selected_vars].copy()
        return self.data.copy()
    
    def _cancel(self):
        """キャンセル"""
        self.window.destroy()

class AIAnalysisWindow:
    """AI統計分析ウィンドウ"""
    
    def __init__(self, parent, data):
        self.parent = parent
        self.data = data
        self.window = None
        
    def show(self):
        """AI分析ウィンドウ表示"""
        if not AI_INTEGRATION_AVAILABLE:
            messagebox.showerror("Error", "AI統合機能が利用できません。必要なライブラリをインストールしてください。")
            return
            
        self.window = ctk.CTkToplevel(self.parent)
        self.window.title("🤖 AI Statistical Analysis - AI統計分析")
        self.window.geometry("1000x800")
        self.window.grab_set()
        
        # メインフレーム
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # タイトル
        title_label = ctk.CTkLabel(main_frame, text="🤖 AI Statistical Analysis", font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 20))
        
        # タブビュー
        tabview = ctk.CTkTabview(main_frame)
        tabview.pack(fill="both", expand=True)
        
        # 自然言語分析タブ
        nlp_tab = tabview.add("Natural Language")
        self._create_nlp_tab(nlp_tab)
        
        # 画像分析タブ
        image_tab = tabview.add("Image Analysis")
        self._create_image_tab(image_tab)
        
        # 履歴タブ
        history_tab = tabview.add("Analysis History")
        self._create_history_tab(history_tab)
    
    def _create_nlp_tab(self, parent):
        """自然言語分析タブ作成"""
        # 説明文
        info_label = ctk.CTkLabel(parent, text="🗣️ 自然言語で統計分析を指示してください", font=("Arial", 14, "bold"))
        info_label.pack(pady=10)
        
        # 入力フレーム
        input_frame = ctk.CTkFrame(parent)
        input_frame.pack(fill="x", padx=10, pady=10)
        
        # クエリ入力
        ctk.CTkLabel(input_frame, text="分析要求:").pack(anchor="w", padx=10, pady=(10, 5))
        self.query_text = ctk.CTkTextbox(input_frame, height=100)
        self.query_text.pack(fill="x", padx=10, pady=(0, 10))
        
        # サンプルクエリボタン
        sample_frame = ctk.CTkFrame(input_frame)
        sample_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        samples = [
            "データの相関分析を行って",
            "線形回帰分析をして結果を可視化",
            "記述統計を表示して",
            "異常値の検出と除去",
            "クラスター分析を実行"
        ]
        
        for i, sample in enumerate(samples):
            btn = ctk.CTkButton(sample_frame, text=sample, width=180, height=30,
                               command=lambda s=sample: self.query_text.insert("1.0", s))
            btn.grid(row=i//3, column=i%3, padx=5, pady=5)
        
        # 実行ボタン
        ctk.CTkButton(input_frame, text="🚀 AI Analysis", command=self._run_nlp_analysis,
                     fg_color="green", width=200, height=40).pack(pady=10)
        
        # 結果表示
        ctk.CTkLabel(parent, text="AI Response:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(20, 5))
        self.nlp_result_text = ctk.CTkTextbox(parent, height=300)
        self.nlp_result_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # コード実行ボタン
        ctk.CTkButton(parent, text="🔧 Execute Generated Code", command=self._execute_nlp_code,
                     fg_color="blue", width=200, height=35).pack(pady=10)
    
    def _create_image_tab(self, parent):
        """画像分析タブ作成"""
        # 説明文
        info_label = ctk.CTkLabel(parent, text="📷 画像からデータを抽出・分析", font=("Arial", 14, "bold"))
        info_label.pack(pady=10)
        
        # ファイル選択フレーム
        file_frame = ctk.CTkFrame(parent)
        file_frame.pack(fill="x", padx=10, pady=10)
        
        self.image_path_var = tk.StringVar()
        ctk.CTkLabel(file_frame, text="画像ファイル:").pack(anchor="w", padx=10, pady=(10, 5))
        
        path_frame = ctk.CTkFrame(file_frame)
        path_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.image_path_entry = ctk.CTkEntry(path_frame, textvariable=self.image_path_var, width=400)
        self.image_path_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        ctk.CTkButton(path_frame, text="📁 Browse", command=self._browse_image, width=100).pack(side="right")
        
        # コンテキスト入力
        ctk.CTkLabel(file_frame, text="追加コンテキスト (オプション):").pack(anchor="w", padx=10, pady=(10, 5))
        self.context_entry = ctk.CTkEntry(file_frame, placeholder_text="例: この表は売上データです")
        self.context_entry.pack(fill="x", padx=10, pady=(0, 10))
        
        # 実行ボタン
        ctk.CTkButton(file_frame, text="🔍 Analyze Image", command=self._run_image_analysis,
                     fg_color="orange", width=200, height=40).pack(pady=10)
        
        # 結果表示
        ctk.CTkLabel(parent, text="Image Analysis Result:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(20, 5))
        self.image_result_text = ctk.CTkTextbox(parent, height=300)
        self.image_result_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # コード実行ボタン
        ctk.CTkButton(parent, text="🔧 Execute Generated Code", command=self._execute_image_code,
                     fg_color="blue", width=200, height=35).pack(pady=10)
    
    def _create_history_tab(self, parent):
        """履歴タブ作成"""
        ctk.CTkLabel(parent, text="📚 Analysis History", font=("Arial", 14, "bold")).pack(pady=10)
        
        # 履歴リスト
        self.history_listbox = tk.Listbox(parent, height=20)
        self.history_listbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 履歴操作ボタン
        history_btn_frame = ctk.CTkFrame(parent)
        history_btn_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(history_btn_frame, text="🔄 Refresh", command=self._refresh_history, width=100).pack(side="left", padx=5)
        ctk.CTkButton(history_btn_frame, text="📤 Export", command=self._export_history, width=100).pack(side="left", padx=5)
        ctk.CTkButton(history_btn_frame, text="🗑️ Clear", command=self._clear_history, width=100).pack(side="left", padx=5)
    
    def _browse_image(self):
        """画像ファイル選択"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.image_path_var.set(file_path)
    
    def _run_nlp_analysis(self):
        """自然言語分析実行"""
        query = self.query_text.get("1.0", "end-1c").strip()
        if not query:
            messagebox.showwarning("Warning", "分析要求を入力してください")
            return
        
        self.nlp_result_text.delete("1.0", "end")
        self.nlp_result_text.insert("1.0", "🤖 AI分析中... しばらくお待ちください...")
        
        # 非同期実行
        import threading
        thread = threading.Thread(target=self._nlp_analysis_thread, args=(query,))
        thread.daemon = True
        thread.start()
    
    def _nlp_analysis_thread(self, query):
        """自然言語分析スレッド"""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(ai_analyzer.analyze_natural_language_query(query, self.data))
            
            # UI更新（メインスレッドで実行）
            self.window.after(0, lambda: self._update_nlp_result(result))
            
        except Exception as e:
            error_msg = f"❌ エラー: {str(e)}"
            self.window.after(0, lambda: self.nlp_result_text.delete("1.0", "end"))
            self.window.after(0, lambda: self.nlp_result_text.insert("1.0", error_msg))
    
    def _update_nlp_result(self, result):
        """自然言語分析結果更新"""
        self.nlp_result_text.delete("1.0", "end")
        
        if result["success"]:
            display_text = f"""🤖 AI Provider: {result.get('provider', 'Unknown')} ({result.get('model', 'Unknown')})

📝 AI Response:
{result['ai_response']}

🐍 Generated Python Code:
{result['python_code']}
"""
            self.nlp_result_text.insert("1.0", display_text)
            self.last_nlp_code = result['python_code']
        else:
            self.nlp_result_text.insert("1.0", f"❌ エラー: {result['error']}")
    
    def _run_image_analysis(self):
        """画像分析実行"""
        image_path = self.image_path_var.get().strip()
        if not image_path:
            messagebox.showwarning("Warning", "画像ファイルを選択してください")
            return
        
        if not Path(image_path).exists():
            messagebox.showerror("Error", "指定された画像ファイルが見つかりません")
            return
        
        context = self.context_entry.get().strip()
        
        self.image_result_text.delete("1.0", "end")
        self.image_result_text.insert("1.0", "🔍 画像分析中... しばらくお待ちください...")
        
        # 非同期実行
        import threading
        thread = threading.Thread(target=self._image_analysis_thread, args=(image_path, context))
        thread.daemon = True
        thread.start()
    
    def _image_analysis_thread(self, image_path, context):
        """画像分析スレッド"""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(ai_analyzer.analyze_image_data(image_path, context))
            
            # UI更新（メインスレッドで実行）
            self.window.after(0, lambda: self._update_image_result(result))
            
        except Exception as e:
            error_msg = f"❌ エラー: {str(e)}"
            self.window.after(0, lambda: self.image_result_text.delete("1.0", "end"))
            self.window.after(0, lambda: self.image_result_text.insert("1.0", error_msg))
    
    def _update_image_result(self, result):
        """画像分析結果更新"""
        self.image_result_text.delete("1.0", "end")
        
        if result["success"]:
            display_text = f"""🤖 AI Provider: {result.get('provider', 'Unknown')} ({result.get('model', 'Unknown')})

📝 AI Response:
{result['ai_response']}

🔤 OCR Text:
{result.get('ocr_text', 'なし')}

🐍 Generated Python Code:
{result['python_code']}
"""
            self.image_result_text.insert("1.0", display_text)
            self.last_image_code = result['python_code']
        else:
            self.image_result_text.insert("1.0", f"❌ エラー: {result['error']}")
    
    def _execute_nlp_code(self):
        """自然言語分析コード実行"""
        if hasattr(self, 'last_nlp_code') and self.last_nlp_code:
            self._execute_code(self.last_nlp_code)
        else:
            messagebox.showwarning("Warning", "実行するコードがありません")
    
    def _execute_image_code(self):
        """画像分析コード実行"""
        if hasattr(self, 'last_image_code') and self.last_image_code:
            self._execute_code(self.last_image_code)
        else:
            messagebox.showwarning("Warning", "実行するコードがありません")
    
    def _execute_code(self, code):
        """コード実行"""
        try:
            result = ai_analyzer.execute_generated_code(code, self.data)
            
            if result["success"]:
                messagebox.showinfo("Success", "コードが正常に実行されました！")
            else:
                messagebox.showerror("Execution Error", f"実行エラー:\n{result['error']}")
        except Exception as e:
            messagebox.showerror("Error", f"実行中にエラーが発生しました:\n{str(e)}")
    
    def _refresh_history(self):
        """履歴更新"""
        self.history_listbox.delete(0, tk.END)
        for i, record in enumerate(ai_analyzer.analysis_history):
            timestamp = record.get('timestamp', 'Unknown')
            query = record.get('query', 'No query')[:50]
            self.history_listbox.insert(tk.END, f"{i+1}. [{timestamp}] {query}...")
    
    def _export_history(self):
        """履歴エクスポート"""
        if not ai_analyzer.analysis_history:
            messagebox.showinfo("Info", "エクスポートする履歴がありません")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Analysis History",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(ai_analyzer.analysis_history, f, ensure_ascii=False, indent=2, default=str)
                messagebox.showinfo("Success", f"履歴をエクスポートしました:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"エクスポートエラー:\n{str(e)}")
    
    def _clear_history(self):
        """履歴クリア"""
        if messagebox.askyesno("Confirm", "すべての履歴を削除しますか？"):
            ai_analyzer.analysis_history.clear()
            self.history_listbox.delete(0, tk.END)
            messagebox.showinfo("Success", "履歴をクリアしました")

class StatisticalAnalysisGUI:
    """プロフェッショナル統計解析GUI"""
    
    def __init__(self):
        # セッション管理初期化
        self.session_manager = SessionManager(self)
        
        # プロフェッショナル機能初期化
        try:
            from professional_utils import professional_logger, performance_monitor
            from professional_reports import professional_report_generator
            from advanced_statistics import advanced_analyzer
            from parallel_optimization import parallel_processor, optimized_stats, memory_manager
            from survival_analysis import complete_survival_analyzer
            from bayesian_analysis import deep_bayesian_analyzer
            from ml_pipeline_automation import ml_pipeline_automator
            
            print("✅ 完全プロフェッショナル機能が利用可能です")
            print("🧠 高度統計解析機能が利用可能です")
            print("🚀 並列処理・最適化機能が利用可能です")
            print("⏱️ 生存解析機能が利用可能です")
            print("🎲 ベイズ統計解析機能が利用可能です")
            print("🤖 機械学習パイプライン自動化が利用可能です")
            
            self.professional_features = True
            self.advanced_features = True
            self.advanced_analyzer = advanced_analyzer
            self.parallel_processor = parallel_processor
            self.memory_manager = memory_manager
            self.survival_analyzer = complete_survival_analyzer
            self.bayesian_analyzer = deep_bayesian_analyzer
            self.ml_automator = ml_pipeline_automator
            
        except ImportError as e:
            print(f"⚠️ プロフェッショナル機能が一部利用できません: {e}")
            self.professional_features = False
            self.advanced_features = False
        
        # データ管理
        self.current_data = None
        self.analysis_results = {}
        self.user_settings = {}
        
        # 変数選択状態
        self.control_variables = []  # 統制変数
        self.target_variables = []   # 目的変数
        self.residual_variables = [] # 剰余変数
        self.variable_selection_applied = False
        
        # GUI初期化
        self._initialize_gui()
        
        # 復旧チェック
        self._check_recovery()
    
    def _initialize_gui(self):
        """GUI初期化"""
        self.root = ctk.CTk()
        self.root.title("🔬 Professional Statistical Analysis Software")
        self.root.geometry("1400x900")
        
        # メニューバー
        self._create_menu()
        
        # メインフレーム
        self._create_main_frame()
        
        # ステータスバー
        self._create_status_bar()
        
        # 初期状態表示
        self._update_status(f"Ready | Session: {self.session_manager.session_id[:8]}...")
    
    def _create_menu(self):
        """メニューバー作成"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # ファイルメニュー
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Data...", command=self.load_data)
        file_menu.add_command(label="Save Session", command=self.save_session)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # 解析メニュー
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Variable Selection", command=self.variable_selection)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="🤖 AI Analysis", command=self.ai_analysis)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Descriptive Statistics", command=self.descriptive_stats)
        analysis_menu.add_command(label="Hypothesis Testing", command=self.hypothesis_testing)
        analysis_menu.add_command(label="Regression Analysis", command=self.regression_analysis)
        analysis_menu.add_command(label="Machine Learning", command=self.machine_learning)
        analysis_menu.add_command(label="Deep Learning", command=self.deep_learning)
        
        # 高度解析メニュー（プロフェッショナル機能が利用可能な場合）
        if hasattr(self, 'advanced_features') and self.advanced_features:
            analysis_menu.add_separator()
            advanced_menu = tk.Menu(analysis_menu, tearoff=0)
            analysis_menu.add_cascade(label="🧠 Advanced Analysis", menu=advanced_menu)
            advanced_menu.add_command(label="🔬 Multivariate Analysis", command=self.multivariate_analysis)
            advanced_menu.add_command(label="📈 Time Series Analysis", command=self.time_series_analysis)
            advanced_menu.add_command(label="⏱️ Survival Analysis", command=self.survival_analysis)
            advanced_menu.add_command(label="🎲 Bayesian Analysis", command=self.bayesian_analysis)
            advanced_menu.add_command(label="📊 Comprehensive EDA", command=self.comprehensive_eda)
        
        # ツールメニュー
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Data Quality Check", command=self.data_quality_check)
        tools_menu.add_separator()
        if PROFESSIONAL_FEATURES_AVAILABLE:
            tools_menu.add_command(label="📋 Generate Report", command=self.generate_professional_report)
            tools_menu.add_command(label="📊 Performance Report", command=self.show_performance_report)
            tools_menu.add_separator()
        tools_menu.add_command(label="GPU Status", command=self.show_gpu_status)
        tools_menu.add_command(label="Memory Usage", command=self.show_memory_usage)
    
    def _create_main_frame(self):
        """メインフレーム作成"""
        # サイドパネル
        self.sidebar = ctk.CTkFrame(self.root, width=300)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        
        # データ情報パネル
        self.data_info_frame = ctk.CTkFrame(self.sidebar)
        self.data_info_frame.pack(fill="x", padx=5, pady=5)
        
        self.data_info_label = ctk.CTkLabel(self.data_info_frame, text="📊 Data Information")
        self.data_info_label.pack(pady=5)
        
        self.data_text = ctk.CTkTextbox(self.data_info_frame, height=150)
        self.data_text.pack(fill="x", padx=5, pady=5)
        
        # 操作パネル
        self.control_frame = ctk.CTkFrame(self.sidebar)
        self.control_frame.pack(fill="x", padx=5, pady=5)
        
        # ボタン配置
        self.load_btn = ctk.CTkButton(self.control_frame, text="📁 Load Data", command=self.load_data)
        self.load_btn.pack(fill="x", padx=5, pady=2)
        
        self.var_select_btn = ctk.CTkButton(self.control_frame, text="🎯 Variable Selection", command=self.variable_selection)
        self.var_select_btn.pack(fill="x", padx=5, pady=2)
        
        self.ai_btn = ctk.CTkButton(self.control_frame, text="🤖 AI Analysis", command=self.ai_analysis, fg_color="purple")
        self.ai_btn.pack(fill="x", padx=5, pady=2)
        
        self.desc_btn = ctk.CTkButton(self.control_frame, text="📈 Descriptive Stats", command=self.descriptive_stats)
        self.desc_btn.pack(fill="x", padx=5, pady=2)
        
        self.test_btn = ctk.CTkButton(self.control_frame, text="🔬 Hypothesis Test", command=self.hypothesis_testing)
        self.test_btn.pack(fill="x", padx=5, pady=2)
        
        self.reg_btn = ctk.CTkButton(self.control_frame, text="📊 Regression", command=self.regression_analysis)
        self.reg_btn.pack(fill="x", padx=5, pady=2)
        
        self.ml_btn = ctk.CTkButton(self.control_frame, text="🤖 Machine Learning", command=self.machine_learning)
        self.ml_btn.pack(fill="x", padx=5, pady=2)
        
        self.dl_btn = ctk.CTkButton(self.control_frame, text="🧠 Deep Learning", command=self.deep_learning)
        self.dl_btn.pack(fill="x", padx=5, pady=2)
        
        # メイン作業エリア
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # 結果表示エリア
        self.result_text = ctk.CTkTextbox(self.main_frame, font=("Consolas", 12))
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 初期メッセージ
        welcome_msg = f"""
🔬 Professional Statistical Analysis Software
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🛡️ Session ID: {self.session_manager.session_id}
⚡ GPU Status: {'✅ Available' if CUDA_AVAILABLE else '❌ Not Available'}
🎯 GPU Device: {GPU_NAME}
💾 Auto-save: Every 5 minutes
🔄 Backup: Automatic rotation (10 files)

📋 Quick Start:
1. Load your dataset using 'Load Data' button
2. Explore with 'Descriptive Stats'
3. Run hypothesis tests or regression analysis
4. Apply machine learning algorithms
5. Export results and visualizations

Ready for professional statistical analysis! 🚀
        """
        self.result_text.insert("1.0", welcome_msg)
    
    def _create_status_bar(self):
        """ステータスバー作成"""
        self.status_bar = ctk.CTkFrame(self.root, height=30)
        self.status_bar.pack(side="bottom", fill="x", padx=10, pady=5)
        
        self.status_label = ctk.CTkLabel(self.status_bar, text="Ready")
        self.status_label.pack(side="left", padx=10)
        
        # メモリ使用量表示
        self.memory_label = ctk.CTkLabel(self.status_bar, text="Memory: 0 MB")
        self.memory_label.pack(side="right", padx=10)
        
        # メモリ監視タイマー
        self._update_memory_usage()
    
    def _update_memory_usage(self):
        """メモリ使用量更新"""
        try:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.memory_label.configure(text=f"Memory: {memory_mb:.1f} MB")
        except:
            pass
        
        # 30秒後に再実行
        self.root.after(30000, self._update_memory_usage)
    
    def _update_status(self, message):
        """ステータス更新"""
        self.status_label.configure(text=message)
    
    def _check_recovery(self):
        """セッション復旧チェック"""
        latest_session = self.session_manager.load_latest_session()
        if latest_session:
            result = messagebox.askyesno(
                "Session Recovery", 
                "Previous session found. Do you want to restore it?"
            )
            if result:
                self._restore_session(latest_session)
    
    def _restore_session(self, session_data):
        """セッション復旧"""
        try:
            if session_data.get('data') is not None:
                self.current_data = pd.DataFrame(session_data['data'])
                self._update_data_info()
            
            self.analysis_results = session_data.get('analysis_results', {})
            self.user_settings = session_data.get('settings', {})
            
            self._update_status("Session restored successfully")
            self.result_text.insert("end", "\n\n✅ Session restored from previous backup\n")
            
        except Exception as e:
            messagebox.showerror("Recovery Error", f"Failed to restore session: {e}")
    
    def load_data(self):
        """データ読み込み"""
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx;*.xls"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self._update_status("Loading data...")
            
            # プログレスバー表示
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Loading Data")
            progress_window.geometry("400x100")
            
            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
            progress_bar.pack(pady=20)
            progress_bar.start()
            
            # ファイル拡張子による読み込み
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.csv':
                self.current_data = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                self.current_data = pd.read_excel(file_path)
            elif file_ext == '.json':
                self.current_data = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            progress_window.destroy()
            
            self._update_data_info()
            self._update_status(f"Data loaded: {len(self.current_data)} rows")
            
            # セッション保存
            self.session_manager.save_session()
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self._update_status("Data loading failed")
    
    def _update_data_info(self):
        """データ情報更新"""
        if self.current_data is None:
            self.data_text.delete("1.0", "end")
            return
        
        info_text = f"""Shape: {self.current_data.shape[0]} rows × {self.current_data.shape[1]} columns

Columns:
{chr(10).join(f"• {col} ({self.current_data[col].dtype})" for col in self.current_data.columns[:10])}
{"..." if len(self.current_data.columns) > 10 else ""}

Missing Values: {self.current_data.isnull().sum().sum()}
Memory Usage: {self.current_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"""
        
        # 変数選択状態を追加
        if self.variable_selection_applied:
            info_text += f"""

🎯 Variable Selection:
Control: {len(self.control_variables)} vars
Target: {len(self.target_variables)} vars  
Residual: {len(self.residual_variables)} vars"""
        
        self.data_text.delete("1.0", "end")
        self.data_text.insert("1.0", info_text)
    
    def descriptive_stats(self):
        """記述統計"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        try:
            self._update_status("Computing descriptive statistics...")
            
            numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                messagebox.showwarning("Warning", "No numeric columns found")
                return
            
            # 基本統計量
            desc_stats = self.current_data[numeric_cols].describe()
            
            # 歪度・尖度
            skewness = self.current_data[numeric_cols].skew()
            kurtosis = self.current_data[numeric_cols].kurtosis()
            
            # 結果表示
            result = f"""
📊 DESCRIPTIVE STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Basic Statistics:
{desc_stats.round(4)}

Distribution Shape:
Skewness:
{skewness.round(4)}

Kurtosis:
{kurtosis.round(4)}

🔍 Data Quality Summary:
• Total observations: {len(self.current_data)}
• Numeric variables: {len(numeric_cols)}
• Complete cases: {self.current_data.dropna().shape[0]}
• Missing rate: {(self.current_data.isnull().sum().sum() / (len(self.current_data) * len(self.current_data.columns)) * 100):.1f}%
            """
            
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", result)
            
            # 結果保存
            self.analysis_results['descriptive_stats'] = {
                'basic_stats': desc_stats.to_dict(),
                'skewness': skewness.to_dict(),
                'kurtosis': kurtosis.to_dict()
            }
            
            self._update_status("Descriptive statistics completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            self._update_status("Analysis failed")
    
    def hypothesis_testing(self):
        """仮説検定"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        # シンプルな正規性検定例
        try:
            numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                messagebox.showwarning("Warning", "No numeric columns found")
                return
            
            self._update_status("Running hypothesis tests...")
            
            results = []
            for col in numeric_cols[:5]:  # 最初の5列のみ
                data = self.current_data[col].dropna()
                if len(data) < 3:
                    continue
                
                # Shapiro-Wilk正規性検定
                stat, p_value = stats.shapiro(data)
                results.append({
                    'Variable': col,
                    'Test': 'Shapiro-Wilk',
                    'Statistic': stat,
                    'P-value': p_value,
                    'Result': 'Normal' if p_value > 0.05 else 'Non-normal'
                })
            
            # 結果表示
            result_text = "\n🔬 HYPOTHESIS TESTING RESULTS\n"
            result_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            result_text += "Normality Tests (Shapiro-Wilk):\n\n"
            
            for res in results:
                result_text += f"Variable: {res['Variable']}\n"
                result_text += f"  Statistic: {res['Statistic']:.4f}\n"
                result_text += f"  P-value: {res['P-value']:.6f}\n"
                result_text += f"  Result: {res['Result']} (α = 0.05)\n\n"
            
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", result_text)
            
            self.analysis_results['hypothesis_tests'] = results
            self._update_status("Hypothesis testing completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Testing failed: {e}")
            self._update_status("Testing failed")
    
    def regression_analysis(self):
        """回帰分析"""
        messagebox.showinfo("Info", "Regression analysis feature coming soon!")
    
    def machine_learning(self):
        """高度な機械学習分析"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        # 変数選択情報を準備
        variable_selection = {
            'control_variables': self.control_variables,
            'target_variables': self.target_variables,
            'residual_variables': self.residual_variables,
            'applied': self.variable_selection_applied
        }
        
        # 機械学習ウィンドウを開く
        ml_window = MLAnalysisWindow(self.root, self.current_data, variable_selection)
        ml_window.show()
    
    def deep_learning(self):
        """深層学習分析"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        # 変数選択情報を準備
        variable_selection = {
            'control_variables': self.control_variables,
            'target_variables': self.target_variables,
            'residual_variables': self.residual_variables,
            'applied': self.variable_selection_applied
        }
        
        # 深層学習ウィンドウを開く
        dl_window = DeepLearningWindow(self.root, self.current_data, variable_selection)
        dl_window.show()
    
    def data_quality_check(self):
        """データ品質チェック"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        try:
            self._update_status("Checking data quality...")
            
            # データ品質レポート
            total_cells = len(self.current_data) * len(self.current_data.columns)
            missing_cells = self.current_data.isnull().sum().sum()
            duplicate_rows = self.current_data.duplicated().sum()
            
            quality_report = f"""
🔍 DATA QUALITY REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset Overview:
• Total rows: {len(self.current_data):,}
• Total columns: {len(self.current_data.columns)}
• Total cells: {total_cells:,}

Completeness:
• Missing cells: {missing_cells:,} ({missing_cells/total_cells*100:.2f}%)
• Complete rows: {len(self.current_data.dropna()):,}
• Duplicate rows: {duplicate_rows:,}

Column Details:
"""
            
            for col in self.current_data.columns:
                missing_pct = self.current_data[col].isnull().sum() / len(self.current_data) * 100
                unique_count = self.current_data[col].nunique()
                quality_report += f"• {col}: {missing_pct:.1f}% missing, {unique_count} unique values\n"
            
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", quality_report)
            
            self._update_status("Data quality check completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Quality check failed: {e}")
    
    def show_gpu_status(self):
        """GPU状態表示"""
        gpu_info = f"""
⚡ GPU INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CUDA Available: {'✅ Yes' if CUDA_AVAILABLE else '❌ No'}
GPU Device: {GPU_NAME}
"""
        
        if CUDA_AVAILABLE:
            gpu_info += f"""
GPU Memory:
• Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB
• Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB
• Cached: {torch.cuda.memory_reserved() / 1024**3:.1f} GB

Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}
"""
        
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", gpu_info)
    
    def show_memory_usage(self):
        """メモリ使用量表示"""
        memory_info = f"""
💾 MEMORY INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Process Memory:
• RSS: {psutil.Process().memory_info().rss / 1024**2:.1f} MB
• VMS: {psutil.Process().memory_info().vms / 1024**2:.1f} MB

System Memory:
• Total: {psutil.virtual_memory().total / 1024**3:.1f} GB
• Available: {psutil.virtual_memory().available / 1024**3:.1f} GB
• Used: {psutil.virtual_memory().percent:.1f}%
"""
        
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", memory_info)
    
    def variable_selection(self):
        """変数選択ウィンドウを開く"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        def on_variable_selection(result):
            """変数選択結果を処理"""
            self.control_variables = result['control_variables']
            self.target_variables = result['target_variables']
            self.residual_variables = result['residual_variables']
            self.variable_selection_applied = True
            
            # データ情報更新
            self._update_data_info()
            
            # 結果表示
            summary = f"""
🎯 Variable Selection Applied
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Control Variables ({len(self.control_variables)}):
{', '.join(self.control_variables) if self.control_variables else 'None'}

🎯 Target Variables ({len(self.target_variables)}):
{', '.join(self.target_variables) if self.target_variables else 'None'}

📦 Residual Variables ({len(self.residual_variables)}):
{', '.join(self.residual_variables) if self.residual_variables else 'None'}

✅ Variable selection has been applied successfully!
Use these selections in subsequent analyses (ML, regression, etc.)
            """
            
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", summary)
            self._update_status("Variable selection applied")
        
        # 変数選択ウィンドウを開く
        var_window = VariableSelectionWindow(self.root, self.current_data, on_variable_selection)
        var_window.show()
    
    def generate_professional_report(self):
        """プロフェッショナルレポート生成"""
        if not PROFESSIONAL_FEATURES_AVAILABLE:
            messagebox.showerror("Error", "プロフェッショナル機能が利用できません")
            return
        
        if self.current_data is None:
            messagebox.showerror("Error", "データが読み込まれていません")
            return
        
        try:
            # プログレスバーダイアログ
            progress_window = ctk.CTkToplevel(self.root)
            progress_window.title("レポート生成中...")
            progress_window.geometry("400x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_label = ctk.CTkLabel(progress_window, text="プロフェッショナルレポートを生成中...")
            progress_label.pack(pady=20)
            
            progress_bar = ctk.CTkProgressBar(progress_window)
            progress_bar.pack(pady=20, padx=40, fill="x")
            progress_bar.set(0)
            
            def generate_report():
                try:
                    progress_bar.set(0.2)
                    progress_window.update()
                    
                    # レポート生成
                    report_file = report_generator.generate_comprehensive_report(
                        data=self.current_data,
                        analysis_results=self.analysis_results,
                        title="HAD Professional Statistical Analysis Report",
                        subtitle=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    
                    progress_bar.set(1.0)
                    progress_window.update()
                    
                    progress_window.destroy()
                    
                    # 成功メッセージ
                    result = messagebox.askyesno(
                        "レポート生成完了", 
                        f"プロフェッショナルレポートが生成されました:\n{report_file}\n\nレポートを開きますか？"
                    )
                    
                    if result:
                        import webbrowser
                        webbrowser.open(f"file://{Path(report_file).absolute()}")
                    
                    # ログ記録
                    if PROFESSIONAL_FEATURES_AVAILABLE:
                        professional_logger.info("プロフェッショナルレポート生成完了", 
                                                file_path=report_file,
                                                data_shape=self.current_data.shape)
                
                except Exception as e:
                    progress_window.destroy()
                    messagebox.showerror("Error", f"レポート生成エラー:\n{str(e)}")
                    if PROFESSIONAL_FEATURES_AVAILABLE:
                        professional_logger.error("レポート生成失敗", exception=e)
            
            # レポート生成を別スレッドで実行
            import threading
            thread = threading.Thread(target=generate_report)
            thread.daemon = True
            thread.start()
        
        except Exception as e:
            messagebox.showerror("Error", f"レポート生成エラー:\n{str(e)}")
    
    def show_performance_report(self):
        """パフォーマンスレポート表示"""
        if not PROFESSIONAL_FEATURES_AVAILABLE:
            messagebox.showerror("Error", "プロフェッショナル機能が利用できません")
            return
        
        try:
            # パフォーマンスレポート取得
            perf_report = performance_monitor.get_performance_report()
            
            # レポートウィンドウ
            report_window = ctk.CTkToplevel(self.root)
            report_window.title("📊 Performance Report")
            report_window.geometry("800x600")
            report_window.transient(self.root)
            
            # テキストボックス
            text_widget = ctk.CTkTextbox(report_window, font=("Consolas", 11))
            text_widget.pack(fill="both", expand=True, padx=10, pady=10)
            
            # レポート内容生成
            report_content = "🚀 HAD Professional Performance Report\n"
            report_content += "=" * 60 + "\n\n"
            report_content += f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_content += f"💾 Session ID: {self.session_manager.session_id}\n\n"
            
            if perf_report:
                report_content += "📊 Function Performance Metrics:\n"
                report_content += "-" * 40 + "\n"
                
                for func_name, metrics in perf_report.items():
                    report_content += f"\n🔧 {func_name}:\n"
                    report_content += f"   • Call Count: {metrics['call_count']}\n"
                    report_content += f"   • Success Rate: {metrics['success_rate']:.2%}\n"
                    report_content += f"   • Avg Duration: {metrics['avg_duration']:.4f}s\n"
                    report_content += f"   • Max Duration: {metrics['max_duration']:.4f}s\n"
                    report_content += f"   • Min Duration: {metrics['min_duration']:.4f}s\n"
                    report_content += f"   • Avg Memory: {metrics['avg_memory_diff']:.2f}MB\n"
            else:
                report_content += "ℹ️ パフォーマンスデータがまだ収集されていません。\n"
                report_content += "いくつかの解析を実行してから再度確認してください。\n"
            
            # システム情報
            report_content += "\n" + "=" * 60 + "\n"
            report_content += "🖥️ System Information:\n"
            report_content += f"   • Python: {sys.version.split()[0]}\n"
            report_content += f"   • CUDA Available: {CUDA_AVAILABLE}\n"
            report_content += f"   • GPU: {GPU_NAME}\n"
            
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                report_content += f"   • Total Memory: {memory_info.total / 1024**3:.1f} GB\n"
                report_content += f"   • Available Memory: {memory_info.available / 1024**3:.1f} GB\n"
                report_content += f"   • Memory Usage: {memory_info.percent}%\n"
                report_content += f"   • CPU Count: {psutil.cpu_count()}\n"
            except:
                pass
            
            text_widget.insert("1.0", report_content)
            
            # エクスポートボタン
            export_btn = ctk.CTkButton(
                report_window,
                text="📄 Export Report",
                command=lambda: self._export_performance_report(report_content)
            )
            export_btn.pack(pady=10)
        
        except Exception as e:
            messagebox.showerror("Error", f"パフォーマンスレポートエラー:\n{str(e)}")
    
    def _export_performance_report(self, content):
        """パフォーマンスレポートエクスポート"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"パフォーマンスレポートをエクスポートしました:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"エクスポートエラー:\n{str(e)}")

    def ai_analysis(self):
        """AI統計分析"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        # AI分析ウィンドウを開く
        ai_window = AIAnalysisWindow(self.root, self.current_data)
        ai_window.show()
    
    def save_session(self):
        """手動セッション保存"""
        try:
            saved_file = self.session_manager.save_session()
            messagebox.showinfo("Success", f"Session saved: {saved_file.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save session: {e}")
    
    def _open_ml_dialog(self):
        """機械学習ダイアログを開く"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "データが読み込まれていません。")
            return
        
        # 機械学習ダイアログの実装（簡易版）
        ml_window = tk.Toplevel(self.root)
        ml_window.title("機械学習解析")
        ml_window.geometry("600x400")
        
        # 機械学習手法選択
        ttk.Label(ml_window, text="機械学習手法:").pack(pady=5)
        ml_method = ttk.Combobox(ml_window, values=[
            "Random Forest", "XGBoost", "LightGBM", 
            "SVM", "Logistic Regression", "Linear Regression"
        ])
        ml_method.pack(pady=5)
        ml_method.set("Random Forest")
        
        # 実行ボタン
        def run_ml():
            method = ml_method.get()
            messagebox.showinfo("Info", f"{method}を実行します（実装予定）")
        
        ttk.Button(ml_window, text="実行", command=run_ml).pack(pady=10)
    
    def _open_dl_dialog(self):
        """深層学習ダイアログを開く"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "データが読み込まれていません。")
            return
        
        # 深層学習ダイアログの実装（簡易版）
        dl_window = tk.Toplevel(self.root)
        dl_window.title("深層学習解析")
        dl_window.geometry("600x400")
        
        # GPU状態表示
        gpu_status = "✅ CUDA利用可能" if CUDA_AVAILABLE else "❌ CPU使用"
        ttk.Label(dl_window, text=f"GPU状態: {gpu_status}").pack(pady=5)
        
        # 深層学習手法選択
        ttk.Label(dl_window, text="深層学習手法:").pack(pady=5)
        dl_method = ttk.Combobox(dl_window, values=[
            "Neural Network", "CNN", "RNN", "LSTM", "Transformer"
        ])
        dl_method.pack(pady=5)
        dl_method.set("Neural Network")
        
        # 実行ボタン
        def run_dl():
            method = dl_method.get()
            messagebox.showinfo("Info", f"{method}を実行します（実装予定）")
        
        ttk.Button(dl_window, text="実行", command=run_dl).pack(pady=10)
        dl_window = DeepLearningWindow(self.root, self.current_data)
        dl_window.show()
    
    # ==================== 高度統計解析メソッド ====================
    
    def multivariate_analysis(self):
        """多変量解析"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        if not hasattr(self, 'advanced_features') or not self.advanced_features:
            messagebox.showwarning("Warning", "Advanced features not available!")
            return
        
        try:
            self._update_status("🔬 実行中: 多変量解析...")
            
            # プログレスバーダイアログ作成
            progress_window = tk.Toplevel(self.root)
            progress_window.title("多変量解析実行中")
            progress_window.geometry("400x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_label = ctk.CTkLabel(progress_window, text="多変量解析を実行中です...")
            progress_label.pack(pady=20)
            
            progress_bar = ctk.CTkProgressBar(progress_window)
            progress_bar.pack(pady=10, padx=20, fill="x")
            progress_bar.set(0)
            
            def run_analysis():
                try:
                    # データ最適化
                    progress_bar.set(0.2)
                    progress_window.update()
                    optimized_data = self.parallel_processor.optimize_dataframe_operations(self.current_data)
                    
                    # 多変量解析実行
                    progress_bar.set(0.5)
                    progress_window.update()
                    results = self.advanced_analyzer.multivariate_analysis(optimized_data)
                    
                    # 結果表示
                    progress_bar.set(1.0)
                    progress_window.update()
                    
                    # 結果ウィンドウ作成
                    result_window = tk.Toplevel(self.root)
                    result_window.title("🔬 多変量解析結果")
                    result_window.geometry("1000x700")
                    
                    # ノートブック（タブ）作成
                    notebook = ctk.CTkTabview(result_window)
                    notebook.pack(fill="both", expand=True, padx=10, pady=10)
                    
                    # PCA結果タブ
                    pca_tab = notebook.add("主成分分析")
                    pca_text = ctk.CTkTextbox(pca_tab, font=("Consolas", 11))
                    pca_text.pack(fill="both", expand=True, padx=5, pady=5)
                    
                    pca_results = results.get('pca', {})
                    pca_content = f"""
📊 主成分分析（PCA）結果
═══════════════════════════════════════════════════════════════

🎯 Kaiser基準による成分数: {pca_results.get('n_components_kaiser', 'N/A')}
📈 累積寄与率80%の成分数: {pca_results.get('n_components_80', 'N/A')}

📊 寄与率（上位10成分）:
"""
                    explained_var = pca_results.get('explained_variance_ratio', [])
                    for i, var in enumerate(explained_var[:10]):
                        pca_content += f"PC{i+1}: {var*100:.2f}%\n"
                    
                    cumulative_var = pca_results.get('cumulative_variance', [])
                    if cumulative_var:
                        pca_content += f"\n📈 累積寄与率（PC10まで）: {cumulative_var[min(9, len(cumulative_var)-1)]*100:.2f}%\n"
                    
                    pca_text.insert("1.0", pca_content)
                    
                    # クラスタリング結果タブ
                    cluster_tab = notebook.add("クラスタリング")
                    cluster_text = ctk.CTkTextbox(cluster_tab, font=("Consolas", 11))
                    cluster_text.pack(fill="both", expand=True, padx=5, pady=5)
                    
                    clustering_results = results.get('clustering', {})
                    kmeans_results = clustering_results.get('kmeans', {})
                    
                    cluster_content = f"""
🎯 K-means クラスタリング結果
═══════════════════════════════════════════════════════════════

🎯 最適クラスタ数: {kmeans_results.get('optimal_k', 'N/A')}

📊 シルエットスコア（クラスタ数別）:
"""
                    silhouette_scores = kmeans_results.get('silhouette_scores', [])
                    for i, score in enumerate(silhouette_scores):
                        cluster_content += f"K={i+2}: {score:.4f}\n"
                    
                    dbscan_results = clustering_results.get('dbscan', {})
                    if 'n_clusters' in dbscan_results:
                        cluster_content += f"\n🔍 DBSCAN検出クラスタ数: {dbscan_results['n_clusters']}\n"
                    
                    cluster_text.insert("1.0", cluster_content)
                    
                    # 相関分析タブ
                    corr_tab = notebook.add("相関分析")
                    corr_text = ctk.CTkTextbox(corr_tab, font=("Consolas", 11))
                    corr_text.pack(fill="both", expand=True, padx=5, pady=5)
                    
                    correlation_results = results.get('correlation', {})
                    high_corr = correlation_results.get('high_correlations', [])
                    
                    corr_content = """
🔗 相関分析結果
═══════════════════════════════════════════════════════════════

⚠️ 高相関ペア（|r| > 0.7）:

"""
                    if high_corr:
                        for pair in high_corr:
                            corr_content += f"• {pair['var1']} ⟷ {pair['var2']}: r = {pair['correlation']:.4f}\n"
                    else:
                        corr_content += "高相関ペアは検出されませんでした。\n"
                    
                    corr_text.insert("1.0", corr_content)
                    
                    progress_window.destroy()
                    self._update_status("✅ 多変量解析完了")
                    
                except Exception as e:
                    progress_window.destroy()
                    messagebox.showerror("Error", f"多変量解析エラー: {e}")
                    self._update_status("❌ 多変量解析失敗")
            
            # 別スレッドで実行
            threading.Thread(target=run_analysis, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"多変量解析エラー: {e}")
            self._update_status("❌ 多変量解析失敗")
    
    def time_series_analysis(self):
        """時系列解析"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        if not hasattr(self, 'advanced_features') or not self.advanced_features:
            messagebox.showwarning("Warning", "Advanced features not available!")
            return
        
        # 時系列解析用のダイアログ
        dialog = tk.Toplevel(self.root)
        dialog.title("📈 時系列解析設定")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 日付列選択
        date_frame = ctk.CTkFrame(dialog)
        date_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(date_frame, text="日付列:").pack(side="left", padx=5)
        date_var = ctk.CTkComboBox(date_frame, values=list(self.current_data.columns))
        date_var.pack(side="right", padx=5)
        
        # 値列選択
        value_frame = ctk.CTkFrame(dialog)
        value_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(value_frame, text="値列:").pack(side="left", padx=5)
        value_var = ctk.CTkComboBox(value_frame, values=list(self.current_data.select_dtypes(include=[np.number]).columns))
        value_var.pack(side="right", padx=5)
        
        def run_ts_analysis():
            date_col = date_var.get()
            value_col = value_var.get()
            
            if not date_col or not value_col:
                messagebox.showwarning("Warning", "Please select both date and value columns!")
                return
            
            dialog.destroy()
            
            try:
                self._update_status("📈 実行中: 時系列解析...")
                results = self.advanced_analyzer.time_series_analysis(self.current_data, date_col, value_col)
                
                # 結果表示
                result_text = f"""
📈 時系列解析結果
═══════════════════════════════════════════════════════════════

📊 基本統計:
平均: {results['basic_stats']['mean']:.4f}
標準偏差: {results['basic_stats']['std']:.4f}
観測数: {results['basic_stats']['observations']}

🔍 定常性検定:
ADF検定 p値: {results['stationarity_tests']['adf']['p_value']:.6f}
→ 定常性: {'Yes' if results['stationarity_tests']['adf']['is_stationary'] else 'No'}

KPSS検定 p値: {results['stationarity_tests']['kpss']['p_value']:.6f}
→ 定常性: {'Yes' if results['stationarity_tests']['kpss']['is_stationary'] else 'No'}
"""
                
                arima_results = results.get('arima', {})
                if 'best_params' in arima_results:
                    result_text += f"""
📊 ARIMA モデル:
最適パラメータ: {arima_results['best_params']}
AIC: {arima_results['aic']:.4f}
BIC: {arima_results['bic']:.4f}
"""
                
                self.result_text.delete("1.0", tk.END)
                self.result_text.insert("1.0", result_text)
                self._update_status("✅ 時系列解析完了")
                
            except Exception as e:
                messagebox.showerror("Error", f"時系列解析エラー: {e}")
                self._update_status("❌ 時系列解析失敗")
        
        # 実行ボタン
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(button_frame, text="実行", command=run_ts_analysis).pack(side="right", padx=5)
        ctk.CTkButton(button_frame, text="キャンセル", command=dialog.destroy).pack(side="right", padx=5)
    
    def survival_analysis(self):
        """生存解析"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        if not hasattr(self, 'advanced_features') or not self.advanced_features:
            messagebox.showwarning("Warning", "Advanced features not available!")
            return
        
        messagebox.showinfo("Info", "生存解析機能は開発中です。次回アップデートで追加予定です。")
    
    def bayesian_analysis(self):
        """ベイズ統計解析"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        if not hasattr(self, 'advanced_features') or not self.advanced_features:
            messagebox.showwarning("Warning", "Advanced features not available!")
            return
        
        messagebox.showinfo("Info", "ベイズ統計解析機能は開発中です。PyMCライブラリが必要です。")
    
    def comprehensive_eda(self):
        """包括的探索的データ解析"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        if not hasattr(self, 'advanced_features') or not self.advanced_features:
            messagebox.showwarning("Warning", "Advanced features not available!")
            return
        
        try:
            self._update_status("📊 実行中: 包括的EDA...")
            
            # プログレスダイアログ
            progress_window = tk.Toplevel(self.root)
            progress_window.title("包括的EDA実行中")
            progress_window.geometry("400x120")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_label = ctk.CTkLabel(progress_window, text="データ解析中...")
            progress_label.pack(pady=10)
            
            progress_bar = ctk.CTkProgressBar(progress_window)
            progress_bar.pack(pady=10, padx=20, fill="x")
            progress_bar.set(0)
            
            def run_eda():
                try:
                    progress_bar.set(0.3)
                    progress_window.update()
                    
                    results = self.advanced_analyzer.comprehensive_eda(self.current_data)
                    
                    progress_bar.set(1.0)
                    progress_window.update()
                    
                    # 結果表示
                    result_text = f"""
📊 包括的探索的データ解析（EDA）結果
═══════════════════════════════════════════════════════════════

📋 データ概要:
• 総行数: {results['data_overview']['total_rows']:,}
• 総列数: {results['data_overview']['total_columns']}
• メモリ使用量: {results['data_overview']['memory_usage_mb']:.1f} MB
• 数値列: {results['data_overview']['numeric_columns']}
• カテゴリ列: {results['data_overview']['categorical_columns']}

❌ 欠損値分析:
• 欠損値総数: {results['missing_values']['total_missing']:,}
• 欠損値のある列数: {results['missing_values']['columns_with_missing']}
• 完全な行数: {results['missing_values']['complete_rows']:,}

🎯 外れ値検出（上位5列）:
"""
                    outliers = results.get('outliers', {})
                    for i, (col, info) in enumerate(list(outliers.items())[:5]):
                        result_text += f"• {col}: IQR法 {info['iqr_outliers']}個 ({info['outlier_percentage_iqr']:.1f}%)\n"
                    
                    # 分布分析
                    distributions = results.get('distributions', {})
                    normal_cols = [col for col, info in distributions.items() if info.get('is_normal', False)]
                    
                    result_text += f"""

📈 分布分析:
• 正規分布に従う列: {len(normal_cols)}個
• 非正規分布の列: {len(distributions) - len(normal_cols)}個

🔗 関係性分析:
"""
                    relationships = results.get('relationships', {})
                    if 'strong_correlations' in relationships:
                        strong_corr = relationships['strong_correlations']
                        result_text += f"• 強い相関を持つペア: {len(strong_corr)}個\n"
                        
                        if strong_corr:
                            result_text += "\n強相関ペア:\n"
                            for pair in strong_corr[:5]:  # 上位5個
                                result_text += f"  - {pair['var1']} ⟷ {pair['var2']}: r = {pair['correlation']:.3f}\n"
                    
                    result_text += f"""

⚡ パフォーマンス情報:
• 解析完了時刻: {results['timestamp']}
• データ形状: {results['data_shape']}
"""
                    
                    progress_window.destroy()
                    
                    self.result_text.delete("1.0", tk.END)
                    self.result_text.insert("1.0", result_text)
                    self._update_status("✅ 包括的EDA完了")
                    
                except Exception as e:
                    progress_window.destroy()
                    messagebox.showerror("Error", f"EDAエラー: {e}")
                    self._update_status("❌ EDA失敗")
            
            threading.Thread(target=run_eda, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"EDAエラー: {e}")
            self._update_status("❌ EDA失敗")
    
    def run(self):
        """アプリケーション実行"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n🛑 Application interrupted by user")
        finally:
            # 終了時保存
            try:
                self.session_manager.save_session()
                print("✅ Final session saved")
            except:
                pass

def main():
    """メイン関数"""
    print("🔬 Professional Statistical Analysis Software")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"CUDA: {'Available' if CUDA_AVAILABLE else 'Not Available'}")
    if CUDA_AVAILABLE:
        print(f"GPU: {GPU_NAME}")
    print("=" * 50)
    
    try:
        app = StatisticalAnalysisGUI()
        app.run()
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 