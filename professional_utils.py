#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Professional Utilities Module
プロフェッショナルユーティリティモジュール

Author: Ryo Minegishi
License: MIT
"""

import os
import sys
import json
import logging
import traceback
import time
import threading
import functools
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
import sqlite3
import pickle
import psutil
from tqdm import tqdm

class ProfessionalLogger:
    """プロフェッショナルロギングシステム"""
    
    def __init__(self, name: str = "HAD_Statistics", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # ログローテーション設定
        self.max_log_size = 10 * 1024 * 1024  # 10MB
        self.backup_count = 5
        
        self._setup_loggers()
        
    def _setup_loggers(self):
        """ロガー設定"""
        # メインロガー
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # フォーマッター
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s:%(lineno)d | %(funcName)s() | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # ファイルハンドラー（情報レベル）
        info_handler = self._create_rotating_handler('info.log', logging.INFO, formatter)
        self.logger.addHandler(info_handler)
        
        # ファイルハンドラー（エラーレベル）
        error_handler = self._create_rotating_handler('error.log', logging.ERROR, formatter)
        self.logger.addHandler(error_handler)
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # パフォーマンスロガー
        self.perf_logger = logging.getLogger(f"{self.name}.performance")
        perf_handler = self._create_rotating_handler('performance.log', logging.INFO, formatter)
        self.perf_logger.addHandler(perf_handler)
        
    def _create_rotating_handler(self, filename: str, level: int, formatter):
        """ローテーションハンドラー作成"""
        from logging.handlers import RotatingFileHandler
        
        filepath = self.log_dir / filename
        handler = RotatingFileHandler(
            filepath, maxBytes=self.max_log_size, backupCount=self.backup_count
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        return handler
    
    def debug(self, message: str, **kwargs):
        """デバッグログ"""
        self.logger.debug(self._format_message(message, kwargs))
    
    def info(self, message: str, **kwargs):
        """情報ログ"""
        self.logger.info(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """警告ログ"""
        self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, exception: Exception = None, **kwargs):
        """エラーログ"""
        msg = self._format_message(message, kwargs)
        if exception:
            msg += f"\n例外詳細: {str(exception)}\n{traceback.format_exc()}"
        self.logger.error(msg)
    
    def critical(self, message: str, exception: Exception = None, **kwargs):
        """クリティカルログ"""
        msg = self._format_message(message, kwargs)
        if exception:
            msg += f"\n例外詳細: {str(exception)}\n{traceback.format_exc()}"
        self.logger.critical(msg)
    
    def performance(self, message: str, duration: float = None, **kwargs):
        """パフォーマンスログ"""
        msg = self._format_message(message, kwargs)
        if duration:
            msg += f" | 実行時間: {duration:.4f}秒"
        self.perf_logger.info(msg)
    
    def _format_message(self, message: str, kwargs: Dict) -> str:
        """メッセージフォーマット"""
        if kwargs:
            context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            return f"{message} | {context}"
        return message

class PerformanceMonitor:
    """パフォーマンス監視システム"""
    
    def __init__(self, logger: ProfessionalLogger):
        self.logger = logger
        self.metrics = {}
        self.monitoring_active = True
        
    def monitor_function(self, func_name: str = None):
        """関数パフォーマンス監視デコレータ"""
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.monitoring_active:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                try:
                    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                except:
                    start_memory = 0
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error_msg = None
                except Exception as e:
                    success = False
                    error_msg = str(e)
                    raise
                finally:
                    end_time = time.time()
                    try:
                        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    except:
                        end_memory = start_memory
                    
                    duration = end_time - start_time
                    memory_diff = end_memory - start_memory
                    
                    # メトリクス記録
                    if name not in self.metrics:
                        self.metrics[name] = []
                    
                    self.metrics[name].append({
                        'timestamp': datetime.now(),
                        'duration': duration,
                        'memory_diff': memory_diff,
                        'success': success,
                        'error': error_msg
                    })
                    
                    # ログ出力
                    self.logger.performance(
                        f"関数実行完了: {name}",
                        duration=duration,
                        memory_diff=f"{memory_diff:.2f}MB",
                        success=success
                    )
                
                return result
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict:
        """パフォーマンスレポート生成"""
        report = {}
        for func_name, metrics in self.metrics.items():
            durations = [m['duration'] for m in metrics if m['success']]
            memory_diffs = [m['memory_diff'] for m in metrics if m['success']]
            
            if durations:
                report[func_name] = {
                    'call_count': len(metrics),
                    'success_rate': sum(1 for m in metrics if m['success']) / len(metrics),
                    'avg_duration': sum(durations) / len(durations),
                    'max_duration': max(durations),
                    'min_duration': min(durations),
                    'avg_memory_diff': sum(memory_diffs) / len(memory_diffs) if memory_diffs else 0
                }
        
        return report

class ExceptionHandler:
    """プロフェッショナル例外処理システム"""
    
    def __init__(self, logger: ProfessionalLogger):
        self.logger = logger
        self.exception_handlers = {}
    
    def register_handler(self, exception_type: type, handler: Callable):
        """例外ハンドラー登録"""
        self.exception_handlers[exception_type] = handler
    
    def handle_exception(self, exception: Exception, context: str = ""):
        """例外処理"""
        exception_type = type(exception)
        
        # カスタムハンドラーチェック
        if exception_type in self.exception_handlers:
            try:
                return self.exception_handlers[exception_type](exception, context)
            except Exception as handler_error:
                self.logger.error(
                    f"例外ハンドラーエラー: {context}",
                    exception=handler_error
                )
        
        # デフォルト処理
        self.logger.error(
            f"未処理例外: {context}",
            exception=exception,
            exception_type=exception_type.__name__
        )
        
        return {
            'handled': False,
            'error_message': str(exception),
            'exception_type': exception_type.__name__,
            'traceback': traceback.format_exc()
        }
    
    def safe_execute(self, func: Callable, *args, **kwargs):
        """安全実行ラッパー"""
        try:
            return {
                'success': True,
                'result': func(*args, **kwargs),
                'error': None
            }
        except Exception as e:
            error_info = self.handle_exception(e, f"{func.__name__}")
            return {
                'success': False,
                'result': None,
                'error': error_info
            }

class SecurityManager:
    """セキュリティ管理システム"""
    
    def __init__(self, key_file: str = "security/master.key"):
        self.key_file = Path(key_file)
        self.key_file.parent.mkdir(exist_ok=True)
        
        # セッション管理
        self.session_timeout = timedelta(hours=8)
        self.active_sessions = {}
        
    def create_session(self, user_id: str = "default") -> str:
        """セッション作成"""
        session_id = secrets.token_urlsafe(32)
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """セッション検証"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        if datetime.now() - session['last_activity'] > self.session_timeout:
            del self.active_sessions[session_id]
            return False
        
        session['last_activity'] = datetime.now()
        return True

class DatabaseManager:
    """プロフェッショナルデータベース管理"""
    
    def __init__(self, db_path: str = "data/analysis_results.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 解析結果テーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    parameters TEXT,
                    results BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT DEFAULT 'default'
                )
            ''')
            
            # パフォーマンスメトリクステーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    function_name TEXT NOT NULL,
                    duration REAL NOT NULL,
                    memory_usage REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def save_analysis_result(self, session_id: str, analysis_type: str, 
                           parameters: Dict, results: Any, user_id: str = "default"):
        """解析結果保存"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # データ暗号化
            encrypted_results = pickle.dumps(results)
            
            cursor.execute('''
                INSERT INTO analysis_results 
                (session_id, analysis_type, parameters, results, user_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, analysis_type, json.dumps(parameters), 
                  encrypted_results, user_id))
            
            conn.commit()
    
    def load_analysis_results(self, session_id: str = None, 
                            analysis_type: str = None) -> List[Dict]:
        """解析結果読み込み"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM analysis_results WHERE 1=1"
            params = []
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            if analysis_type:
                query += " AND analysis_type = ?"
                params.append(analysis_type)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                try:
                    decrypted_results = pickle.loads(row[4])
                    
                    results.append({
                        'id': row[0],
                        'session_id': row[1],
                        'analysis_type': row[2],
                        'parameters': json.loads(row[3]) if row[3] else {},
                        'results': decrypted_results,
                        'created_at': row[5],
                        'user_id': row[6]
                    })
                except Exception as e:
                    print(f"データ復号化エラー: {e}")
                    continue
            
            return results

# グローバルインスタンス
professional_logger = ProfessionalLogger()
performance_monitor = PerformanceMonitor(professional_logger)
exception_handler = ExceptionHandler(professional_logger)
security_manager = SecurityManager()
database_manager = DatabaseManager()
