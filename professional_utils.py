#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Professional Utilities
プロフェッショナルユーティリティモジュール

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import logging
import time
import traceback
from functools import wraps
from datetime import datetime
from pathlib import Path
import json

class ProfessionalLogger:
    """プロフェッショナルロガー"""
    
    def __init__(self):
        self.setup_logger()
    
    def setup_logger(self):
        """ロガー設定"""
        self.logger = logging.getLogger('HAD_Professional')
        self.logger.setLevel(logging.INFO)
        
        # ファイルハンドラー
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"had_professional_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # フォーマッター
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
    
    def info(self, message, **kwargs):
        """情報ログ"""
        extra_info = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.logger.info(full_message)
    
    def error(self, message, exception=None, **kwargs):
        """エラーログ"""
        extra_info = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        if exception:
            full_message += f" | Exception: {str(exception)}"
        self.logger.error(full_message)
    
    def warning(self, message, **kwargs):
        """警告ログ"""
        extra_info = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.logger.warning(full_message)

class PerformanceMonitor:
    """パフォーマンス監視"""
    
    def __init__(self):
        self.performance_data = {}
    
    def monitor_function(self, func):
        """関数実行時間監視デコレータ"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                duration = end_time - start_time
                
                func_name = func.__name__
                if func_name not in self.performance_data:
                    self.performance_data[func_name] = {
                        'call_count': 0,
                        'total_time': 0,
                        'successes': 0,
                        'failures': 0,
                        'min_duration': float('inf'),
                        'max_duration': 0,
                        'memory_usage': []
                    }
                
                self.performance_data[func_name]['call_count'] += 1
                self.performance_data[func_name]['total_time'] += duration
                
                if success:
                    self.performance_data[func_name]['successes'] += 1
                else:
                    self.performance_data[func_name]['failures'] += 1
                
                self.performance_data[func_name]['min_duration'] = min(
                    self.performance_data[func_name]['min_duration'], duration
                )
                self.performance_data[func_name]['max_duration'] = max(
                    self.performance_data[func_name]['max_duration'], duration
                )
            
            return result
        return wrapper
    
    def get_performance_report(self):
        """パフォーマンスレポート取得"""
        report = {}
        for func_name, data in self.performance_data.items():
            if data['call_count'] > 0:
                report[func_name] = {
                    'call_count': data['call_count'],
                    'success_rate': data['successes'] / data['call_count'],
                    'avg_duration': data['total_time'] / data['call_count'],
                    'min_duration': data['min_duration'],
                    'max_duration': data['max_duration'],
                    'avg_memory_diff': sum(data['memory_usage']) / len(data['memory_usage']) if data['memory_usage'] else 0
                }
        return report

class ExceptionHandler:
    """例外ハンドラー"""
    
    def __init__(self):
        self.error_log = []
    
    def handle_exception(self, func):
        """例外処理デコレータ"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = {
                    'function': func.__name__,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                }
                self.error_log.append(error_info)
                professional_logger.error(f"Exception in {func.__name__}", exception=e)
                raise
        return wrapper
    
    def get_error_log(self):
        """エラーログ取得"""
        return self.error_log

class SecurityManager:
    """セキュリティ管理"""
    
    def __init__(self):
        self.access_log = []
    
    def secure_function(self, func):
        """セキュア関数デコレータ"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            access_info = {
                'function': func.__name__,
                'timestamp': datetime.now().isoformat(),
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            self.access_log.append(access_info)
            return func(*args, **kwargs)
        return wrapper
    
    def validate_data_access(self, data_source):
        """データアクセス検証"""
        # セキュリティチェックのスタブ実装
        return True

class DatabaseManager:
    """データベース管理"""
    
    def __init__(self):
        self.connection = None
        self.query_log = []
    
    def connect(self, database_url):
        """データベース接続"""
        # データベース接続のスタブ実装
        self.connection = f"connected_to_{database_url}"
        return True
    
    def execute_query(self, query):
        """クエリ実行"""
        query_info = {
            'query': query,
            'timestamp': datetime.now().isoformat()
        }
        self.query_log.append(query_info)
        # クエリ実行のスタブ実装
        return {"result": "query executed successfully"}
    
    def close(self):
        """接続クローズ"""
        self.connection = None

# グローバルインスタンス
professional_logger = ProfessionalLogger()
performance_monitor = PerformanceMonitor()
exception_handler = ExceptionHandler()
security_manager = SecurityManager()
database_manager = DatabaseManager()
