#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel Processing and Optimization Module
並列処理・最適化モジュール

Author: Ryo Minegishi
License: MIT
"""

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial, wraps
import time
import psutil
import gc
from typing import Callable, List, Dict, Any, Optional, Tuple
import warnings
from tqdm import tqdm
import threading
import queue
import asyncio
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
import joblib
from joblib import Parallel, delayed
from datetime import datetime
import os
import signal
import sys

# プロフェッショナル機能インポート
try:
    from professional_utils import professional_logger, performance_monitor
    PROFESSIONAL_LOGGING = True
except ImportError:
    PROFESSIONAL_LOGGING = False

warnings.filterwarnings('ignore')

class ParallelProcessor:
    """並列処理マネージャー"""
    
    def __init__(self, n_jobs: int = -1, backend: str = 'threading'):
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.backend = backend  # 'threading', 'multiprocessing'
        self.max_memory_usage = 0.8  # 最大メモリ使用率
        
        # システム情報取得
        self.system_info = self._get_system_info()
        
        if PROFESSIONAL_LOGGING:
            professional_logger.info("並列処理システム初期化", 
                                   n_jobs=self.n_jobs, 
                                   backend=backend,
                                   **self.system_info)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_freq_max = cpu_freq.max if cpu_freq else None
        except:
            cpu_freq_max = None
            
        return {
            'cpu_count': mp.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_freq_max': cpu_freq_max
        }
    
    def parallel_analysis(self, func: Callable, data_chunks: List[Any], 
                         *args, **kwargs) -> List[Any]:
        """並列解析実行"""
        
        # メモリ使用量チェック
        if not self._check_memory_availability():
            if PROFESSIONAL_LOGGING:
                professional_logger.warning("メモリ不足のため並列処理を制限します")
            self.n_jobs = min(self.n_jobs, 2)
        
        results = []
        
        if self.backend == 'threading':
            results = self._thread_parallel(func, data_chunks, *args, **kwargs)
        elif self.backend == 'multiprocessing':
            results = self._process_parallel(func, data_chunks, *args, **kwargs)
        else:
            # フォールバック: シーケンシャル処理
            results = [func(chunk, *args, **kwargs) for chunk in data_chunks]
        
        return results
    
    def _thread_parallel(self, func: Callable, data_chunks: List[Any], 
                        *args, **kwargs) -> List[Any]:
        """スレッド並列処理"""
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(func, chunk, *args, **kwargs) 
                      for chunk in data_chunks]
            
            results = []
            for future in tqdm(as_completed(futures), 
                             total=len(futures), 
                             desc="🔄 Thread Parallel Processing"):
                try:
                    result = future.result(timeout=300)  # 5分タイムアウト
                    results.append(result)
                except Exception as e:
                    if PROFESSIONAL_LOGGING:
                        professional_logger.error(f"スレッド処理エラー: {e}")
                    results.append(None)
        
        return results
    
    def _process_parallel(self, func: Callable, data_chunks: List[Any], 
                         *args, **kwargs) -> List[Any]:
        """プロセス並列処理"""
        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(func, chunk, *args, **kwargs) 
                          for chunk in data_chunks]
                
                results = []
                for future in tqdm(as_completed(futures), 
                                 total=len(futures), 
                                 desc="🚀 Process Parallel Processing"):
                    try:
                        result = future.result(timeout=600)  # 10分タイムアウト
                        results.append(result)
                    except Exception as e:
                        if PROFESSIONAL_LOGGING:
                            professional_logger.error(f"プロセス処理エラー: {e}")
                        results.append(None)
            
            return results
        
        except Exception as e:
            if PROFESSIONAL_LOGGING:
                professional_logger.warning(f"プロセス並列処理失敗、スレッド並列に切り替え: {e}")
            return self._thread_parallel(func, data_chunks, *args, **kwargs)
    
    def _check_memory_availability(self) -> bool:
        """メモリ使用可能性チェック"""
        memory = psutil.virtual_memory()
        available_ratio = memory.available / memory.total
        return available_ratio > (1 - self.max_memory_usage)
    
    def optimize_dataframe_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """データフレーム操作最適化"""
        
        if PROFESSIONAL_LOGGING:
            professional_logger.info("データフレーム最適化開始", shape=df.shape)
        
        # メモリ使用量最適化
        optimized_df = self._optimize_memory_usage(df.copy())
        
        # カテゴリカル変数の最適化
        optimized_df = self._optimize_categorical_columns(optimized_df)
        
        # 欠損値処理の最適化
        optimized_df = self._optimize_missing_values(optimized_df)
        
        if PROFESSIONAL_LOGGING:
            memory_saved = df.memory_usage(deep=True).sum() - optimized_df.memory_usage(deep=True).sum()
            professional_logger.info("データフレーム最適化完了", 
                                   memory_saved_mb=memory_saved / (1024**2))
        
        return optimized_df
    
    def _optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """メモリ使用量最適化"""
        
        for col in df.columns:
            if df[col].dtype == 'object':
                continue
                
            # 整数型の最適化
            if df[col].dtype.kind in 'biufc':
                if df[col].dtype == 'int64':
                    if df[col].min() >= -128 and df[col].max() <= 127:
                        df[col] = df[col].astype('int8')
                    elif df[col].min() >= -32768 and df[col].max() <= 32767:
                        df[col] = df[col].astype('int16')
                    elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                        df[col] = df[col].astype('int32')
                
                # 浮動小数点型の最適化
                elif df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    def _optimize_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """カテゴリカル変数最適化"""
        
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # 50%未満がユニークな場合
                df[col] = df[col].astype('category')
        
        return df
    
    def _optimize_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値処理最適化"""
        
        # スパース配列の活用
        for col in df.columns:
            if df[col].isnull().sum() > len(df) * 0.9:  # 90%以上が欠損
                try:
                    df[col] = df[col].astype(pd.SparseDtype(df[col].dtype))
                except:
                    pass  # エラーが出ても続行
        
        return df

class OptimizedStatistics:
    """最適化された統計計算"""
    
    def __init__(self, use_numba: bool = NUMBA_AVAILABLE):
        self.use_numba = use_numba and NUMBA_AVAILABLE
        
        if PROFESSIONAL_LOGGING:
            professional_logger.info("最適化統計システム初期化", use_numba=self.use_numba)
    
    def fast_correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """高速相関行列計算"""
        if self.use_numba:
            return self._fast_correlation_matrix_numba(data)
        else:
            return np.corrcoef(data.T)
    
    def _fast_correlation_matrix_numba(self, data: np.ndarray) -> np.ndarray:
        """Numba最適化相関行列計算"""
        if not NUMBA_AVAILABLE:
            return np.corrcoef(data.T)
        
        @jit(nopython=True, parallel=True)
        def _compute_correlation(data):
            n_vars = data.shape[1]
            corr_matrix = np.zeros((n_vars, n_vars))
            
            # 平均の計算
            means = np.mean(data, axis=0)
            
            # 相関係数の計算
            for i in prange(n_vars):
                for j in prange(i, n_vars):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        # 共分散の計算
                        cov = np.mean((data[:, i] - means[i]) * (data[:, j] - means[j]))
                        # 標準偏差の計算
                        std_i = np.sqrt(np.mean((data[:, i] - means[i]) ** 2))
                        std_j = np.sqrt(np.mean((data[:, j] - means[j]) ** 2))
                        
                        if std_i > 0 and std_j > 0:
                            corr_matrix[i, j] = cov / (std_i * std_j)
                            corr_matrix[j, i] = corr_matrix[i, j]
                        else:
                            corr_matrix[i, j] = 0.0
                            corr_matrix[j, i] = 0.0
            
            return corr_matrix
        
        return _compute_correlation(data)
    
    def fast_percentile(self, data: np.ndarray, percentile: float) -> float:
        """高速パーセンタイル計算"""
        if self.use_numba:
            return self._fast_percentile_numba(data, percentile)
        else:
            return np.percentile(data, percentile)
    
    def _fast_percentile_numba(self, data: np.ndarray, percentile: float) -> float:
        """Numba最適化パーセンタイル計算"""
        if not NUMBA_AVAILABLE:
            return np.percentile(data, percentile)
        
        @jit(nopython=True)
        def _compute_percentile(data, percentile):
            sorted_data = np.sort(data.flatten())
            n = len(sorted_data)
            index = (n - 1) * percentile / 100.0
            
            if index == int(index):
                return sorted_data[int(index)]
            else:
                lower = sorted_data[int(index)]
                upper = sorted_data[int(index) + 1]
                return lower + (upper - lower) * (index - int(index))
        
        return _compute_percentile(data, percentile)
    
    def compute_descriptive_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """最適化された記述統計計算"""
        
        if self.use_numba:
            return self._compute_stats_numba(data)
        else:
            return self._compute_stats_numpy(data)
    
    def _compute_stats_numba(self, data: np.ndarray) -> Dict[str, float]:
        """Numba最適化統計計算"""
        if not NUMBA_AVAILABLE:
            return self._compute_stats_numpy(data)
        
        @jit(nopython=True)
        def _compute_stats(data):
            flat_data = data.flatten()
            n = len(flat_data)
            
            # 基本統計
            mean = np.mean(flat_data)
            std = np.std(flat_data)
            var = np.var(flat_data)
            min_val = np.min(flat_data)
            max_val = np.max(flat_data)
            
            # 歪度・尖度（簡易版）
            centered = flat_data - mean
            skewness = np.mean(centered**3) / (std**3) if std > 0 else 0.0
            kurtosis = np.mean(centered**4) / (std**4) - 3.0 if std > 0 else 0.0
            
            return mean, std, var, min_val, max_val, skewness, kurtosis, float(n)
        
        mean, std, var, min_val, max_val, skewness, kurtosis, count = _compute_stats(data)
        
        return {
            'mean': mean,
            'std': std,
            'var': var,
            'min': min_val,
            'max': max_val,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'count': count
        }
    
    def _compute_stats_numpy(self, data: np.ndarray) -> Dict[str, float]:
        """NumPy統計計算"""
        flat_data = data.flatten()
        
        return {
            'mean': np.mean(flat_data),
            'std': np.std(flat_data),
            'var': np.var(flat_data),
            'min': np.min(flat_data),
            'max': np.max(flat_data),
            'median': np.median(flat_data),
            'q25': np.percentile(flat_data, 25),
            'q75': np.percentile(flat_data, 75),
            'skewness': float(pd.Series(flat_data).skew()),
            'kurtosis': float(pd.Series(flat_data).kurtosis()),
            'count': float(len(flat_data))
        }

class MemoryManager:
    """メモリ管理システム"""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.memory_warnings = []
        
        if PROFESSIONAL_LOGGING:
            professional_logger.info("メモリ管理システム初期化", 
                                   max_memory_percent=max_memory_percent)
    
    def monitor_memory(self) -> Dict[str, Any]:
        """メモリ監視"""
        memory = psutil.virtual_memory()
        
        memory_info = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent,
            'is_critical': memory.percent > self.max_memory_percent
        }
        
        if memory_info['is_critical']:
            self.memory_warnings.append({
                'timestamp': datetime.now(),
                'memory_percent': memory.percent,
                'message': f"メモリ使用率が{memory.percent:.1f}%に達しました"
            })
            
            if PROFESSIONAL_LOGGING:
                professional_logger.warning("メモリ使用率警告", 
                                          memory_percent=memory.percent)
        
        return memory_info
    
    def cleanup_memory(self):
        """メモリクリーンアップ"""
        if PROFESSIONAL_LOGGING:
            professional_logger.info("メモリクリーンアップ実行")
        
        # ガベージコレクション強制実行
        collected = gc.collect()
        
        # メモリ最適化のヒント
        if hasattr(gc, 'set_threshold'):
            gc.set_threshold(700, 10, 10)
        
        return {
            'objects_collected': collected,
            'memory_after_cleanup': psutil.virtual_memory().percent
        }
    
    def get_memory_recommendations(self) -> List[str]:
        """メモリ最適化推奨事項"""
        recommendations = []
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            recommendations.append("🚨 メモリ使用率が90%を超えています。データを分割して処理してください")
        elif memory.percent > 80:
            recommendations.append("⚠️ メモリ使用率が80%を超えています。不要なデータを削除してください")
        elif memory.percent > 70:
            recommendations.append("💡 メモリ使用率が70%を超えています。ガベージコレクションを実行してください")
        
        if len(self.memory_warnings) > 5:
            recommendations.append("📊 頻繁にメモリ警告が発生しています。データ処理方法を見直してください")
        
        return recommendations

class BatchProcessor:
    """バッチ処理システム"""
    
    def __init__(self, batch_size: int = 1000, progress_callback: Callable = None):
        self.batch_size = batch_size
        self.progress_callback = progress_callback
        self.processed_batches = 0
        self.failed_batches = 0
        
        if PROFESSIONAL_LOGGING:
            professional_logger.info("バッチ処理システム初期化", batch_size=batch_size)
    
    def process_in_batches(self, data: pd.DataFrame, 
                          processing_func: Callable,
                          *args, **kwargs) -> List[Any]:
        """バッチ処理実行"""
        
        total_rows = len(data)
        num_batches = (total_rows + self.batch_size - 1) // self.batch_size
        results = []
        
        if PROFESSIONAL_LOGGING:
            professional_logger.info("バッチ処理開始", 
                                   total_rows=total_rows, 
                                   num_batches=num_batches)
        
        with tqdm(total=num_batches, desc="📦 Batch Processing") as pbar:
            for i in range(0, total_rows, self.batch_size):
                batch_data = data.iloc[i:i + self.batch_size]
                
                try:
                    batch_result = processing_func(batch_data, *args, **kwargs)
                    results.append(batch_result)
                    self.processed_batches += 1
                    
                    if self.progress_callback:
                        self.progress_callback(i + len(batch_data), total_rows)
                
                except Exception as e:
                    self.failed_batches += 1
                    if PROFESSIONAL_LOGGING:
                        professional_logger.error(f"バッチ処理エラー (batch {i//self.batch_size}): {e}")
                    results.append(None)
                
                pbar.update(1)
                
                # メモリ使用量チェック
                if psutil.virtual_memory().percent > 85:
                    gc.collect()
        
        success_rate = (self.processed_batches / num_batches) * 100 if num_batches > 0 else 0
        
        if PROFESSIONAL_LOGGING:
            professional_logger.info("バッチ処理完了", 
                                   success_rate=success_rate,
                                   failed_batches=self.failed_batches)
        
        return results
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """バッチ処理統計"""
        total_batches = self.processed_batches + self.failed_batches
        
        return {
            'total_batches': total_batches,
            'successful_batches': self.processed_batches,
            'failed_batches': self.failed_batches,
            'success_rate': (self.processed_batches / total_batches * 100) if total_batches > 0 else 0,
            'current_batch_size': self.batch_size
        }

# グローバルインスタンス
parallel_processor = ParallelProcessor(backend='threading')
optimized_stats = OptimizedStatistics()
memory_manager = MemoryManager()
batch_processor = BatchProcessor()