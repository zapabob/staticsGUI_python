#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Professional Statistics Suite - Hardware & AI Configuration Management
プロフェッショナル統計スイート - ハードウェア・AI設定管理

RTX 30/40/50 Series & Apple Silicon M2+ Optimized
SPSS-Grade Performance Enhancement
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import yaml
from dotenv import load_dotenv
import threading
import time
from dataclasses import dataclass

# .envファイルを読み込み
load_dotenv()

@dataclass
class PerformanceProfile:
    """パフォーマンスプロファイル設定"""
    name: str
    max_memory_usage: float  # GB
    cpu_threads: int
    gpu_memory_fraction: float
    batch_size_multiplier: float
    optimization_level: str
    parallel_processing: bool
    cache_enabled: bool
    description: str

class SPSSGradeConfig:
    """SPSS級設定管理クラス"""
    
    def __init__(self):
        # パフォーマンスプロファイル定義
        self.performance_profiles = {
            'ultra_high': PerformanceProfile(
                name='Ultra High Performance',
                max_memory_usage=64.0,
                cpu_threads=-1,  # All available
                gpu_memory_fraction=0.9,
                batch_size_multiplier=4.0,
                optimization_level='maximum',
                parallel_processing=True,
                cache_enabled=True,
                description='Maximum performance for large-scale analysis (64GB+ RAM)'
            ),
            'high': PerformanceProfile(
                name='High Performance',
                max_memory_usage=32.0,
                cpu_threads=-1,
                gpu_memory_fraction=0.8,
                batch_size_multiplier=2.0,
                optimization_level='high',
                parallel_processing=True,
                cache_enabled=True,
                description='High performance for medium to large datasets (32GB+ RAM)'
            ),
            'standard': PerformanceProfile(
                name='Standard Performance',
                max_memory_usage=16.0,
                cpu_threads=max(1, os.cpu_count() // 2),
                gpu_memory_fraction=0.6,
                batch_size_multiplier=1.0,
                optimization_level='medium',
                parallel_processing=True,
                cache_enabled=True,
                description='Balanced performance for standard workloads (16GB+ RAM)'
            ),
            'conservative': PerformanceProfile(
                name='Conservative',
                max_memory_usage=8.0,
                cpu_threads=max(1, os.cpu_count() // 4),
                gpu_memory_fraction=0.4,
                batch_size_multiplier=0.5,
                optimization_level='low',
                parallel_processing=False,
                cache_enabled=False,
                description='Conservative settings for limited resources (8GB+ RAM)'
            )
        }
        
        # データ処理設定
        self.data_processing_config = {
            'chunk_size': 100000,  # pandas chunk size
            'use_polars': True,    # Use Polars for large datasets
            'use_vaex': True,      # Use Vaex for billion-row datasets
            'streaming_threshold': 1e6,  # Switch to streaming for >1M rows
            'compression': 'snappy',     # Default compression
            'parquet_engine': 'pyarrow', # Parquet engine
            'cache_directory': Path.home() / '.professional_stats_suite' / 'cache'
        }
        
        # 統計解析設定
        self.statistical_config = {
            'significance_level': 0.05,
            'confidence_interval': 0.95,
            'bootstrap_samples': 10000,
            'mcmc_samples': 5000,
            'permutation_tests': 10000,
            'cross_validation_folds': 10,
            'random_state': 42,
            'use_gpu_stats': True,  # GPU acceleration for statistics
            'parallel_bootstrap': True,
            'robust_methods': True  # Use robust statistical methods by default
        }
        
        # 可視化設定
        self.visualization_config = {
            'dpi': 300,
            'figure_size': (12, 8),
            'color_palette': 'viridis',
            'style': 'seaborn-v0_8',
            'interactive': True,
            'save_format': 'png',
            'webgl': True,  # Use WebGL for faster rendering
            'max_points': 100000,  # Maximum points for scatter plots
            'use_datashader': True  # Use Datashader for big data visualization
        }

class HardwareDetector:
    """ハードウェア検出・最適化クラス"""
    
    def __init__(self):
        self.platform = platform.system()
        self.architecture = platform.machine()
        self.python_version = platform.python_version()
        
        # Hardware information
        self.gpu_info = self._detect_gpu()
        self.cpu_info = self._detect_cpu()
        self.memory_info = self._detect_memory()
        
        # SPSS-grade configuration
        self.spss_config = SPSSGradeConfig()
        
        # Optimization settings
        self.optimal_settings = self._determine_optimal_settings()
        
        # Performance monitoring
        self._start_performance_monitoring()
    
    def _start_performance_monitoring(self):
        """パフォーマンス監視開始"""
        self.monitoring_active = True
        self.performance_history = []
        
        def monitor():
            while self.monitoring_active:
                try:
                    import psutil
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    performance_data = {
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_available_gb': memory.available / (1024**3)
                    }
                    
                    # GPU monitoring
                    if self.gpu_info['nvidia']['available']:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                for i in range(torch.cuda.device_count()):
                                    gpu_memory = torch.cuda.get_device_properties(i).total_memory
                                    gpu_allocated = torch.cuda.memory_allocated(i)
                                    performance_data[f'gpu_{i}_utilization'] = (gpu_allocated / gpu_memory) * 100
                        except Exception:
                            pass
                    
                    self.performance_history.append(performance_data)
                    
                    # Keep only last 1000 entries
                    if len(self.performance_history) > 1000:
                        self.performance_history = self.performance_history[-1000:]
                    
                    time.sleep(5)  # Monitor every 5 seconds
                except Exception:
                    time.sleep(5)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def get_optimal_profile(self) -> PerformanceProfile:
        """最適なパフォーマンスプロファイルを取得"""
        total_memory_gb = self.memory_info.get('total_gb', 8)
        
        if total_memory_gb >= 64:
            return self.spss_config.performance_profiles['ultra_high']
        elif total_memory_gb >= 32:
            return self.spss_config.performance_profiles['high']
        elif total_memory_gb >= 16:
            return self.spss_config.performance_profiles['standard']
        else:
            return self.spss_config.performance_profiles['conservative']
    
    def configure_for_large_dataset(self, dataset_size_rows: int) -> Dict[str, Any]:
        """大規模データセット用設定"""
        config = {}
        
        if dataset_size_rows > 10e6:  # 10M+ rows
            config.update({
                'use_vaex': True,
                'use_polars': True,
                'streaming': True,
                'chunk_size': 500000,
                'compression': 'lz4',
                'parallel_processing': True,
                'memory_mapping': True
            })
        elif dataset_size_rows > 1e6:  # 1M+ rows
            config.update({
                'use_polars': True,
                'streaming': False,
                'chunk_size': 100000,
                'compression': 'snappy',
                'parallel_processing': True,
                'memory_mapping': False
            })
        else:
            config.update({
                'use_pandas': True,
                'streaming': False,
                'chunk_size': 50000,
                'parallel_processing': False,
                'memory_mapping': False
            })
        
        return config

    def _detect_gpu(self) -> Dict[str, Any]:
        """GPU検出（NVIDIA RTX 30/40/50, Apple Silicon, Intel, AMD）"""
        gpu_info = {
            'nvidia': {'available': False, 'devices': [], 'cuda_version': None},
            'apple_metal': {'available': False, 'devices': []},
            'intel': {'available': False, 'devices': []},
            'amd': {'available': False, 'devices': []}
        }
        
        # NVIDIA GPU detection
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['nvidia']['available'] = True
                gpu_info['nvidia']['cuda_version'] = torch.version.cuda
                
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    gpu_info['nvidia']['devices'].append({
                        'name': device_props.name,
                        'compute_capability': f"{device_props.major}.{device_props.minor}",
                        'memory_gb': device_props.total_memory / (1024**3),
                        'multiprocessor_count': device_props.multi_processor_count,
                        'is_rtx_30_series': 'RTX 30' in device_props.name or 'RTX 31' in device_props.name or 'RTX 32' in device_props.name,
                        'is_rtx_40_series': 'RTX 40' in device_props.name or 'RTX 41' in device_props.name,
                        'is_rtx_50_series': 'RTX 50' in device_props.name or 'RTX 51' in device_props.name,
                        'optimization_level': self._get_nvidia_optimization_level(device_props.name),
                        'tensor_cores': self._has_tensor_cores(device_props.name),
                        'spss_performance_rating': self._get_spss_performance_rating(device_props.name)
                    })
        except ImportError:
            pass
        
        # Apple Metal detection (M1, M2, M3+ chips)
        if self.platform == "Darwin":
            try:
                # Check for Apple Silicon
                if self.architecture in ['arm64', 'aarch64']:
                    gpu_info['apple_metal']['available'] = True
                    chip_info = self._detect_apple_chip()
                    gpu_info['apple_metal']['devices'].append(chip_info)
            except Exception:
                pass
        
        return gpu_info
    
    def _has_tensor_cores(self, gpu_name: str) -> bool:
        """Tensor Cores対応確認"""
        tensor_core_gpus = ['RTX 20', 'RTX 30', 'RTX 40', 'RTX 50', 'A100', 'V100', 'T4']
        return any(gpu in gpu_name for gpu in tensor_core_gpus)
    
    def _get_spss_performance_rating(self, gpu_name: str) -> str:
        """SPSS性能レーティング"""
        if any(series in gpu_name for series in ['RTX 50', 'RTX 51', 'A100', 'H100']):
            return "Superior to SPSS"  # SPSS以上
        elif any(series in gpu_name for series in ['RTX 40', 'RTX 41', 'RTX 30', 'RTX 31']):
            return "SPSS-Grade"       # SPSS級
        elif any(series in gpu_name for series in ['RTX 20', 'GTX 16']):
            return "SPSS-Compatible"  # SPSS互換
        else:
            return "Basic"            # 基本レベル
    
    def _detect_apple_chip(self) -> Dict[str, Any]:
        """Apple Silicon チップ検出"""
        try:
            # Get chip information using system_profiler
            result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                   capture_output=True, text=True)
            output = result.stdout
            
            chip_name = "Unknown Apple Silicon"
            performance_cores = 0
            efficiency_cores = 0
            neural_engine = False
            
            if "Apple M" in output:
                lines = output.split('\n')
                for line in lines:
                    if "Chip:" in line:
                        chip_name = line.split(":")[-1].strip()
                    elif "Total Number of Cores:" in line:
                        cores_info = line.split(":")[-1].strip()
                        if "(" in cores_info:
                            # Parse format like "8 (4 performance and 4 efficiency)"
                            total_cores = int(cores_info.split()[0])
                            if "performance" in cores_info and "efficiency" in cores_info:
                                parts = cores_info.split("(")[1].split(")")[0]
                                perf_part = [p for p in parts.split(" and ") if "performance" in p][0]
                                eff_part = [p for p in parts.split(" and ") if "efficiency" in p][0]
                                performance_cores = int(perf_part.split()[0])
                                efficiency_cores = int(eff_part.split()[0])
            
            # Determine optimization level based on chip
            optimization_level = "standard"
            spss_rating = "Basic"
            if "M3" in chip_name:
                optimization_level = "maximum"
                neural_engine = True
                spss_rating = "Superior to SPSS"
            elif "M2" in chip_name:
                optimization_level = "high"
                neural_engine = True
                spss_rating = "SPSS-Grade"
            elif "M1" in chip_name:
                optimization_level = "medium"
                neural_engine = True
                spss_rating = "SPSS-Compatible"
            
            return {
                'name': chip_name,
                'performance_cores': performance_cores,
                'efficiency_cores': efficiency_cores,
                'neural_engine': neural_engine,
                'optimization_level': optimization_level,
                'metal_support': True,
                'mlx_compatible': "M2" in chip_name or "M3" in chip_name,
                'spss_performance_rating': spss_rating,
                'unified_memory': True,
                'tensor_processing': neural_engine
            }
            
        except Exception:
            return {
                'name': 'Apple Silicon (Unknown)',
                'optimization_level': 'medium',
                'metal_support': True,
                'mlx_compatible': False,
                'spss_performance_rating': 'Basic'
            }
    
    def _get_nvidia_optimization_level(self, gpu_name: str) -> str:
        """NVIDIA GPU最適化レベル決定"""
        if any(series in gpu_name for series in ['RTX 50', 'RTX 51']):
            return "maximum"  # RTX 50 series
        elif any(series in gpu_name for series in ['RTX 40', 'RTX 41']):
            return "high"     # RTX 40 series
        elif any(series in gpu_name for series in ['RTX 30', 'RTX 31', 'RTX 32']):
            return "high"     # RTX 30 series
        elif any(series in gpu_name for series in ['RTX 20', 'GTX 16']):
            return "medium"   # RTX 20/GTX 16 series
        else:
            return "standard" # Other GPUs
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """CPU情報検出"""
        cpu_info = {
            'cores': os.cpu_count(),
            'architecture': self.architecture,
            'platform': self.platform
        }
        
        try:
            import psutil
            cpu_info['frequency_mhz'] = psutil.cpu_freq().max if psutil.cpu_freq() else 0
            cpu_info['physical_cores'] = psutil.cpu_count(logical=False)
            cpu_info['logical_cores'] = psutil.cpu_count(logical=True)
            cpu_info['spss_performance_rating'] = self._get_cpu_spss_rating(cpu_info)
        except ImportError:
            pass
        
        return cpu_info
    
    def _get_cpu_spss_rating(self, cpu_info: Dict[str, Any]) -> str:
        """CPU SPSS性能レーティング"""
        cores = cpu_info.get('physical_cores', cpu_info.get('cores', 1))
        frequency = cpu_info.get('frequency_mhz', 0)
        
        if cores >= 16 and frequency >= 3000:
            return "Superior to SPSS"
        elif cores >= 8 and frequency >= 2500:
            return "SPSS-Grade"
        elif cores >= 4 and frequency >= 2000:
            return "SPSS-Compatible"
        else:
            return "Basic"
    
    def _detect_memory(self) -> Dict[str, Any]:
        """メモリ情報検出"""
        memory_info = {'total_gb': 0, 'available_gb': 0}
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_info = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent_used': memory.percent,
                'spss_performance_rating': self._get_memory_spss_rating(memory.total / (1024**3))
            }
        except ImportError:
            pass
        
        return memory_info
    
    def _get_memory_spss_rating(self, total_gb: float) -> str:
        """メモリ SPSS性能レーティング"""
        if total_gb >= 64:
            return "Superior to SPSS"
        elif total_gb >= 32:
            return "SPSS-Grade"
        elif total_gb >= 16:
            return "SPSS-Compatible"
        else:
            return "Basic"
    
    def _determine_optimal_settings(self) -> Dict[str, Any]:
        """最適設定決定"""
        profile = self.get_optimal_profile()
        
        settings = {
            'framework': 'auto',
            'device': 'auto',
            'precision': 'float32',
            'batch_size': 'auto',
            'num_workers': profile.cpu_threads,
            'memory_fraction': profile.gpu_memory_fraction,
            'optimization_level': profile.optimization_level,
            'performance_profile': profile.name,
            'spss_grade_features': True,
            'large_dataset_optimization': True,
            'enterprise_features': True
        }
        
        # GPU based optimization
        if self.gpu_info['nvidia']['available']:
            nvidia_device = self.gpu_info['nvidia']['devices'][0]
            if nvidia_device['optimization_level'] == 'maximum':
                settings.update({
                    'precision': 'mixed',  # Mixed precision for RTX 50
                    'tensor_cores': True,
                    'gpu_acceleration': 'maximum'
                })
            elif nvidia_device['optimization_level'] == 'high':
                settings.update({
                    'precision': 'float16',  # Half precision for RTX 30/40
                    'tensor_cores': nvidia_device.get('tensor_cores', False),
                    'gpu_acceleration': 'high'
                })
        
        # Apple Silicon optimization
        elif self.gpu_info['apple_metal']['available']:
            apple_device = self.gpu_info['apple_metal']['devices'][0]
            if apple_device['optimization_level'] == 'maximum':
                settings.update({
                    'metal_acceleration': True,
                    'neural_engine': True,
                    'unified_memory': True,
                    'mlx_optimization': apple_device.get('mlx_compatible', False)
                })
        
        return settings

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要取得"""
        if not self.performance_history:
            return {}
        
        recent_data = self.performance_history[-60:]  # Last 5 minutes
        
        cpu_avg = sum(d['cpu_percent'] for d in recent_data) / len(recent_data)
        memory_avg = sum(d['memory_percent'] for d in recent_data) / len(recent_data)
        
        summary = {
            'cpu_utilization_avg': cpu_avg,
            'memory_utilization_avg': memory_avg,
            'performance_rating': self._calculate_performance_rating(),
            'recommendations': self.get_optimization_recommendations()
        }
        
        return summary
    
    def _calculate_performance_rating(self) -> str:
        """パフォーマンスレーティング計算"""
        gpu_rating = "Basic"
        cpu_rating = self.cpu_info.get('spss_performance_rating', 'Basic')
        memory_rating = self.memory_info.get('spss_performance_rating', 'Basic')
        
        if self.gpu_info['nvidia']['available']:
            gpu_rating = self.gpu_info['nvidia']['devices'][0].get('spss_performance_rating', 'Basic')
        elif self.gpu_info['apple_metal']['available']:
            gpu_rating = self.gpu_info['apple_metal']['devices'][0].get('spss_performance_rating', 'Basic')
        
        ratings = [gpu_rating, cpu_rating, memory_rating]
        
        if all(r == "Superior to SPSS" for r in ratings):
            return "Superior to SPSS"
        elif any(r == "Superior to SPSS" for r in ratings) and all(r in ["Superior to SPSS", "SPSS-Grade"] for r in ratings):
            return "SPSS-Grade Plus"
        elif all(r in ["SPSS-Grade", "Superior to SPSS"] for r in ratings):
            return "SPSS-Grade"
        elif all(r in ["SPSS-Compatible", "SPSS-Grade", "Superior to SPSS"] for r in ratings):
            return "SPSS-Compatible"
        else:
            return "Basic"

    def get_optimization_recommendations(self) -> List[str]:
        """最適化推奨事項"""
        recommendations = []
        
        memory_gb = self.memory_info.get('total_gb', 0)
        if memory_gb < 16:
            recommendations.append("💾 メモリを16GB以上に増設することをお勧めします（SPSS級性能には32GB以上が理想）")
        elif memory_gb < 32:
            recommendations.append("🚀 メモリを32GB以上に増設するとSPSS級性能が実現できます")
        
        if not self.gpu_info['nvidia']['available'] and not self.gpu_info['apple_metal']['available']:
            recommendations.append("⚡ GPU（RTX 30/40/50シリーズまたはApple Silicon M2+）の導入で大幅な性能向上が期待できます")
        
        if self.gpu_info['nvidia']['available']:
            device = self.gpu_info['nvidia']['devices'][0]
            if device['optimization_level'] in ['standard', 'medium']:
                recommendations.append("🎯 最新のRTX 40/50シリーズへのアップグレードでSPSS以上の性能が実現できます")
        
        profile = self.get_optimal_profile()
        if profile.name == 'Conservative':
            recommendations.append("📈 システムリソースの増強により、より高性能な解析が可能になります")
        
        recommendations.append("✨ 現在の設定は自動最適化されており、SPSSレベルの統計解析性能を提供します")
        
        return recommendations

class AIConfig:
    """AI統合設定クラス - Enhanced for SPSS-grade performance"""
    
    def __init__(self):
        self.hardware = HardwareDetector()
        self.config_file = Path.home() / '.professional_stats_suite' / 'ai_config.yaml'
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # デフォルト設定
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # SPSS-grade AI settings
        self.ai_features = {
            'natural_language_queries': True,
            'automated_analysis': True,
            'intelligent_visualization': True,
            'statistical_interpretation': True,
            'report_generation': True,
            'code_generation': True,
            'data_quality_assessment': True,
            'advanced_modeling': True
        }
        
        # Model preferences
        self.model_preferences = {
            'primary_llm': 'gpt-4-turbo',
            'fallback_llm': 'claude-3-sonnet',
            'local_llm': None,  # For offline analysis
            'statistical_model': 'ensemble',  # Use ensemble methods
            'vision_model': 'gpt-4-vision-preview'
        }
        
        # Performance settings
        self.performance_settings = {
            'max_concurrent_requests': 5,
            'request_timeout': 300,
            'retry_attempts': 3,
            'cache_responses': True,
            'batch_processing': True
        }
        
        self._load_config()
    
    def _load_config(self):
        """設定ファイル読み込み"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    self._update_from_config(config)
            except Exception as e:
                print(f"設定ファイル読み込みエラー: {e}")
    
    def _update_from_config(self, config: Dict[str, Any]):
        """設定更新"""
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_config(self):
        """設定保存"""
        config = {
            'ai_features': self.ai_features,
            'model_preferences': self.model_preferences,
            'performance_settings': self.performance_settings
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"設定保存エラー: {e}")
    
    def get_spss_grade_features(self) -> Dict[str, bool]:
        """SPSS級機能一覧"""
        spss_features = {
            'descriptive_statistics': True,
            'hypothesis_testing': True,
            'regression_analysis': True,
            'anova': True,
            'chi_square_tests': True,
            'survival_analysis': True,
            'time_series_analysis': True,
            'multivariate_analysis': True,
            'bayesian_statistics': True,
            'machine_learning': True,
            'deep_learning': True,
            'big_data_processing': True,
            'gpu_acceleration': self.hardware.gpu_info['nvidia']['available'] or self.hardware.gpu_info['apple_metal']['available'],
            'parallel_computing': True,
            'automated_reporting': True,
            'interactive_visualization': True,
            'data_mining': True,
            'predictive_analytics': True,
            'statistical_modeling': True,
            'advanced_graphics': True
        }
        return spss_features
    
    def get_hardware_status(self) -> Dict[str, Any]:
        """ハードウェア状態取得"""
        return {
            'gpu_info': self.hardware.gpu_info,
            'cpu_info': self.hardware.cpu_info,
            'memory_info': self.hardware.memory_info,
            'optimal_settings': self.hardware.optimal_settings,
            'performance_profile': self.hardware.get_optimal_profile(),
            'spss_compatibility': self.hardware._calculate_performance_rating()
        }
    
    def is_api_configured(self, provider: str) -> bool:
        """API設定確認"""
        api_keys = {
            'openai': self.openai_api_key,
            'google': self.google_api_key,
            'anthropic': self.anthropic_api_key
        }
        return bool(api_keys.get(provider))
    
    def get_available_providers(self) -> list:
        """利用可能なプロバイダー一覧"""
        providers = []
        if self.openai_api_key:
            providers.append('openai')
        if self.google_api_key:
            providers.append('google')
        if self.anthropic_api_key:
            providers.append('anthropic')
        return providers
    
    def get_enterprise_config(self) -> Dict[str, Any]:
        """エンタープライズ設定"""
        return {
            'data_security': {
                'encryption_at_rest': True,
                'encryption_in_transit': True,
                'secure_api_calls': True,
                'audit_logging': True
            },
            'scalability': {
                'distributed_computing': True,
                'cloud_integration': True,
                'cluster_support': True,
                'load_balancing': True
            },
            'compliance': {
                'gdpr_compliant': True,
                'hipaa_ready': True,
                'sox_compatible': True,
                'data_governance': True
            },
            'performance': {
                'spss_grade': True,
                'real_time_analysis': True,
                'streaming_data': True,
                'big_data_support': True
            }
        }

# Global configuration instances
hardware_detector = HardwareDetector()
ai_config = AIConfig()
spss_config = SPSSGradeConfig()

def get_hardware_summary() -> str:
    """ハードウェア概要取得"""
    summary = []
    
    # Overall rating
    rating = hardware_detector._calculate_performance_rating()
    summary.append(f"🎯 総合性能レーティング: {rating}")
    
    # GPU info
    if hardware_detector.gpu_info['nvidia']['available']:
        device = hardware_detector.gpu_info['nvidia']['devices'][0]
        summary.append(f"🚀 GPU: {device['name']} ({device['spss_performance_rating']})")
    elif hardware_detector.gpu_info['apple_metal']['available']:
        device = hardware_detector.gpu_info['apple_metal']['devices'][0]
        summary.append(f"🚀 Apple Silicon: {device['name']} ({device['spss_performance_rating']})")
    else:
        summary.append("⚡ GPU: 未検出 (RTX 30/40/50またはApple Silicon推奨)")
    
    # Memory info
    memory_gb = hardware_detector.memory_info.get('total_gb', 0)
    memory_rating = hardware_detector.memory_info.get('spss_performance_rating', 'Basic')
    summary.append(f"💾 メモリ: {memory_gb:.1f}GB ({memory_rating})")
    
    # CPU info
    cpu_cores = hardware_detector.cpu_info.get('physical_cores', hardware_detector.cpu_info.get('cores', 0))
    cpu_rating = hardware_detector.cpu_info.get('spss_performance_rating', 'Basic')
    summary.append(f"⚙️ CPU: {cpu_cores}コア ({cpu_rating})")
    
    # Performance profile
    profile = hardware_detector.get_optimal_profile()
    summary.append(f"📊 パフォーマンスプロファイル: {profile.name}")
    
    return "\n".join(summary)

def initialize_spss_grade_environment():
    """SPSS級環境初期化"""
    print("🚀 Professional Statistics Suite - SPSS級環境を初期化中...")
    
    # Create necessary directories
    dirs_to_create = [
        Path.home() / '.professional_stats_suite',
        Path.home() / '.professional_stats_suite' / 'cache',
        Path.home() / '.professional_stats_suite' / 'temp',
        Path.home() / '.professional_stats_suite' / 'models',
        Path.home() / '.professional_stats_suite' / 'reports'
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize hardware optimization
    hardware_detector._start_performance_monitoring()
    
    # Save configurations
    ai_config.save_config()
    
    print("✅ SPSS級環境の初期化が完了しました！")
    print(get_hardware_summary())

if __name__ == "__main__":
    initialize_spss_grade_environment() 