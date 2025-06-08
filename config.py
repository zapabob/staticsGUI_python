#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Professional Statistics Suite - Hardware & AI Configuration Management
プロフェッショナル統計スイート - ハードウェア・AI設定管理

RTX 30/40/50 Series & Apple Silicon M2+ Optimized
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

# .envファイルを読み込み
load_dotenv()

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
        
        # Optimization settings
        self.optimal_settings = self._determine_optimal_settings()
    
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
                        'optimization_level': self._get_nvidia_optimization_level(device_props.name)
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
            if "M2" in chip_name or "M3" in chip_name:
                optimization_level = "high"
                neural_engine = True
            elif "M1" in chip_name:
                optimization_level = "medium"
                neural_engine = True
            
            return {
                'name': chip_name,
                'performance_cores': performance_cores,
                'efficiency_cores': efficiency_cores,
                'neural_engine': neural_engine,
                'optimization_level': optimization_level,
                'metal_support': True,
                'mlx_compatible': "M2" in chip_name or "M3" in chip_name
            }
            
        except Exception:
            return {
                'name': 'Apple Silicon (Unknown)',
                'optimization_level': 'medium',
                'metal_support': True,
                'mlx_compatible': False
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
        except ImportError:
            pass
        
        return cpu_info
    
    def _detect_memory(self) -> Dict[str, Any]:
        """メモリ情報検出"""
        memory_info = {'total_gb': 0, 'available_gb': 0}
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_info = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent_used': memory.percent
            }
        except ImportError:
            pass
        
        return memory_info
    
    def _determine_optimal_settings(self) -> Dict[str, Any]:
        """最適設定決定"""
        settings = {
            'framework': 'auto',
            'device': 'auto',
            'precision': 'float32',
            'batch_size': 'auto',
            'num_workers': 'auto',
            'memory_fraction': 0.8,
            'optimization_level': 'medium'
        }
        
        # GPU based optimization
        if self.gpu_info['nvidia']['available']:
            nvidia_devices = self.gpu_info['nvidia']['devices']
            if nvidia_devices:
                best_device = max(nvidia_devices, key=lambda x: x['memory_gb'])
                settings['framework'] = 'pytorch_cuda'
                settings['device'] = 'cuda'
                settings['optimization_level'] = best_device['optimization_level']
                
                # RTX 40/50 series optimizations
                if best_device['is_rtx_40_series'] or best_device['is_rtx_50_series']:
                    settings['precision'] = 'mixed'  # Use mixed precision
                    settings['memory_fraction'] = 0.9
                
        elif self.gpu_info['apple_metal']['available']:
            apple_device = self.gpu_info['apple_metal']['devices'][0]
            settings['framework'] = 'tensorflow_metal'
            settings['device'] = 'mps'  # Metal Performance Shaders
            settings['optimization_level'] = apple_device['optimization_level']
            
            # M2+ optimizations
            if apple_device.get('mlx_compatible', False):
                settings['framework'] = 'mlx'
                settings['precision'] = 'float16'
        
        # CPU optimizations
        cpu_cores = self.cpu_info.get('logical_cores', os.cpu_count())
        settings['num_workers'] = min(cpu_cores - 1, 8)  # Leave one core for system
        
        # Memory optimizations
        total_memory = self.memory_info.get('total_gb', 8)
        if total_memory >= 32:
            settings['batch_size'] = 'large'
        elif total_memory >= 16:
            settings['batch_size'] = 'medium'
        else:
            settings['batch_size'] = 'small'
        
        return settings
    
    def get_optimization_recommendations(self) -> List[str]:
        """最適化推奨事項取得"""
        recommendations = []
        
        # GPU recommendations
        if self.gpu_info['nvidia']['available']:
            nvidia_devices = self.gpu_info['nvidia']['devices']
            for device in nvidia_devices:
                if device['is_rtx_40_series'] or device['is_rtx_50_series']:
                    recommendations.append(f"🚀 {device['name']} detected: Use mixed precision training for maximum performance")
                    recommendations.append("💡 Enable CUDA memory optimization for large datasets")
                elif device['is_rtx_30_series']:
                    recommendations.append(f"⚡ {device['name']} detected: Optimize batch size for RTX 30 series")
        
        elif self.gpu_info['apple_metal']['available']:
            apple_device = self.gpu_info['apple_metal']['devices'][0]
            if apple_device.get('mlx_compatible', False):
                recommendations.append(f"🍎 {apple_device['name']} detected: Use MLX framework for native acceleration")
                recommendations.append("🔥 Enable Metal Performance Shaders for optimal performance")
            else:
                recommendations.append("🍎 Apple Silicon detected: Use TensorFlow Metal backend")
        
        # Memory recommendations
        total_memory = self.memory_info.get('total_gb', 0)
        if total_memory < 16:
            recommendations.append("⚠️ Consider upgrading to 16GB+ RAM for large dataset processing")
        elif total_memory >= 32:
            recommendations.append("💪 Sufficient RAM detected: Enable large batch processing")
        
        return recommendations

class AIConfig:
    """AI API設定管理クラス（ハードウェア最適化対応）"""
    
    def __init__(self):
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Hardware detection
        self.hardware = HardwareDetector()
        
        # API設定
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # MCP設定
        self.mcp_server_url = os.getenv("MCP_SERVER_URL", "ws://localhost:8080")
        self.mcp_enabled = os.getenv("MCP_ENABLED", "false").lower() == "true"
        
        # AI機能設定
        self.enable_vision = os.getenv("ENABLE_VISION", "true").lower() == "true"
        self.enable_nlp = os.getenv("ENABLE_NLP", "true").lower() == "true"
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4-vision-preview")
        
        # ハードウェア最適化設定
        self.gpu_acceleration = self.hardware.gpu_info['nvidia']['available'] or self.hardware.gpu_info['apple_metal']['available']
        self.optimal_framework = self.hardware.optimal_settings['framework']
        self.optimal_device = self.hardware.optimal_settings['device']
        self.optimal_precision = self.hardware.optimal_settings['precision']
        
        # 画像処理設定
        self.max_image_size = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB
        self.supported_formats = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"]
        
        # OCR設定
        self.tesseract_cmd = os.getenv("TESSERACT_CMD")
        self.ocr_languages = os.getenv("OCR_LANGUAGES", "eng+jpn").split("+")
        
        self._load_config()
    
    def _load_config(self):
        """設定ファイル読み込み"""
        config_file = self.config_dir / "ai_config.yaml"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    self._update_from_config(config)
            except Exception as e:
                print(f"設定ファイル読み込みエラー: {e}")
    
    def _update_from_config(self, config: Dict[str, Any]):
        """設定辞書から更新"""
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_config(self):
        """設定をファイルに保存"""
        config = {
            'mcp_server_url': self.mcp_server_url,
            'mcp_enabled': self.mcp_enabled,
            'enable_vision': self.enable_vision,
            'enable_nlp': self.enable_nlp,
            'default_model': self.default_model,
            'max_image_size': self.max_image_size,
            'supported_formats': self.supported_formats,
            'ocr_languages': self.ocr_languages,
            'hardware_info': {
                'gpu_info': self.hardware.gpu_info,
                'cpu_info': self.hardware.cpu_info,
                'memory_info': self.hardware.memory_info,
                'optimal_settings': self.hardware.optimal_settings
            }
        }
        
        config_file = self.config_dir / "ai_config.yaml"
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"設定保存エラー: {e}")
    
    def get_hardware_status(self) -> Dict[str, Any]:
        """ハードウェア状態取得"""
        return {
            'platform': self.hardware.platform,
            'architecture': self.hardware.architecture,
            'gpu_acceleration': self.gpu_acceleration,
            'optimal_framework': self.optimal_framework,
            'optimal_device': self.optimal_device,
            'gpu_devices': self.hardware.gpu_info,
            'cpu_cores': self.hardware.cpu_info.get('logical_cores', 'Unknown'),
            'total_memory_gb': round(self.hardware.memory_info.get('total_gb', 0), 1),
            'recommendations': self.hardware.get_optimization_recommendations()
        }
    
    def is_api_configured(self, provider: str) -> bool:
        """API設定確認"""
        if provider.lower() == "openai":
            return bool(self.openai_api_key)
        elif provider.lower() == "google":
            return bool(self.google_api_key)
        elif provider.lower() == "anthropic":
            return bool(self.anthropic_api_key)
        return False
    
    def get_available_providers(self) -> list:
        """利用可能なAPIプロバイダー一覧"""
        providers = []
        if self.is_api_configured("openai"):
            providers.append("OpenAI")
        if self.is_api_configured("google"):
            providers.append("Google")
        if self.is_api_configured("anthropic"):
            providers.append("Anthropic")
        return providers

# グローバル設定インスタンス
ai_config = AIConfig()

# プロンプトテンプレート（ハードウェア最適化対応）
STATISTICAL_ANALYSIS_PROMPTS = {
    'hardware_optimized_analysis': """
以下のデータについて、利用可能なハードウェアに最適化された分析を実行してください：

ハードウェア情報:
- GPU: {gpu_info}
- 推奨フレームワーク: {framework}
- 最適化レベル: {optimization_level}

データ情報:
- 形状: {shape}
- 列名: {columns}
- データ型: {dtypes}

分析要求: {analysis_request}

以下を含む最適化されたPythonコードを生成してください：
1. ハードウェア検出と設定
2. 最適化されたデータ処理
3. GPU/Apple Silicon加速の活用
4. メモリ効率的な処理
""",
    
    'data_description': """
以下のデータについて分析してください：

データ形状: {shape}
列名: {columns}
データ型: {dtypes}
欠損値: {missing_values}

以下の統計解析を実行し、結果をPythonコードとして返してください：
{analysis_request}

回答は以下の形式で：
1. 分析の説明
2. 実行するPythonコード
3. 結果の解釈
""",
    
    'gpu_accelerated_ml': """
GPU加速を活用した機械学習分析を実行してください：

利用可能GPU: {gpu_devices}
推奨設定: {optimal_settings}

データ: {data_info}
分析目標: {objective}

以下を含むコードを生成：
1. GPU検出と初期化
2. 最適化されたデータローダー
3. GPU加速モデル訓練
4. 結果の可視化
"""
}

# ハードウェア最適化モデル設定
HARDWARE_OPTIMIZED_CONFIGS = {
    'nvidia_rtx_50': {
        'precision': 'mixed',
        'batch_size_multiplier': 2.0,
        'memory_fraction': 0.95,
        'optimization_flags': ['--enable-tensor-cores', '--enable-flash-attention']
    },
    'nvidia_rtx_40': {
        'precision': 'mixed',
        'batch_size_multiplier': 1.8,
        'memory_fraction': 0.9,
        'optimization_flags': ['--enable-tensor-cores']
    },
    'nvidia_rtx_30': {
        'precision': 'float32',
        'batch_size_multiplier': 1.5,
        'memory_fraction': 0.85,
        'optimization_flags': ['--enable-tensor-cores']
    },
    'apple_m3': {
        'framework': 'mlx',
        'precision': 'float16',
        'batch_size_multiplier': 1.2,
        'metal_optimization': True
    },
    'apple_m2': {
        'framework': 'tensorflow_metal',
        'precision': 'float32',
        'batch_size_multiplier': 1.0,
        'metal_optimization': True
    }
}

# モデル設定（ハードウェア対応）
MODEL_CONFIGS = {
    'openai': {
        'gpt-4-vision-preview': {
            'supports_vision': True,
            'max_tokens': 4096,
            'temperature': 0.1
        },
        'gpt-4': {
            'supports_vision': False,
            'max_tokens': 8192,
            'temperature': 0.1
        },
        'gpt-3.5-turbo': {
            'supports_vision': False,
            'max_tokens': 4096,
            'temperature': 0.1
        }
    },
    'google': {
        'gemini-pro-vision': {
            'supports_vision': True,
            'max_tokens': 2048,
            'temperature': 0.1
        },
        'gemini-pro': {
            'supports_vision': False,
            'max_tokens': 2048,
            'temperature': 0.1
        }
    }
}

def get_hardware_summary() -> str:
    """ハードウェア概要取得"""
    hardware = ai_config.hardware
    summary = f"🖥️ Platform: {hardware.platform} ({hardware.architecture})\n"
    
    if hardware.gpu_info['nvidia']['available']:
        for device in hardware.gpu_info['nvidia']['devices']:
            summary += f"🎮 NVIDIA: {device['name']} ({device['memory_gb']:.1f}GB)\n"
    
    if hardware.gpu_info['apple_metal']['available']:
        device = hardware.gpu_info['apple_metal']['devices'][0]
        summary += f"🍎 Apple Silicon: {device['name']}\n"
    
    summary += f"🧠 CPU: {hardware.cpu_info.get('logical_cores', 'Unknown')} cores\n"
    summary += f"💾 Memory: {hardware.memory_info.get('total_gb', 0):.1f}GB\n"
    
    return summary 