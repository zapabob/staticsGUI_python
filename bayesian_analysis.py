#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep Bayesian Statistical Analysis Module
ベイズ統計解析深化モジュール

Author: Ryo Minegishi
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime
import scipy.stats as stats
from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp

# ベイズ統計専門ライブラリ
try:
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt
    PYMC_AVAILABLE = True
except ImportError:
    try:
        import pymc3 as pm
        import arviz as az
        import theano.tensor as tt
        pt = tt  # PyMC3互換
        PYMC_AVAILABLE = True
    except ImportError:
        PYMC_AVAILABLE = False

# 機械学習ライブラリ
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# プロフェッショナル機能インポート
try:
    from professional_utils import professional_logger, performance_monitor
    PROFESSIONAL_LOGGING = True
except ImportError:
    PROFESSIONAL_LOGGING = False

warnings.filterwarnings('ignore')

class DeepBayesianAnalyzer:
    """深化ベイズ統計解析クラス"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.results_cache = {}
        self.fitted_models = {}
        self.traces = {}
        
        if not PYMC_AVAILABLE:
            print("⚠️ PyMCが利用できません。基本的な機能のみ提供されます。")
        
        if PROFESSIONAL_LOGGING:
            professional_logger.info("深化ベイズ統計解析システム初期化", 
                                   random_seed=random_seed)
    
    @performance_monitor.monitor_function("bayesian_linear_regression") if PROFESSIONAL_LOGGING else lambda x: x
    def bayesian_linear_regression(self, data: pd.DataFrame, target_col: str, 
                                 predictor_cols: List[str], 
                                 n_samples: int = 2000) -> Dict[str, Any]:
        """ベイズ線形回帰"""
        
        if not PYMC_AVAILABLE:
            return {'error': 'PyMCライブラリが必要です'}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Bayesian Linear Regression',
            'n_samples': n_samples
        }
        
        # データ準備
        analysis_cols = [target_col] + predictor_cols
        clean_data = data[analysis_cols].dropna()
        
        X = clean_data[predictor_cols].values
        y = clean_data[target_col].values
        
        # 標準化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        try:
            with pm.Model() as linear_model:
                # 事前分布
                alpha = pm.Normal('intercept', mu=0, sigma=1)
                beta = pm.Normal('coefficients', mu=0, sigma=1, shape=X_scaled.shape[1])
                sigma = pm.HalfNormal('noise', sigma=1)
                
                # 線形予測子
                mu = alpha + pm.math.dot(X_scaled, beta)
                
                # 尤度
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_scaled)
                
                # サンプリング
                trace = pm.sample(n_samples, tune=1000, return_inferencedata=True,
                                random_seed=self.random_seed, cores=1, 
                                progressbar=False)
                
                # 事後予測チェック
                ppc = pm.sample_posterior_predictive(trace, random_seed=self.random_seed)
            
            self.fitted_models['bayesian_linear'] = linear_model
            self.traces['bayesian_linear'] = trace
            
            # 結果の要約
            summary = az.summary(trace, var_names=['intercept', 'coefficients', 'noise'])
            
            # 信頼区間
            hdi = az.hdi(trace, hdi_prob=0.95)
            
            # 係数の解釈
            coefficients_summary = {}
            for i, col in enumerate(predictor_cols):
                coef_samples = trace.posterior['coefficients'].values[:, :, i].flatten()
                coefficients_summary[col] = {
                    'mean': float(np.mean(coef_samples)),
                    'std': float(np.std(coef_samples)),
                    'hdi_2.5%': float(hdi['coefficients'].values[i, 0]),
                    'hdi_97.5%': float(hdi['coefficients'].values[i, 1]),
                    'probability_positive': float(np.mean(coef_samples > 0)),
                    'probability_negative': float(np.mean(coef_samples < 0))
                }
            
            results['coefficients'] = coefficients_summary
            results['intercept'] = {
                'mean': float(summary.loc['intercept', 'mean']),
                'std': float(summary.loc['intercept', 'sd']),
                'hdi_2.5%': float(hdi['intercept'].values[0]),
                'hdi_97.5%': float(hdi['intercept'].values[1])
            }
            
            # モデル診断
            results['diagnostics'] = {
                'r_hat_max': float(summary['r_hat'].max()),
                'ess_bulk_min': float(summary['ess_bulk'].min()),
                'converged': bool(summary['r_hat'].max() < 1.1),
                'effective_sample_size_ok': bool(summary['ess_bulk'].min() > 400)
            }
            
            # 予測性能
            y_pred_mean = np.mean(ppc.posterior_predictive['y_obs'], axis=(0, 1))
            r2_bayesian = 1 - np.var(y_scaled - y_pred_mean) / np.var(y_scaled)
            
            results['model_performance'] = {
                'bayesian_r2': float(r2_bayesian),
                'posterior_predictive_mean': y_pred_mean.tolist()[:10],  # 最初の10個
                'prediction_uncertainty': float(np.std(ppc.posterior_predictive['y_obs']))
            }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    @performance_monitor.monitor_function("bayesian_logistic_regression") if PROFESSIONAL_LOGGING else lambda x: x
    def bayesian_logistic_regression(self, data: pd.DataFrame, target_col: str, 
                                   predictor_cols: List[str], 
                                   n_samples: int = 2000) -> Dict[str, Any]:
        """ベイズロジスティック回帰"""
        
        if not PYMC_AVAILABLE:
            return {'error': 'PyMCライブラリが必要です'}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Bayesian Logistic Regression',
            'n_samples': n_samples
        }
        
        # データ準備
        analysis_cols = [target_col] + predictor_cols
        clean_data = data[analysis_cols].dropna()
        
        X = clean_data[predictor_cols].values
        y = clean_data[target_col].values
        
        # ターゲット変数のエンコーディング
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        try:
            with pm.Model() as logistic_model:
                # 事前分布
                alpha = pm.Normal('intercept', mu=0, sigma=2)
                beta = pm.Normal('coefficients', mu=0, sigma=2, shape=X_scaled.shape[1])
                
                # ロジット関数
                logit_p = alpha + pm.math.dot(X_scaled, beta)
                p = pm.Deterministic('probability', pm.math.sigmoid(logit_p))
                
                # 尤度
                y_obs = pm.Bernoulli('y_obs', p=p, observed=y_encoded)
                
                # サンプリング
                trace = pm.sample(n_samples, tune=1000, return_inferencedata=True,
                                random_seed=self.random_seed, cores=1,
                                progressbar=False)
                
                # 事後予測チェック
                ppc = pm.sample_posterior_predictive(trace, random_seed=self.random_seed)
            
            self.fitted_models['bayesian_logistic'] = logistic_model
            self.traces['bayesian_logistic'] = trace
            
            # 結果の要約
            summary = az.summary(trace, var_names=['intercept', 'coefficients'])
            hdi = az.hdi(trace, hdi_prob=0.95)
            
            # オッズ比の計算
            odds_ratios = {}
            for i, col in enumerate(predictor_cols):
                coef_samples = trace.posterior['coefficients'].values[:, :, i].flatten()
                or_samples = np.exp(coef_samples)
                
                odds_ratios[col] = {
                    'odds_ratio_mean': float(np.mean(or_samples)),
                    'odds_ratio_median': float(np.median(or_samples)),
                    'hdi_2.5%': float(np.exp(hdi['coefficients'].values[i, 0])),
                    'hdi_97.5%': float(np.exp(hdi['coefficients'].values[i, 1])),
                    'probability_beneficial': float(np.mean(coef_samples > 0)) if np.mean(coef_samples) > 0 else float(np.mean(coef_samples < 0))
                }
            
            results['odds_ratios'] = odds_ratios
            
            # 予測精度
            y_pred_prob = np.mean(ppc.posterior_predictive['y_obs'], axis=(0, 1))
            y_pred_class = (y_pred_prob > 0.5).astype(int)
            accuracy = np.mean(y_pred_class == y_encoded)
            
            results['model_performance'] = {
                'accuracy': float(accuracy),
                'log_likelihood': float(trace.log_likelihood.y_obs.sum().mean()),
                'prediction_probabilities': y_pred_prob.tolist()[:10]
            }
            
            # モデル診断
            results['diagnostics'] = {
                'r_hat_max': float(summary['r_hat'].max()),
                'ess_bulk_min': float(summary['ess_bulk'].min()),
                'converged': bool(summary['r_hat'].max() < 1.1)
            }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    @performance_monitor.monitor_function("hierarchical_modeling") if PROFESSIONAL_LOGGING else lambda x: x
    def hierarchical_modeling(self, data: pd.DataFrame, target_col: str, 
                            predictor_cols: List[str], group_col: str,
                            n_samples: int = 2000) -> Dict[str, Any]:
        """階層ベイズモデリング"""
        
        if not PYMC_AVAILABLE:
            return {'error': 'PyMCライブラリが必要です'}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Hierarchical Bayesian Modeling',
            'group_column': group_col
        }
        
        # データ準備
        analysis_cols = [target_col, group_col] + predictor_cols
        clean_data = data[analysis_cols].dropna()
        
        X = clean_data[predictor_cols].values
        y = clean_data[target_col].values
        groups = clean_data[group_col].values
        
        # グループのエンコーディング
        unique_groups = np.unique(groups)
        group_idx = np.array([np.where(unique_groups == g)[0][0] for g in groups])
        n_groups = len(unique_groups)
        
        # 標準化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        try:
            with pm.Model() as hierarchical_model:
                # ハイパー事前分布
                mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=1)
                sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)
                
                mu_beta = pm.Normal('mu_beta', mu=0, sigma=1, shape=X_scaled.shape[1])
                sigma_beta = pm.HalfNormal('sigma_beta', sigma=1, shape=X_scaled.shape[1])
                
                # グループレベルパラメータ
                alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=n_groups)
                beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta, shape=(n_groups, X_scaled.shape[1]))
                
                # 観測レベル
                sigma = pm.HalfNormal('sigma', sigma=1)
                
                # 線形予測子
                mu = alpha[group_idx] + pm.math.sum(beta[group_idx] * X_scaled, axis=1)
                
                # 尤度
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_scaled)
                
                # サンプリング
                trace = pm.sample(n_samples, tune=1000, return_inferencedata=True,
                                random_seed=self.random_seed, cores=1,
                                progressbar=False)
            
            self.fitted_models['hierarchical'] = hierarchical_model
            self.traces['hierarchical'] = trace
            
            # グループ別結果
            group_results = {}
            for i, group in enumerate(unique_groups):
                alpha_samples = trace.posterior['alpha'].values[:, :, i].flatten()
                
                group_results[str(group)] = {
                    'intercept_mean': float(np.mean(alpha_samples)),
                    'intercept_std': float(np.std(alpha_samples)),
                    'intercept_hdi': [
                        float(np.percentile(alpha_samples, 2.5)),
                        float(np.percentile(alpha_samples, 97.5))
                    ]
                }
                
                # 各予測変数の係数
                for j, col in enumerate(predictor_cols):
                    beta_samples = trace.posterior['beta'].values[:, :, i, j].flatten()
                    group_results[str(group)][f'coef_{col}'] = {
                        'mean': float(np.mean(beta_samples)),
                        'std': float(np.std(beta_samples)),
                        'hdi': [
                            float(np.percentile(beta_samples, 2.5)),
                            float(np.percentile(beta_samples, 97.5))
                        ]
                    }
            
            results['group_effects'] = group_results
            
            # 全体効果（ハイパーパラメータ）
            summary = az.summary(trace, var_names=['mu_alpha', 'mu_beta', 'sigma_alpha', 'sigma_beta'])
            
            results['population_effects'] = {
                'intercept_population_mean': float(summary.loc['mu_alpha', 'mean']),
                'coefficient_population_means': summary.loc[summary.index.str.startswith('mu_beta'), 'mean'].tolist()
            }
            
            # 分散成分
            results['variance_components'] = {
                'between_group_intercept': float(summary.loc['sigma_alpha', 'mean']),
                'between_group_slopes': summary.loc[summary.index.str.startswith('sigma_beta'), 'mean'].tolist(),
                'within_group': float(summary.loc['sigma', 'mean'])
            }
            
            # モデル診断
            summary_all = az.summary(trace)
            results['diagnostics'] = {
                'r_hat_max': float(summary_all['r_hat'].max()),
                'ess_bulk_min': float(summary_all['ess_bulk'].min()),
                'converged': bool(summary_all['r_hat'].max() < 1.1)
            }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    @performance_monitor.monitor_function("bayesian_time_series") if PROFESSIONAL_LOGGING else lambda x: x
    def bayesian_time_series(self, data: pd.DataFrame, date_col: str, 
                           value_col: str, forecast_periods: int = 12) -> Dict[str, Any]:
        """ベイズ時系列解析"""
        
        if not PYMC_AVAILABLE:
            return {'error': 'PyMCライブラリが必要です'}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Bayesian Time Series Analysis',
            'forecast_periods': forecast_periods
        }
        
        # データ準備
        ts_data = data[[date_col, value_col]].copy()
        ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        ts_data = ts_data.sort_values(date_col).dropna()
        
        y = ts_data[value_col].values
        n_obs = len(y)
        
        # 標準化
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_scaled = (y - y_mean) / y_std
        
        try:
            with pm.Model() as ts_model:
                # ローカルレベルモデル（ランダムウォーク + ノイズ）
                
                # 初期状態
                initial_level = pm.Normal('initial_level', mu=0, sigma=1)
                
                # プロセスノイズ
                sigma_level = pm.HalfNormal('sigma_level', sigma=0.1)
                sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.1)
                
                # 状態の進化
                level_innovations = pm.Normal('level_innovations', mu=0, sigma=sigma_level, shape=n_obs-1)
                
                # 状態ベクトル
                levels = pt.concatenate([[initial_level], initial_level + pt.cumsum(level_innovations)])
                
                # 観測方程式
                y_obs = pm.Normal('y_obs', mu=levels, sigma=sigma_obs, observed=y_scaled)
                
                # サンプリング
                trace = pm.sample(1000, tune=500, return_inferencedata=True,
                                random_seed=self.random_seed, cores=1,
                                progressbar=False)
                
                # 予測
                with ts_model:
                    # 将来の状態進化
                    future_innovations = pm.Normal('future_innovations', 
                                                 mu=0, sigma=sigma_level, 
                                                 shape=forecast_periods)
                    
                    last_level = levels[-1]
                    future_levels = last_level + pt.cumsum(future_innovations)
                    
                    # 将来の観測
                    y_future = pm.Normal('y_future', mu=future_levels, sigma=sigma_obs,
                                       shape=forecast_periods)
                    
                    # 事後予測サンプリング
                    ppc = pm.sample_posterior_predictive(trace, predictions=True,
                                                       random_seed=self.random_seed)
            
            self.fitted_models['bayesian_ts'] = ts_model
            self.traces['bayesian_ts'] = trace
            
            # フィッティング結果
            fitted_levels = trace.posterior['levels'].mean(dim=['chain', 'draw']).values
            fitted_levels_scaled = fitted_levels * y_std + y_mean
            
            # 予測結果
            forecast_samples = ppc.predictions['y_future']
            forecast_mean = forecast_samples.mean(dim=['chain', 'draw']).values * y_std + y_mean
            forecast_hdi = az.hdi(forecast_samples, hdi_prob=0.95).values * y_std + y_mean
            
            results['fitted_values'] = fitted_levels_scaled.tolist()
            results['forecast'] = {
                'mean': forecast_mean.tolist(),
                'hdi_lower': forecast_hdi[:, 0].tolist(),
                'hdi_upper': forecast_hdi[:, 1].tolist()
            }
            
            # モデル診断
            summary = az.summary(trace)
            results['diagnostics'] = {
                'r_hat_max': float(summary['r_hat'].max()),
                'ess_bulk_min': float(summary['ess_bulk'].min()),
                'converged': bool(summary['r_hat'].max() < 1.1)
            }
            
            # 状態成分の分析
            level_variance = trace.posterior['sigma_level'].mean().values
            obs_variance = trace.posterior['sigma_obs'].mean().values
            
            results['variance_decomposition'] = {
                'level_variance': float(level_variance),
                'observation_variance': float(obs_variance),
                'signal_to_noise_ratio': float(level_variance / obs_variance)
            }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def model_comparison(self, models: List[str]) -> Dict[str, Any]:
        """ベイズモデル比較"""
        
        if not PYMC_AVAILABLE:
            return {'error': 'PyMCライブラリが必要です'}
        
        available_traces = {k: v for k, v in self.traces.items() if k in models}
        
        if len(available_traces) < 2:
            return {'error': '比較するには最低2つのモデルが必要です'}
        
        try:
            # WAIC計算
            waic_comparison = az.compare(available_traces, ic='waic')
            
            # LOO計算
            loo_comparison = az.compare(available_traces, ic='loo')
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'method': 'Bayesian Model Comparison',
                'models_compared': list(available_traces.keys()),
                'waic_comparison': {
                    'ranking': waic_comparison.index.tolist(),
                    'waic_values': waic_comparison['waic'].to_dict(),
                    'waic_se': waic_comparison['waic_se'].to_dict(),
                    'dwaic': waic_comparison['dwaic'].to_dict(),
                    'weight': waic_comparison['weight'].to_dict()
                },
                'loo_comparison': {
                    'ranking': loo_comparison.index.tolist(),
                    'loo_values': loo_comparison['loo'].to_dict(),
                    'loo_se': loo_comparison['loo_se'].to_dict(),
                    'dloo': loo_comparison['dloo'].to_dict(),
                    'weight': loo_comparison['weight'].to_dict()
                }
            }
            
            # 最良モデル
            best_waic = waic_comparison.index[0]
            best_loo = loo_comparison.index[0]
            
            results['best_models'] = {
                'waic_best': best_waic,
                'loo_best': best_loo,
                'agreement': best_waic == best_loo
            }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_trace_plots(self, model_name: str, output_path: str = None) -> str:
        """トレースプロット生成"""
        
        if model_name not in self.traces:
            return f"Model {model_name} not found"
        
        trace = self.traces[model_name]
        
        # トレースプロット
        ax = az.plot_trace(trace, compact=True, figsize=(12, 8))
        plt.suptitle(f'Trace Plot: {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return f"Trace plot saved to {output_path}"
        else:
            return "Trace plot generated successfully"

# グローバルインスタンス
deep_bayesian_analyzer = DeepBayesianAnalyzer() 