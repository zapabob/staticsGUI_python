#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Statistical Analysis Module
高度統計解析モジュール

Author: Ryo Minegishi
License: MIT
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import special
import scipy.optimize as optimize
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
import pingouin as pg
from lifelines import KaplanMeierFitter, CoxPHFitter, LogNormalFitter
import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp

# プロフェッショナル機能インポート
try:
    from professional_utils import professional_logger, performance_monitor
    PROFESSIONAL_LOGGING = True
except ImportError:
    PROFESSIONAL_LOGGING = False

warnings.filterwarnings('ignore')

class AdvancedStatisticalAnalyzer:
    """高度統計解析クラス"""
    
    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.results_cache = {}
        
        if PROFESSIONAL_LOGGING:
            professional_logger.info("高度統計解析システム初期化", cpu_cores=self.n_jobs)
    
    @performance_monitor.monitor_function("multivariate_analysis") if PROFESSIONAL_LOGGING else lambda x: x
    def multivariate_analysis(self, data: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """多変量解析（主成分分析、因子分析、クラスタリング）"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("数値データが見つかりません")
        
        # 欠損値処理
        numeric_data = numeric_data.dropna()
        
        # 標準化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        results = {}
        
        # 1. 主成分分析 (PCA)
        pca = PCA()
        pca_scores = pca.fit_transform(scaled_data)
        
        # 寄与率計算
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # 固有値が1以上の成分数
        n_components_kaiser = np.sum(pca.explained_variance_ > 1)
        
        # 寄与率80%の成分数
        n_components_80 = np.where(cumulative_variance >= 0.8)[0][0] + 1
        
        results['pca'] = {
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'n_components_kaiser': n_components_kaiser,
            'n_components_80': n_components_80,
            'eigenvalues': pca.explained_variance_.tolist(),
            'components': pca.components_[:5].tolist(),  # 上位5成分
            'feature_names': numeric_data.columns.tolist()
        }
        
        # 2. 因子分析
        try:
            fa = FactorAnalysis(n_components=min(5, numeric_data.shape[1]))
            fa_scores = fa.fit_transform(scaled_data)
            
            results['factor_analysis'] = {
                'components': fa.components_.tolist(),
                'noise_variance': fa.noise_variance_.tolist(),
                'log_likelihood': fa.score(scaled_data)
            }
        except Exception as e:
            results['factor_analysis'] = {'error': str(e)}
        
        # 3. クラスタリング分析
        clustering_results = self._perform_clustering(scaled_data, numeric_data.columns)
        results['clustering'] = clustering_results
        
        # 4. 相関分析
        correlation_matrix = numeric_data.corr()
        
        # 高相関ペア検出 (|r| > 0.7)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        results['correlation'] = {
            'matrix': correlation_matrix.to_dict(),
            'high_correlations': high_corr_pairs
        }
        
        return results
    
    def _perform_clustering(self, data: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """クラスタリング解析実行"""
        
        results = {}
        
        # K-means クラスタリング
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(11, len(data)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            inertias.append(kmeans.inertia_)
            
            # シルエット分析
            from sklearn.metrics import silhouette_score
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # 最適クラスタ数（シルエット法）
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # 最適K-meansの実行
        best_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        best_labels = best_kmeans.fit_predict(data)
        
        results['kmeans'] = {
            'optimal_k': optimal_k,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'cluster_centers': best_kmeans.cluster_centers_.tolist(),
            'labels': best_labels.tolist()
        }
        
        # DBSCAN
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(data)
            n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            
            results['dbscan'] = {
                'n_clusters': n_clusters_dbscan,
                'labels': dbscan_labels.tolist(),
                'core_samples': dbscan.core_sample_indices_.tolist()
            }
        except Exception as e:
            results['dbscan'] = {'error': str(e)}
        
        return results
    
    @performance_monitor.monitor_function("time_series_analysis") if PROFESSIONAL_LOGGING else lambda x: x
    def time_series_analysis(self, data: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """時系列解析"""
        
        # データ準備
        ts_data = data[[date_col, value_col]].copy()
        ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        ts_data = ts_data.set_index(date_col).dropna()
        ts_data = ts_data.sort_index()
        
        series = ts_data[value_col]
        
        results = {}
        
        # 1. 基本統計
        results['basic_stats'] = {
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'observations': len(series)
        }
        
        # 2. 定常性検定
        # ADF検定
        adf_result = adfuller(series.dropna())
        
        # KPSS検定
        kpss_result = kpss(series.dropna())
        
        results['stationarity_tests'] = {
            'adf': {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            },
            'kpss': {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05
            }
        }
        
        # 3. 季節性・トレンド分解
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            decomposition = seasonal_decompose(series, model='additive', period=12)
            
            results['decomposition'] = {
                'trend': decomposition.trend.dropna().tolist(),
                'seasonal': decomposition.seasonal.dropna().tolist(),
                'residual': decomposition.resid.dropna().tolist()
            }
        except Exception as e:
            results['decomposition'] = {'error': str(e)}
        
        # 4. ARIMA モデリング
        try:
            # Grid search for best ARIMA parameters
            best_aic = float('inf')
            best_params = None
            
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                        except:
                            continue
            
            if best_params:
                # 最適モデルで再フィッティング
                best_model = ARIMA(series, order=best_params)
                fitted_best = best_model.fit()
                
                # 予測
                forecast = fitted_best.forecast(steps=12)
                forecast_ci = fitted_best.get_forecast(steps=12).conf_int()
                
                results['arima'] = {
                    'best_params': best_params,
                    'aic': fitted_best.aic,
                    'bic': fitted_best.bic,
                    'forecast': forecast.tolist(),
                    'forecast_ci_lower': forecast_ci.iloc[:, 0].tolist(),
                    'forecast_ci_upper': forecast_ci.iloc[:, 1].tolist(),
                    'residuals': fitted_best.resid.tolist()
                }
        except Exception as e:
            results['arima'] = {'error': str(e)}
        
        return results
    
    @performance_monitor.monitor_function("survival_analysis") if PROFESSIONAL_LOGGING else lambda x: x
    def survival_analysis(self, data: pd.DataFrame, duration_col: str, 
                         event_col: str, group_col: str = None) -> Dict[str, Any]:
        """生存解析"""
        
        results = {}
        
        # データ準備
        survival_data = data[[duration_col, event_col]].copy()
        if group_col:
            survival_data[group_col] = data[group_col]
        
        survival_data = survival_data.dropna()
        
        # 1. Kaplan-Meier 推定
        kmf = KaplanMeierFitter()
        kmf.fit(survival_data[duration_col], survival_data[event_col])
        
        results['kaplan_meier'] = {
            'median_survival': kmf.median_survival_time_,
            'survival_function': {
                'timeline': kmf.timeline.tolist(),
                'survival_prob': kmf.survival_function_.iloc[:, 0].tolist()
            },
            'confidence_interval': {
                'lower': kmf.confidence_interval_.iloc[:, 0].tolist(),
                'upper': kmf.confidence_interval_.iloc[:, 1].tolist()
            }
        }
        
        # 2. グループ別分析
        if group_col and group_col in survival_data.columns:
            groups = survival_data[group_col].unique()
            group_results = {}
            
            for group in groups:
                group_data = survival_data[survival_data[group_col] == group]
                kmf_group = KaplanMeierFitter()
                kmf_group.fit(group_data[duration_col], group_data[event_col])
                
                group_results[str(group)] = {
                    'median_survival': kmf_group.median_survival_time_,
                    'n_subjects': len(group_data),
                    'n_events': group_data[event_col].sum()
                }
            
            results['group_analysis'] = group_results
            
            # Log-rank test
            try:
                from lifelines.statistics import logrank_test
                groups_list = list(groups)
                if len(groups_list) == 2:
                    group1_data = survival_data[survival_data[group_col] == groups_list[0]]
                    group2_data = survival_data[survival_data[group_col] == groups_list[1]]
                    
                    logrank_result = logrank_test(
                        group1_data[duration_col], group2_data[duration_col],
                        group1_data[event_col], group2_data[event_col]
                    )
                    
                    results['logrank_test'] = {
                        'test_statistic': logrank_result.test_statistic,
                        'p_value': logrank_result.p_value,
                        'significant': logrank_result.p_value < 0.05
                    }
            except Exception as e:
                results['logrank_test'] = {'error': str(e)}
        
        # 3. Cox回帰
        try:
            # 数値変数のみ選択
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            cox_data = data[list(numeric_cols) + [duration_col, event_col]].dropna()
            
            if len(cox_data.columns) > 2:
                cph = CoxPHFitter()
                cph.fit(cox_data, duration_col=duration_col, event_col=event_col)
                
                results['cox_regression'] = {
                    'summary': cph.summary.to_dict(),
                    'concordance': cph.concordance_index_,
                    'log_likelihood': cph.log_likelihood_,
                    'aic': cph.AIC_
                }
        except Exception as e:
            results['cox_regression'] = {'error': str(e)}
        
        return results
    
    @performance_monitor.monitor_function("bayesian_analysis") if PROFESSIONAL_LOGGING else lambda x: x
    def bayesian_analysis(self, data: pd.DataFrame, target_col: str, 
                         predictor_cols: List[str]) -> Dict[str, Any]:
        """ベイズ統計解析"""
        
        try:
            import pymc as pm
            import arviz as az
        except ImportError:
            return {'error': 'PyMC または ArviZ がインストールされていません'}
        
        # データ準備
        analysis_data = data[predictor_cols + [target_col]].dropna()
        X = analysis_data[predictor_cols].values
        y = analysis_data[target_col].values
        
        results = {}
        
        try:
            with pm.Model() as model:
                # 事前分布
                alpha = pm.Normal('alpha', mu=0, sigma=10)
                beta = pm.Normal('beta', mu=0, sigma=10, shape=X.shape[1])
                sigma = pm.HalfNormal('sigma', sigma=1)
                
                # 線形予測子
                mu = alpha + pm.math.dot(X, beta)
                
                # 尤度
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
                
                # MCMC サンプリング
                trace = pm.sample(2000, tune=1000, return_inferencedata=True, 
                                random_seed=42, cores=1)
                
                # 結果の要約
                summary = az.summary(trace)
                
                results['bayesian_regression'] = {
                    'summary': summary.to_dict(),
                    'r_hat_max': summary['r_hat'].max(),
                    'ess_bulk_min': summary['ess_bulk'].min(),
                    'converged': summary['r_hat'].max() < 1.1
                }
                
                # 事後予測チェック
                with model:
                    ppc = pm.sample_posterior_predictive(trace, random_seed=42)
                
                results['posterior_predictive'] = {
                    'mean_prediction': np.mean(ppc.posterior_predictive['y_obs'], axis=(0, 1)).tolist()
                }
        
        except Exception as e:
            results['bayesian_regression'] = {'error': str(e)}
        
        return results
    
    def comprehensive_eda(self, data: pd.DataFrame) -> Dict[str, Any]:
        """包括的探索的データ解析"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.shape
        }
        
        # 1. データ概要
        results['data_overview'] = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(data.select_dtypes(include=['datetime64']).columns)
        }
        
        # 2. 欠損値分析
        missing_analysis = self._analyze_missing_values(data)
        results['missing_values'] = missing_analysis
        
        # 3. 外れ値検出
        outlier_analysis = self._detect_outliers(data)
        results['outliers'] = outlier_analysis
        
        # 4. 分布分析
        distribution_analysis = self._analyze_distributions(data)
        results['distributions'] = distribution_analysis
        
        # 5. 関係性分析
        relationship_analysis = self._analyze_relationships(data)
        results['relationships'] = relationship_analysis
        
        return results
    
    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """欠損値分析"""
        
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        missing_patterns = data.isnull().groupby(data.isnull().columns.tolist()).size()
        
        return {
            'missing_counts': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'total_missing': missing_counts.sum(),
            'columns_with_missing': len(missing_counts[missing_counts > 0]),
            'complete_rows': len(data.dropna()),
            'missing_patterns': len(missing_patterns)
        }
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """外れ値検出"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        outlier_results = {}
        
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            
            if len(series) < 4:
                continue
            
            # IQR法
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            
            # Z-score法
            z_scores = np.abs(stats.zscore(series))
            zscore_outliers = (z_scores > 3).sum()
            
            outlier_results[col] = {
                'iqr_outliers': int(iqr_outliers),
                'zscore_outliers': int(zscore_outliers),
                'outlier_percentage_iqr': (iqr_outliers / len(series)) * 100,
                'outlier_percentage_zscore': (zscore_outliers / len(series)) * 100
            }
        
        return outlier_results
    
    def _analyze_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分布分析"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        distribution_results = {}
        
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            
            if len(series) < 3:
                continue
            
            # 基本統計
            basic_stats = {
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis(),
                'min': series.min(),
                'max': series.max()
            }
            
            # 正規性検定
            try:
                if len(series) <= 5000:  # Shapiro-Wilkは5000サンプルまで
                    stat, p_value = stats.shapiro(series)
                    normality_test = 'shapiro'
                else:
                    stat, p_value = stats.normaltest(series)
                    normality_test = 'dagostino'
                
                basic_stats['normality_test'] = normality_test
                basic_stats['normality_stat'] = stat
                basic_stats['normality_p'] = p_value
                basic_stats['is_normal'] = p_value > 0.05
            except:
                basic_stats['normality_test'] = 'failed'
            
            distribution_results[col] = basic_stats
        
        return distribution_results
    
    def _analyze_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """関係性分析"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return {'error': '数値変数が2つ未満のため関係性分析をスキップ'}
        
        # 相関行列
        correlation_matrix = numeric_data.corr()
        
        # 強い相関の検出
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_val,
                        'strength': 'very_strong' if abs(corr_val) > 0.9 else 'strong'
                    })
        
        # 多重共線性診断（VIF）
        vif_results = {}
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            for i, col in enumerate(numeric_data.columns):
                vif = variance_inflation_factor(numeric_data.values, i)
                vif_results[col] = vif
        except:
            vif_results = {'error': 'VIF計算に失敗'}
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'vif': vif_results,
            'max_correlation': correlation_matrix.abs().values[np.triu_indices_from(correlation_matrix.values, k=1)].max()
        }

# グローバルインスタンス
advanced_analyzer = AdvancedStatisticalAnalyzer()