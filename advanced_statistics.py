#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Statistical Analysis Module
高度統計解析モジュール

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
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
from scipy.stats import mannwhitneyu, kruskal, friedmanchisquare, wilcoxon
import statsmodels.stats.api as sms
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.multivariate.factor import Factor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# プロフェッショナル機能インポート
try:
    from professional_utils import professional_logger, performance_monitor
    PROFESSIONAL_LOGGING = True
except ImportError:
    PROFESSIONAL_LOGGING = False

warnings.filterwarnings('ignore')

class AdvancedStatsAnalyzer:
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
    
    def factor_analysis(self, data, n_factors=None, rotation='varimax', method='ml'):
        """因子分析"""
        try:
            # 数値データのみ抽出
            numeric_data = data.select_dtypes(include=[np.number]).dropna()
            
            if numeric_data.empty:
                return {"success": False, "error": "数値データが見つかりません"}
            
            # 標準化
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # 因子数の決定（Kaiser基準）
            if n_factors is None:
                # 相関行列の固有値計算
                corr_matrix = np.corrcoef(scaled_data.T)
                eigenvalues = np.linalg.eigvals(corr_matrix)
                n_factors = np.sum(eigenvalues > 1)
                n_factors = max(1, min(n_factors, len(numeric_data.columns) - 1))
            
            # 因子分析実行
            fa = FactorAnalysis(n_components=n_factors, rotation=rotation, random_state=42)
            fa.fit(scaled_data)
            
            # 因子負荷量
            loadings = fa.components_.T
            
            # 因子得点
            factor_scores = fa.transform(scaled_data)
            
            # 共通性計算
            communalities = np.sum(loadings**2, axis=1)
            
            # 因子の寄与率
            eigenvalues_fa = np.sum(loadings**2, axis=0)
            variance_explained = eigenvalues_fa / len(numeric_data.columns)
            cumulative_variance = np.cumsum(variance_explained)
            
            # KMO適合性測定
            kmo_statistic = self._calculate_kmo(corr_matrix)
            
            # Bartlett球面性検定
            bartlett_stat, bartlett_p = self._bartlett_test(corr_matrix, len(numeric_data))
            
            return {
                "success": True,
                "n_factors": n_factors,
                "factor_loadings": pd.DataFrame(
                    loadings, 
                    index=numeric_data.columns,
                    columns=[f"Factor{i+1}" for i in range(n_factors)]
                ),
                "factor_scores": pd.DataFrame(
                    factor_scores,
                    columns=[f"Factor{i+1}" for i in range(n_factors)]
                ),
                "communalities": pd.Series(communalities, index=numeric_data.columns),
                "eigenvalues": eigenvalues_fa,
                "variance_explained": variance_explained,
                "cumulative_variance": cumulative_variance,
                "kmo": kmo_statistic,
                "bartlett_test": {"statistic": bartlett_stat, "p_value": bartlett_p},
                "rotation": rotation,
                "method": method
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def discriminant_analysis(self, data, target_col, method='linear'):
        """判別分析"""
        try:
            if target_col not in data.columns:
                return {"success": False, "error": f"目的変数 '{target_col}' が見つかりません"}
            
            # データ準備
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feature_cols:
                feature_cols.remove(target_col)
            
            X = data[feature_cols].dropna()
            y = data[target_col].dropna()
            
            # 共通のインデックスでデータを揃える
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            if len(X) == 0:
                return {"success": False, "error": "有効なデータがありません"}
            
            # 標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 判別分析実行
            if method == 'linear':
                da = LinearDiscriminantAnalysis()
            else:
                da = QuadraticDiscriminantAnalysis()
            
            da.fit(X_scaled, y)
            
            # 予測
            y_pred = da.predict(X_scaled)
            
            # 判別得点
            if method == 'linear':
                discriminant_scores = da.transform(X_scaled)
            else:
                discriminant_scores = da.decision_function(X_scaled)
            
            # 精度計算
            accuracy = np.mean(y == y_pred)
            
            # 混同行列
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(y, y_pred)
            report = classification_report(y, y_pred, output_dict=True)
            
            # 変数の重要度（線形判別の場合）
            feature_importance = None
            if method == 'linear' and hasattr(da, 'coef_'):
                feature_importance = pd.Series(
                    np.abs(da.coef_[0]), 
                    index=feature_cols
                ).sort_values(ascending=False)
            
            return {
                "success": True,
                "method": method,
                "accuracy": accuracy,
                "predictions": y_pred,
                "discriminant_scores": discriminant_scores,
                "confusion_matrix": cm,
                "classification_report": report,
                "feature_importance": feature_importance,
                "classes": da.classes_,
                "n_components": da.n_components_ if hasattr(da, 'n_components_') else None
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def nonparametric_tests(self, data, group_col=None, value_col=None, test_type='auto'):
        """ノンパラメトリック検定"""
        try:
            results = {}
            
            if group_col and value_col:
                # グループ比較検定
                if group_col not in data.columns or value_col not in data.columns:
                    return {"success": False, "error": "指定された列が見つかりません"}
                
                groups = data.groupby(group_col)[value_col].apply(list)
                group_names = list(groups.index)
                group_data = list(groups.values)
                
                # データのフィルタリング（欠損値除去）
                group_data = [np.array(group)[~np.isnan(group)] for group in group_data if len(group) > 0]
                
                if len(group_data) < 2:
                    return {"success": False, "error": "比較可能なグループが不足しています"}
                
                # 検定選択
                if test_type == 'auto':
                    test_type = 'mann_whitney' if len(group_data) == 2 else 'kruskal'
                
                if test_type == 'mann_whitney' and len(group_data) == 2:
                    # Mann-Whitney U検定
                    statistic, p_value = mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
                    results['mann_whitney'] = {
                        "statistic": statistic,
                        "p_value": p_value,
                        "groups": group_names[:2],
                        "n1": len(group_data[0]),
                        "n2": len(group_data[1])
                    }
                
                elif test_type == 'kruskal':
                    # Kruskal-Wallis検定
                    statistic, p_value = kruskal(*group_data)
                    results['kruskal_wallis'] = {
                        "statistic": statistic,
                        "p_value": p_value,
                        "groups": group_names,
                        "n_groups": len(group_data),
                        "total_n": sum(len(group) for group in group_data)
                    }
                    
                    # 事後検定（Dunn's test）
                    if p_value < 0.05:
                        dunn_results = self._dunn_test(data, group_col, value_col)
                        results['dunn_posthoc'] = dunn_results
            
            else:
                # 単一サンプル検定
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                
                for col in numeric_cols[:5]:  # 最初の5列のみ
                    col_data = data[col].dropna()
                    
                    if len(col_data) < 3:
                        continue
                    
                    # Wilcoxon符号順位検定（中央値=0の検定）
                    try:
                        statistic, p_value = wilcoxon(col_data)
                        results[f'{col}_wilcoxon'] = {
                            "statistic": statistic,
                            "p_value": p_value,
                            "n": len(col_data),
                            "median": np.median(col_data)
                        }
                    except:
                        pass
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def effect_size_analysis(self, data, group_col, value_col):
        """効果量分析"""
        try:
            if group_col not in data.columns or value_col not in data.columns:
                return {"success": False, "error": "指定された列が見つかりません"}
            
            # データ準備
            clean_data = data[[group_col, value_col]].dropna()
            groups = clean_data.groupby(group_col)[value_col]
            
            results = {}
            
            # 記述統計
            descriptive = groups.agg(['count', 'mean', 'std', 'median']).round(4)
            results['descriptive'] = descriptive
            
            # 2群比較の場合
            if len(groups) == 2:
                group_names = list(groups.groups.keys())
                group1_data = groups.get_group(group_names[0])
                group2_data = groups.get_group(group_names[1])
                
                # Cohen's d
                cohens_d = self._calculate_cohens_d(group1_data, group2_data)
                
                # Glass's delta
                glass_delta = (group1_data.mean() - group2_data.mean()) / group1_data.std()
                
                # Hedge's g
                pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                     (len(group2_data) - 1) * group2_data.var()) / 
                                    (len(group1_data) + len(group2_data) - 2))
                hedges_g = (group1_data.mean() - group2_data.mean()) / pooled_std
                
                results['effect_sizes'] = {
                    "cohens_d": cohens_d,
                    "glass_delta": glass_delta,
                    "hedges_g": hedges_g,
                    "interpretation_cohens_d": self._interpret_cohens_d(cohens_d)
                }
            
            # 多群比較の場合（η²）
            if len(groups) > 2:
                # One-way ANOVA
                group_data = [group[1] for group in groups]
                f_stat, p_value = stats.f_oneway(*group_data)
                
                # Eta squared (η²)
                ss_between = sum(len(group) * (group.mean() - clean_data[value_col].mean())**2 
                                for group in group_data)
                ss_total = sum((clean_data[value_col] - clean_data[value_col].mean())**2)
                eta_squared = ss_between / ss_total
                
                # Omega squared (ω²)
                ms_within = (ss_total - ss_between) / (len(clean_data) - len(groups))
                omega_squared = (ss_between - (len(groups) - 1) * ms_within) / (ss_total + ms_within)
                
                results['effect_sizes'] = {
                    "eta_squared": eta_squared,
                    "omega_squared": omega_squared,
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "interpretation_eta": self._interpret_eta_squared(eta_squared)
                }
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def multiple_comparison_tests(self, data, group_col, value_col, method='tukey'):
        """多重比較検定"""
        try:
            if group_col not in data.columns or value_col not in data.columns:
                return {"success": False, "error": "指定された列が見つかりません"}
            
            # データ準備
            clean_data = data[[group_col, value_col]].dropna()
            
            if method.lower() == 'tukey':
                # Tukey HSD検定
                tukey_results = pairwise_tukeyhsd(
                    clean_data[value_col], 
                    clean_data[group_col], 
                    alpha=0.05
                )
                
                # 結果をDataFrameに変換
                results_df = pd.DataFrame({
                    'group1': tukey_results.groupsunique[tukey_results._multicomp.pairindices[0]],
                    'group2': tukey_results.groupsunique[tukey_results._multicomp.pairindices[1]],
                    'meandiff': tukey_results.meandiffs,
                    'p_adj': tukey_results.pvalues,
                    'lower': tukey_results.confint[:, 0],
                    'upper': tukey_results.confint[:, 1],
                    'reject': tukey_results.reject
                })
                
                return {
                    "success": True,
                    "method": "Tukey HSD",
                    "results": results_df,
                    "summary": str(tukey_results)
                }
            
            else:
                # Bonferroni補正
                groups = clean_data.groupby(group_col)[value_col]
                group_names = list(groups.groups.keys())
                
                comparisons = []
                for i in range(len(group_names)):
                    for j in range(i+1, len(group_names)):
                        group1_data = groups.get_group(group_names[i])
                        group2_data = groups.get_group(group_names[j])
                        
                        # t検定
                        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                        
                        comparisons.append({
                            'group1': group_names[i],
                            'group2': group_names[j],
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'mean_diff': group1_data.mean() - group2_data.mean()
                        })
                
                # Bonferroni補正
                n_comparisons = len(comparisons)
                for comp in comparisons:
                    comp['p_adj_bonferroni'] = min(comp['p_value'] * n_comparisons, 1.0)
                    comp['significant_bonferroni'] = comp['p_adj_bonferroni'] < 0.05
                
                results_df = pd.DataFrame(comparisons)
                
                return {
                    "success": True,
                    "method": "Bonferroni",
                    "results": results_df,
                    "n_comparisons": n_comparisons
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ヘルパーメソッド
    def _calculate_kmo(self, corr_matrix):
        """KMO適合性測定計算"""
        try:
            inv_corr = np.linalg.inv(corr_matrix)
            partial_corr = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
            np.fill_diagonal(partial_corr, 0)
            
            corr_sum = np.sum(corr_matrix**2) - np.trace(corr_matrix**2)
            partial_sum = np.sum(partial_corr**2)
            
            return corr_sum / (corr_sum + partial_sum)
        except:
            return np.nan
    
    def _bartlett_test(self, corr_matrix, n):
        """Bartlett球面性検定"""
        try:
            p = corr_matrix.shape[0]
            det_corr = np.linalg.det(corr_matrix)
            
            statistic = -(n - 1 - (2*p + 5)/6) * np.log(det_corr)
            df = p * (p - 1) / 2
            p_value = 1 - stats.chi2.cdf(statistic, df)
            
            return statistic, p_value
        except:
            return np.nan, np.nan
    
    def _calculate_cohens_d(self, group1, group2):
        """Cohen's d計算"""
        pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + 
                             (len(group2) - 1) * group2.var()) / 
                            (len(group1) + len(group2) - 2))
        return (group1.mean() - group2.mean()) / pooled_std
    
    def _interpret_cohens_d(self, d):
        """Cohen's d解釈"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta_sq):
        """η²解釈"""
        if eta_sq < 0.01:
            return "negligible"
        elif eta_sq < 0.06:
            return "small"
        elif eta_sq < 0.14:
            return "medium"
        else:
            return "large"
    
    def _dunn_test(self, data, group_col, value_col):
        """Dunn's事後検定（簡易版）"""
        try:
            # pingouin使用
            results = pg.pairwise_tests(
                data=data, 
                dv=value_col, 
                between=group_col, 
                parametric=False,
                padjust='bonf'
            )
            return results.to_dict('records')
        except:
            return "Dunn's test unavailable"

# インスタンス作成
advanced_analyzer = AdvancedStatsAnalyzer()