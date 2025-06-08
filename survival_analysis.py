#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Survival Analysis Module
完全生存解析モジュール

Author: Ryo Minegishi
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import scipy.stats as stats
from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp

# 生存解析専門ライブラリ
try:
    from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter, AalenJohanssonFitter
    from lifelines import LogNormalFitter, WeibullFitter, ExponentialFitter, LogLogisticFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test, pairwise_logrank_test
    from lifelines.plotting import plot_lifetimes
    from lifelines.utils import concordance_index, median_survival_times
    from lifelines.calibration import survival_probability_calibration
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

# プロフェッショナル機能インポート
try:
    from professional_utils import professional_logger, performance_monitor
    PROFESSIONAL_LOGGING = True
except ImportError:
    PROFESSIONAL_LOGGING = False

warnings.filterwarnings('ignore')

class CompleteSurvivalAnalyzer:
    """完全生存解析クラス"""
    
    def __init__(self, confidence_interval: float = 0.95):
        self.confidence_interval = confidence_interval
        self.results_cache = {}
        self.fitted_models = {}
        
        if not LIFELINES_AVAILABLE:
            raise ImportError("Lifelines library is required for survival analysis")
        
        if PROFESSIONAL_LOGGING:
            professional_logger.info("完全生存解析システム初期化", 
                                   confidence_interval=confidence_interval)
    
    @performance_monitor.monitor_function("kaplan_meier_analysis") if PROFESSIONAL_LOGGING else lambda x: x
    def kaplan_meier_analysis(self, data: pd.DataFrame, duration_col: str, 
                            event_col: str, group_col: str = None) -> Dict[str, Any]:
        """Kaplan-Meier生存解析"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Kaplan-Meier',
            'confidence_interval': self.confidence_interval
        }
        
        # データ準備
        survival_data = data[[duration_col, event_col]].copy()
        if group_col and group_col in data.columns:
            survival_data[group_col] = data[group_col]
        
        survival_data = survival_data.dropna()
        
        # 全体のKaplan-Meier推定
        kmf = KaplanMeierFitter(alpha=1-self.confidence_interval)
        kmf.fit(survival_data[duration_col], survival_data[event_col], 
               label='Overall Population')
        
        self.fitted_models['kaplan_meier_overall'] = kmf
        
        # 基本結果
        results['overall'] = {
            'median_survival': kmf.median_survival_time_,
            'median_survival_ci': kmf.confidence_interval_median_survival_time_,
            'survival_function': {
                'timeline': kmf.timeline.tolist(),
                'survival_prob': kmf.survival_function_.iloc[:, 0].tolist(),
                'ci_lower': kmf.confidence_interval_.iloc[:, 0].tolist(),
                'ci_upper': kmf.confidence_interval_.iloc[:, 1].tolist()
            },
            'n_subjects': len(survival_data),
            'n_events': survival_data[event_col].sum(),
            'censoring_rate': 1 - (survival_data[event_col].sum() / len(survival_data))
        }
        
        # 生存確率（特定時点）
        time_points = [0.25, 0.5, 0.75] if kmf.timeline.max() > 1 else [0.1, 0.2, 0.3]
        survival_probs = {}
        for quantile in time_points:
            time_point = kmf.timeline.quantile(quantile)
            try:
                prob = kmf.survival_function_at_times(time_point).iloc[0]
                survival_probs[f'{quantile*100:.0f}%_time'] = {
                    'time': time_point,
                    'survival_probability': prob
                }
            except:
                pass
        
        results['survival_probabilities'] = survival_probs
        
        # グループ別解析
        if group_col and group_col in survival_data.columns:
            groups = survival_data[group_col].unique()
            group_results = {}
            group_models = {}
            
            for group in groups:
                group_data = survival_data[survival_data[group_col] == group]
                if len(group_data) > 5:  # 最小サンプルサイズ
                    kmf_group = KaplanMeierFitter(alpha=1-self.confidence_interval)
                    kmf_group.fit(group_data[duration_col], group_data[event_col],
                                 label=f'Group {group}')
                    
                    group_models[f'group_{group}'] = kmf_group
                    
                    group_results[str(group)] = {
                        'median_survival': kmf_group.median_survival_time_,
                        'median_survival_ci': kmf_group.confidence_interval_median_survival_time_,
                        'n_subjects': len(group_data),
                        'n_events': group_data[event_col].sum(),
                        'censoring_rate': 1 - (group_data[event_col].sum() / len(group_data))
                    }
            
            self.fitted_models.update(group_models)
            results['group_analysis'] = group_results
            
            # Log-rank test
            if len(groups) >= 2:
                logrank_results = self._perform_logrank_tests(survival_data, 
                                                            duration_col, event_col, group_col)
                results['statistical_tests'] = logrank_results
        
        # リスクテーブル生成
        risk_table = self._generate_risk_table(kmf, survival_data[duration_col])
        results['risk_table'] = risk_table
        
        return results
    
    @performance_monitor.monitor_function("cox_regression_analysis") if PROFESSIONAL_LOGGING else lambda x: x
    def cox_regression_analysis(self, data: pd.DataFrame, duration_col: str, 
                              event_col: str, covariate_cols: List[str]) -> Dict[str, Any]:
        """Cox比例ハザード回帰解析"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Cox Proportional Hazards',
            'covariates': covariate_cols
        }
        
        # データ準備
        analysis_cols = [duration_col, event_col] + covariate_cols
        cox_data = data[analysis_cols].dropna()
        
        if len(cox_data) < 10:
            raise ValueError("Cox回帰には最低10のサンプルが必要です")
        
        # Cox回帰フィッティング
        cph = CoxPHFitter(alpha=1-self.confidence_interval)
        cph.fit(cox_data, duration_col=duration_col, event_col=event_col)
        
        self.fitted_models['cox_regression'] = cph
        
        # 結果の詳細抽出
        summary = cph.summary
        results['model_summary'] = {
            'coefficients': summary['coef'].to_dict(),
            'hazard_ratios': summary['exp(coef)'].to_dict(),
            'standard_errors': summary['se(coef)'].to_dict(),
            'z_scores': summary['z'].to_dict(),
            'p_values': summary['p'].to_dict(),
            'confidence_intervals': {
                'lower': summary[f'exp(coef) lower {self.confidence_interval*100:.0f}%'].to_dict(),
                'upper': summary[f'exp(coef) upper {self.confidence_interval*100:.0f}%'].to_dict()
            }
        }
        
        # モデル適合度
        results['model_fit'] = {
            'concordance_index': cph.concordance_index_,
            'log_likelihood': cph.log_likelihood_,
            'aic': cph.AIC_,
            'bic': cph.BIC_,
            'log_likelihood_ratio_test': {
                'statistic': cph.log_likelihood_ratio_test().test_statistic,
                'p_value': cph.log_likelihood_ratio_test().p_value,
                'degrees_of_freedom': cph.log_likelihood_ratio_test().degrees_of_freedom
            }
        }
        
        # 比例ハザード仮定の検定
        try:
            ph_test = cph.check_assumptions(cox_data, show_plots=False)
            results['proportional_hazards_test'] = {
                'test_statistic': ph_test.test_statistic.to_dict(),
                'p_values': ph_test.p_value.to_dict(),
                'assumption_violated': (ph_test.p_value < 0.05).any()
            }
        except Exception as e:
            results['proportional_hazards_test'] = {'error': str(e)}
        
        # 予測（リスクスコア）
        if len(cox_data) > 0:
            risk_scores = cph.predict_partial_hazard(cox_data)
            results['risk_scores'] = {
                'mean': risk_scores.mean(),
                'std': risk_scores.std(),
                'min': risk_scores.min(),
                'max': risk_scores.max(),
                'quartiles': {
                    'q25': risk_scores.quantile(0.25),
                    'q50': risk_scores.quantile(0.5),
                    'q75': risk_scores.quantile(0.75)
                }
            }
        
        return results
    
    @performance_monitor.monitor_function("parametric_survival_analysis") if PROFESSIONAL_LOGGING else lambda x: x
    def parametric_survival_analysis(self, data: pd.DataFrame, duration_col: str, 
                                   event_col: str) -> Dict[str, Any]:
        """パラメトリック生存解析（複数分布の比較）"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Parametric Survival Analysis'
        }
        
        # データ準備
        survival_data = data[[duration_col, event_col]].dropna()
        
        # 検証する分布
        distributions = {
            'exponential': ExponentialFitter,
            'weibull': WeibullFitter,
            'log_normal': LogNormalFitter,
            'log_logistic': LogLogisticFitter
        }
        
        distribution_results = {}
        model_comparison = {}
        
        for dist_name, dist_class in distributions.items():
            try:
                # フィッティング
                fitter = dist_class(alpha=1-self.confidence_interval)
                fitter.fit(survival_data[duration_col], survival_data[event_col])
                
                self.fitted_models[f'parametric_{dist_name}'] = fitter
                
                # 結果保存
                distribution_results[dist_name] = {
                    'parameters': fitter.params_.to_dict(),
                    'median_survival': fitter.median_survival_time_,
                    'mean_survival': fitter.mean_survival_time_ if hasattr(fitter, 'mean_survival_time_') else None,
                    'log_likelihood': fitter.log_likelihood_,
                    'aic': fitter.AIC_,
                    'bic': fitter.BIC_
                }
                
                # モデル比較用
                model_comparison[dist_name] = {
                    'aic': fitter.AIC_,
                    'bic': fitter.BIC_,
                    'log_likelihood': fitter.log_likelihood_
                }
                
            except Exception as e:
                distribution_results[dist_name] = {'error': str(e)}
        
        results['distributions'] = distribution_results
        
        # 最適モデル選択
        if model_comparison:
            best_aic = min(model_comparison.items(), key=lambda x: x[1]['aic'])
            best_bic = min(model_comparison.items(), key=lambda x: x[1]['bic'])
            
            results['model_selection'] = {
                'best_aic': {
                    'distribution': best_aic[0],
                    'aic': best_aic[1]['aic']
                },
                'best_bic': {
                    'distribution': best_bic[0],
                    'bic': best_bic[1]['bic']
                },
                'comparison_table': model_comparison
            }
        
        return results
    
    def _perform_logrank_tests(self, data: pd.DataFrame, duration_col: str, 
                              event_col: str, group_col: str) -> Dict[str, Any]:
        """Log-rank検定の実行"""
        
        groups = data[group_col].unique()
        tests = {}
        
        # 全体のlog-rank検定
        if len(groups) >= 2:
            try:
                # データをグループ別に分割
                group_data = {}
                for group in groups:
                    group_subset = data[data[group_col] == group]
                    group_data[group] = {
                        'durations': group_subset[duration_col],
                        'events': group_subset[event_col]
                    }
                
                # 2群比較の場合
                if len(groups) == 2:
                    group1, group2 = groups
                    lr_test = logrank_test(
                        group_data[group1]['durations'], group_data[group2]['durations'],
                        group_data[group1]['events'], group_data[group2]['events']
                    )
                    
                    tests['overall_logrank'] = {
                        'test_statistic': lr_test.test_statistic,
                        'p_value': lr_test.p_value,
                        'degrees_of_freedom': 1,
                        'groups_compared': [str(group1), str(group2)],
                        'significant': lr_test.p_value < 0.05
                    }
                
                # 多群比較の場合
                elif len(groups) > 2:
                    # 全体比較
                    durations_list = [group_data[g]['durations'] for g in groups]
                    events_list = [group_data[g]['events'] for g in groups]
                    
                    ml_test = multivariate_logrank_test(
                        durations_list, groups, events_list
                    )
                    
                    tests['multivariate_logrank'] = {
                        'test_statistic': ml_test.test_statistic,
                        'p_value': ml_test.p_value,
                        'degrees_of_freedom': len(groups) - 1,
                        'significant': ml_test.p_value < 0.05
                    }
                    
                    # ペアワイズ比較
                    pairwise_results = pairwise_logrank_test(
                        data[duration_col], data[group_col], data[event_col]
                    )
                    
                    tests['pairwise_logrank'] = {
                        'p_values': pairwise_results.p_value.to_dict(),
                        'test_statistics': pairwise_results.test_statistic.to_dict()
                    }
                
            except Exception as e:
                tests['error'] = str(e)
        
        return tests
    
    def _generate_risk_table(self, fitted_model, durations: pd.Series) -> Dict[str, Any]:
        """リスクテーブル生成"""
        
        # 時点の設定
        max_time = durations.max()
        time_points = np.linspace(0, max_time, min(10, int(max_time) + 1))
        
        risk_table = {}
        
        try:
            for time_point in time_points:
                # その時点でのリスク集合サイズ
                at_risk = (durations >= time_point).sum()
                
                # 生存確率
                survival_prob = fitted_model.survival_function_at_times(time_point).iloc[0]
                
                risk_table[f't_{time_point:.1f}'] = {
                    'time': time_point,
                    'at_risk': int(at_risk),
                    'survival_probability': survival_prob
                }
        
        except Exception as e:
            risk_table['error'] = str(e)
        
        return risk_table
    
    @performance_monitor.monitor_function("advanced_survival_features") if PROFESSIONAL_LOGGING else lambda x: x
    def advanced_survival_features(self, data: pd.DataFrame, duration_col: str, 
                                 event_col: str, feature_cols: List[str] = None) -> Dict[str, Any]:
        """高度生存解析機能"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Advanced Survival Features'
        }
        
        survival_data = data[[duration_col, event_col]].copy()
        if feature_cols:
            for col in feature_cols:
                if col in data.columns:
                    survival_data[col] = data[col]
        
        survival_data = survival_data.dropna()
        
        # 1. Nelson-Aalen累積ハザード推定
        try:
            naf = NelsonAalenFitter(alpha=1-self.confidence_interval)
            naf.fit(survival_data[duration_col], survival_data[event_col])
            
            self.fitted_models['nelson_aalen'] = naf
            
            results['nelson_aalen'] = {
                'cumulative_hazard': {
                    'timeline': naf.timeline.tolist(),
                    'cumulative_hazard': naf.cumulative_hazard_.iloc[:, 0].tolist(),
                    'ci_lower': naf.confidence_interval_.iloc[:, 0].tolist(),
                    'ci_upper': naf.confidence_interval_.iloc[:, 1].tolist()
                }
            }
        except Exception as e:
            results['nelson_aalen'] = {'error': str(e)}
        
        # 2. 条件付き生存確率
        try:
            kmf = KaplanMeierFitter()
            kmf.fit(survival_data[duration_col], survival_data[event_col])
            
            # t=0.25, 0.5, 0.75での条件付き生存確率
            timeline = kmf.timeline
            conditional_probs = {}
            
            for conditioning_time in [0.25, 0.5, 0.75]:
                if conditioning_time < timeline.max():
                    conditioning_idx = timeline.searchsorted(conditioning_time)
                    if conditioning_idx < len(timeline):
                        base_survival = kmf.survival_function_.iloc[conditioning_idx, 0]
                        
                        future_times = timeline[timeline > conditioning_time][:5]
                        for future_time in future_times:
                            future_idx = timeline.searchsorted(future_time)
                            if future_idx < len(timeline):
                                future_survival = kmf.survival_function_.iloc[future_idx, 0]
                                conditional_prob = future_survival / base_survival if base_survival > 0 else 0
                                
                                conditional_probs[f'P(T>{future_time:.2f}|T>{conditioning_time:.2f})'] = conditional_prob
            
            results['conditional_survival'] = conditional_probs
        
        except Exception as e:
            results['conditional_survival'] = {'error': str(e)}
        
        # 3. 制限付き平均生存時間 (RMST)
        try:
            # 観察期間の75%時点でのRMST
            tau = survival_data[duration_col].quantile(0.75)
            
            kmf = KaplanMeierFitter()
            kmf.fit(survival_data[duration_col], survival_data[event_col])
            
            # RMST計算（簡易版）
            timeline = kmf.timeline[kmf.timeline <= tau]
            survival_func = kmf.survival_function_.loc[timeline].iloc[:, 0]
            
            # 台形則による積分近似
            rmst = np.trapz(survival_func, timeline)
            
            results['restricted_mean_survival_time'] = {
                'rmst': rmst,
                'restriction_time': tau,
                'interpretation': f'Average survival time up to {tau:.2f} time units'
            }
        
        except Exception as e:
            results['restricted_mean_survival_time'] = {'error': str(e)}
        
        return results
    
    def generate_survival_plots(self, plot_type: str = 'kaplan_meier', 
                               output_path: str = None) -> str:
        """生存曲線プロット生成"""
        
        if plot_type not in self.fitted_models:
            raise ValueError(f"Model {plot_type} not found. Available: {list(self.fitted_models.keys())}")
        
        plt.figure(figsize=(12, 8))
        
        model = self.fitted_models[plot_type]
        
        if 'kaplan_meier' in plot_type:
            model.plot_survival_function()
            plt.title('Kaplan-Meier Survival Curve', fontsize=16, fontweight='bold')
            plt.xlabel('Time', fontsize=14)
            plt.ylabel('Survival Probability', fontsize=14)
            
        elif 'nelson_aalen' in plot_type:
            model.plot_cumulative_hazard()
            plt.title('Nelson-Aalen Cumulative Hazard', fontsize=16, fontweight='bold')
            plt.xlabel('Time', fontsize=14)
            plt.ylabel('Cumulative Hazard', fontsize=14)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return output_path
        else:
            return "Plot generated successfully"
    
    def export_results(self, results: Dict[str, Any], output_path: str, 
                      format: str = 'json') -> str:
        """結果エクスポート"""
        
        import json
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        elif format == 'csv':
            # 主要結果をCSV形式で出力
            summary_data = []
            
            if 'overall' in results:
                summary_data.append({
                    'Metric': 'Median Survival',
                    'Value': results['overall']['median_survival'],
                    'Method': results.get('method', 'Unknown')
                })
                summary_data.append({
                    'Metric': 'Number of Subjects',
                    'Value': results['overall']['n_subjects'],
                    'Method': results.get('method', 'Unknown')
                })
                summary_data.append({
                    'Metric': 'Number of Events',
                    'Value': results['overall']['n_events'],
                    'Method': results.get('method', 'Unknown')
                })
            
            df = pd.DataFrame(summary_data)
            df.to_csv(output_path, index=False)
        
        return f"Results exported to {output_path}"

# グローバルインスタンス
complete_survival_analyzer = CompleteSurvivalAnalyzer() 