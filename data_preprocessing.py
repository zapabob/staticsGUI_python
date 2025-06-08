#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Preprocessing and Feature Engineering Module
データ前処理・特徴量エンジニアリングモジュール

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import zscore, boxcox, yeojohnson
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """データ前処理クラス"""
    
    def __init__(self):
        """初期化"""
        self.scalers = {}
        self.imputers = {}
        self.outlier_indices = {}
        self.transformation_params = {}
        
    def detect_outliers(self, data: pd.DataFrame, methods: List[str] = ['iqr', 'zscore', 'isolation'], 
                       threshold_zscore: float = 3.0, threshold_iqr: float = 1.5) -> Dict[str, Any]:
        """外れ値検出"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return {"success": False, "error": "数値データが見つかりません"}
            
            outlier_results = {}
            
            for column in numeric_data.columns:
                col_data = numeric_data[column].dropna()
                
                if len(col_data) == 0:
                    continue
                
                outliers = {
                    'column': column,
                    'total_count': len(col_data),
                    'outlier_indices': {},
                    'outlier_counts': {},
                    'outlier_percentages': {}
                }
                
                # 1. IQR法
                if 'iqr' in methods:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold_iqr * IQR
                    upper_bound = Q3 + threshold_iqr * IQR
                    
                    iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    outliers['outlier_indices']['iqr'] = iqr_outliers.index.tolist()
                    outliers['outlier_counts']['iqr'] = len(iqr_outliers)
                    outliers['outlier_percentages']['iqr'] = len(iqr_outliers) / len(col_data) * 100
                    outliers['iqr_bounds'] = {'lower': lower_bound, 'upper': upper_bound}
                
                # 2. Z-score法
                if 'zscore' in methods:
                    z_scores = np.abs(zscore(col_data))
                    zscore_outliers = col_data[z_scores > threshold_zscore]
                    outliers['outlier_indices']['zscore'] = zscore_outliers.index.tolist()
                    outliers['outlier_counts']['zscore'] = len(zscore_outliers)
                    outliers['outlier_percentages']['zscore'] = len(zscore_outliers) / len(col_data) * 100
                
                # 3. Isolation Forest法
                if 'isolation' in methods:
                    try:
                        from sklearn.ensemble import IsolationForest
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        outlier_labels = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                        isolation_outliers = col_data[outlier_labels == -1]
                        outliers['outlier_indices']['isolation'] = isolation_outliers.index.tolist()
                        outliers['outlier_counts']['isolation'] = len(isolation_outliers)
                        outliers['outlier_percentages']['isolation'] = len(isolation_outliers) / len(col_data) * 100
                    except ImportError:
                        outliers['outlier_indices']['isolation'] = []
                        outliers['outlier_counts']['isolation'] = 0
                        outliers['outlier_percentages']['isolation'] = 0
                
                outlier_results[column] = outliers
            
            # 統合外れ値リスト（複数手法で検出されたもの）
            consensus_outliers = self._find_consensus_outliers(outlier_results, methods)
            
            return {
                "success": True,
                "detailed_results": outlier_results,
                "consensus_outliers": consensus_outliers,
                "methods_used": methods,
                "parameters": {
                    "threshold_zscore": threshold_zscore,
                    "threshold_iqr": threshold_iqr
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def remove_outliers(self, data: pd.DataFrame, outlier_detection_result: Dict[str, Any], 
                       method: str = 'consensus', min_agreement: int = 2) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """外れ値除去"""
        try:
            if not outlier_detection_result.get("success", False):
                return data, {"success": False, "error": "有効な外れ値検出結果が必要です"}
            
            indices_to_remove = set()
            
            if method == 'consensus':
                # 複数手法での合意による除去
                consensus_outliers = outlier_detection_result.get("consensus_outliers", {})
                for column, outlier_info in consensus_outliers.items():
                    if outlier_info['agreement_count'] >= min_agreement:
                        indices_to_remove.update(outlier_info['indices'])
            
            elif method in ['iqr', 'zscore', 'isolation']:
                # 単一手法による除去
                detailed_results = outlier_detection_result.get("detailed_results", {})
                for column, outlier_info in detailed_results.items():
                    if method in outlier_info['outlier_indices']:
                        indices_to_remove.update(outlier_info['outlier_indices'][method])
            
            # データから外れ値を除去
            clean_data = data.drop(index=list(indices_to_remove))
            
            removal_summary = {
                "success": True,
                "original_shape": data.shape,
                "cleaned_shape": clean_data.shape,
                "removed_count": len(indices_to_remove),
                "removal_percentage": len(indices_to_remove) / len(data) * 100,
                "removed_indices": list(indices_to_remove),
                "method_used": method
            }
            
            return clean_data, removal_summary
            
        except Exception as e:
            return data, {"success": False, "error": str(e)}
    
    def handle_missing_values(self, data: pd.DataFrame, method: str = 'auto', 
                            strategy: str = 'mean', n_neighbors: int = 5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """欠損値処理"""
        try:
            missing_info = self._analyze_missing_patterns(data)
            
            if missing_info['total_missing'] == 0:
                return data, {"success": True, "message": "欠損値は見つかりませんでした"}
            
            imputed_data = data.copy()
            imputation_details = {}
            
            # 数値列と非数値列を分離
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 数値列の処理
            if numeric_cols:
                numeric_data = data[numeric_cols]
                
                if method == 'auto':
                    # 欠損率が高い列（>30%）はKNN、低い列は平均値/中央値
                    high_missing_cols = [col for col in numeric_cols 
                                       if data[col].isnull().sum() / len(data) > 0.3]
                    low_missing_cols = [col for col in numeric_cols if col not in high_missing_cols]
                    
                    # 低欠損率列：Simple Imputation
                    if low_missing_cols:
                        simple_imputer = SimpleImputer(strategy=strategy)
                        imputed_data[low_missing_cols] = simple_imputer.fit_transform(data[low_missing_cols])
                        self.imputers[f'simple_{strategy}'] = simple_imputer
                        imputation_details['low_missing'] = {
                            'method': f'simple_{strategy}',
                            'columns': low_missing_cols
                        }
                    
                    # 高欠損率列：KNN Imputation
                    if high_missing_cols:
                        knn_imputer = KNNImputer(n_neighbors=n_neighbors)
                        imputed_data[high_missing_cols] = knn_imputer.fit_transform(data[high_missing_cols])
                        self.imputers['knn'] = knn_imputer
                        imputation_details['high_missing'] = {
                            'method': 'knn',
                            'columns': high_missing_cols,
                            'n_neighbors': n_neighbors
                        }
                
                elif method == 'simple':
                    # Simple Imputation
                    simple_imputer = SimpleImputer(strategy=strategy)
                    imputed_data[numeric_cols] = simple_imputer.fit_transform(numeric_data)
                    self.imputers[f'simple_{strategy}'] = simple_imputer
                    imputation_details['numeric'] = {
                        'method': f'simple_{strategy}',
                        'columns': numeric_cols
                    }
                
                elif method == 'knn':
                    # KNN Imputation
                    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
                    imputed_data[numeric_cols] = knn_imputer.fit_transform(numeric_data)
                    self.imputers['knn'] = knn_imputer
                    imputation_details['numeric'] = {
                        'method': 'knn',
                        'columns': numeric_cols,
                        'n_neighbors': n_neighbors
                    }
            
            # カテゴリ列の処理（最頻値で補完）
            if categorical_cols:
                for col in categorical_cols:
                    if data[col].isnull().any():
                        mode_value = data[col].mode().iloc[0] if not data[col].mode().empty else 'Unknown'
                        imputed_data[col].fillna(mode_value, inplace=True)
                        imputation_details[f'categorical_{col}'] = {
                            'method': 'mode',
                            'value': mode_value
                        }
            
            # 結果サマリー
            final_missing_info = self._analyze_missing_patterns(imputed_data)
            
            return imputed_data, {
                "success": True,
                "original_missing": missing_info,
                "final_missing": final_missing_info,
                "imputation_details": imputation_details,
                "method_used": method
            }
            
        except Exception as e:
            return data, {"success": False, "error": str(e)}
    
    def transform_data(self, data: pd.DataFrame, transformations: List[str] = ['standard'], 
                      target_distribution: str = 'normal') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """データ変換（標準化、正規化、分布変換）"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return data, {"success": False, "error": "数値データが見つかりません"}
            
            transformed_data = data.copy()
            transformation_details = {}
            
            for transformation in transformations:
                if transformation == 'standard':
                    # 標準化（平均0、標準偏差1）
                    scaler = StandardScaler()
                    transformed_data[numeric_data.columns] = scaler.fit_transform(numeric_data)
                    self.scalers['standard'] = scaler
                    transformation_details['standard'] = {
                        'method': 'StandardScaler',
                        'columns': numeric_data.columns.tolist()
                    }
                
                elif transformation == 'minmax':
                    # Min-Max正規化（0-1スケール）
                    scaler = MinMaxScaler()
                    transformed_data[numeric_data.columns] = scaler.fit_transform(numeric_data)
                    self.scalers['minmax'] = scaler
                    transformation_details['minmax'] = {
                        'method': 'MinMaxScaler',
                        'columns': numeric_data.columns.tolist()
                    }
                
                elif transformation == 'robust':
                    # ロバストスケーリング（外れ値に頑健）
                    scaler = RobustScaler()
                    transformed_data[numeric_data.columns] = scaler.fit_transform(numeric_data)
                    self.scalers['robust'] = scaler
                    transformation_details['robust'] = {
                        'method': 'RobustScaler',
                        'columns': numeric_data.columns.tolist()
                    }
                
                elif transformation == 'log':
                    # 対数変換（正の値のみ）
                    log_columns = []
                    for col in numeric_data.columns:
                        if (numeric_data[col] > 0).all():
                            transformed_data[col] = np.log(numeric_data[col])
                            log_columns.append(col)
                    
                    transformation_details['log'] = {
                        'method': 'log_transform',
                        'columns': log_columns
                    }
                
                elif transformation == 'sqrt':
                    # 平方根変換（非負の値のみ）
                    sqrt_columns = []
                    for col in numeric_data.columns:
                        if (numeric_data[col] >= 0).all():
                            transformed_data[col] = np.sqrt(numeric_data[col])
                            sqrt_columns.append(col)
                    
                    transformation_details['sqrt'] = {
                        'method': 'sqrt_transform',
                        'columns': sqrt_columns
                    }
                
                elif transformation == 'boxcox':
                    # Box-Cox変換（正の値のみ）
                    boxcox_results = {}
                    for col in numeric_data.columns:
                        if (numeric_data[col] > 0).all():
                            try:
                                transformed_col, lambda_param = boxcox(numeric_data[col])
                                transformed_data[col] = transformed_col
                                boxcox_results[col] = lambda_param
                            except:
                                continue
                    
                    transformation_details['boxcox'] = {
                        'method': 'boxcox_transform',
                        'lambda_params': boxcox_results
                    }
                
                elif transformation == 'yeo_johnson':
                    # Yeo-Johnson変換（任意の値）
                    transformer = PowerTransformer(method='yeo-johnson')
                    transformed_data[numeric_data.columns] = transformer.fit_transform(numeric_data)
                    self.scalers['yeo_johnson'] = transformer
                    transformation_details['yeo_johnson'] = {
                        'method': 'PowerTransformer_yeo_johnson',
                        'columns': numeric_data.columns.tolist()
                    }
            
            return transformed_data, {
                "success": True,
                "transformations_applied": transformation_details,
                "original_shape": data.shape,
                "transformed_shape": transformed_data.shape
            }
            
        except Exception as e:
            return data, {"success": False, "error": str(e)}
    
    def feature_engineering(self, data: pd.DataFrame, target_col: str = None, 
                          methods: List[str] = ['polynomial', 'interaction', 'binning']) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """特徴量エンジニアリング"""
        try:
            engineered_data = data.copy()
            feature_details = {}
            new_features = []
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_col and target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            # 1. 多項式特徴量
            if 'polynomial' in methods and len(numeric_cols) >= 1:
                from sklearn.preprocessing import PolynomialFeatures
                
                # 最大3つの特徴量で2次の多項式特徴量を生成
                selected_cols = numeric_cols[:3]
                poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
                poly_features = poly.fit_transform(data[selected_cols])
                
                poly_feature_names = poly.get_feature_names_out(selected_cols)
                poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=data.index)
                
                # 元の特徴量以外を追加
                new_poly_cols = [col for col in poly_feature_names if col not in selected_cols]
                for col in new_poly_cols:
                    engineered_data[f'poly_{col}'] = poly_df[col]
                    new_features.append(f'poly_{col}')
                
                feature_details['polynomial'] = {
                    'method': 'PolynomialFeatures',
                    'degree': 2,
                    'base_columns': selected_cols,
                    'new_features': new_poly_cols
                }
            
            # 2. 交互作用項
            if 'interaction' in methods and len(numeric_cols) >= 2:
                interaction_features = []
                for i, col1 in enumerate(numeric_cols[:3]):
                    for col2 in numeric_cols[i+1:4]:
                        interaction_name = f'interact_{col1}_{col2}'
                        engineered_data[interaction_name] = data[col1] * data[col2]
                        interaction_features.append(interaction_name)
                        new_features.append(interaction_name)
                
                feature_details['interaction'] = {
                    'method': 'multiplication_interaction',
                    'new_features': interaction_features
                }
            
            # 3. ビニング（離散化）
            if 'binning' in methods:
                binning_features = []
                for col in numeric_cols[:3]:
                    try:
                        # 等頻度ビニング
                        binned_col = f'binned_{col}'
                        engineered_data[binned_col] = pd.qcut(data[col], q=5, labels=False, duplicates='drop')
                        binning_features.append(binned_col)
                        new_features.append(binned_col)
                    except:
                        continue
                
                feature_details['binning'] = {
                    'method': 'quantile_binning',
                    'bins': 5,
                    'new_features': binning_features
                }
            
            # 4. 統計的特徴量（数値列が複数ある場合）
            if 'statistical' in methods and len(numeric_cols) >= 2:
                stat_features = []
                
                # 行ごとの統計量
                numeric_subset = data[numeric_cols[:5]]  # 最大5列
                
                engineered_data['row_mean'] = numeric_subset.mean(axis=1)
                engineered_data['row_std'] = numeric_subset.std(axis=1)
                engineered_data['row_min'] = numeric_subset.min(axis=1)
                engineered_data['row_max'] = numeric_subset.max(axis=1)
                engineered_data['row_range'] = engineered_data['row_max'] - engineered_data['row_min']
                
                stat_features = ['row_mean', 'row_std', 'row_min', 'row_max', 'row_range']
                new_features.extend(stat_features)
                
                feature_details['statistical'] = {
                    'method': 'row_wise_statistics',
                    'base_columns': numeric_cols[:5],
                    'new_features': stat_features
                }
            
            return engineered_data, {
                "success": True,
                "original_features": len(data.columns),
                "new_features_count": len(new_features),
                "total_features": len(engineered_data.columns),
                "new_feature_names": new_features,
                "feature_engineering_details": feature_details
            }
            
        except Exception as e:
            return data, {"success": False, "error": str(e)}
    
    def feature_selection(self, data: pd.DataFrame, target_col: str, 
                         method: str = 'f_test', k: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """特徴量選択"""
        try:
            if target_col not in data.columns:
                return data, {"success": False, "error": f"目的変数 '{target_col}' が見つかりません"}
            
            # 特徴量とターゲットを分離
            feature_cols = [col for col in data.columns if col != target_col]
            X = data[feature_cols].select_dtypes(include=[np.number])
            y = data[target_col]
            
            # 共通のインデックスでデータを揃える
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx].fillna(0)
            y = y.loc[common_idx]
            
            if X.empty:
                return data, {"success": False, "error": "有効な特徴量が見つかりません"}
            
            # 選択する特徴量数の調整
            k = min(k, len(X.columns))
            
            # 特徴量選択方法
            if method == 'f_test':
                # F検定による選択
                if y.dtype in ['object', 'category'] or y.nunique() < 10:
                    # 分類問題
                    selector = SelectKBest(score_func=f_classif, k=k)
                else:
                    # 回帰問題
                    selector = SelectKBest(score_func=f_regression, k=k)
            
            elif method == 'mutual_info':
                # 相互情報量による選択
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            
            else:
                return data, {"success": False, "error": f"不明な特徴量選択方法: {method}"}
            
            # 特徴量選択実行
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            feature_scores = selector.scores_
            
            # 選択された特徴量でデータセットを再構成
            selected_data = data[selected_features + [target_col]]
            
            # 特徴量スコアをDataFrameに整理
            feature_scores_df = pd.DataFrame({
                'feature': X.columns,
                'score': feature_scores,
                'selected': selector.get_support()
            }).sort_values('score', ascending=False)
            
            return selected_data, {
                "success": True,
                "method": method,
                "original_features": len(feature_cols),
                "selected_features": len(selected_features),
                "selected_feature_names": selected_features,
                "feature_scores": feature_scores_df,
                "target_column": target_col
            }
            
        except Exception as e:
            return data, {"success": False, "error": str(e)}
    
    # ヘルパーメソッド
    def _analyze_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """欠損値パターン分析"""
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        return {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'complete_rows': len(data.dropna()),
            'missing_pattern': data.isnull().sum(axis=1).value_counts().to_dict()
        }
    
    def _find_consensus_outliers(self, outlier_results: Dict[str, Any], methods: List[str]) -> Dict[str, Any]:
        """複数手法での外れ値合意検出"""
        consensus_outliers = {}
        
        for column, results in outlier_results.items():
            # 各手法で検出された外れ値インデックス
            all_outliers = []
            for method in methods:
                if method in results['outlier_indices']:
                    all_outliers.extend(results['outlier_indices'][method])
            
            # インデックスの出現回数をカウント
            from collections import Counter
            outlier_counts = Counter(all_outliers)
            
            # 2回以上検出されたインデックス
            consensus_indices = [idx for idx, count in outlier_counts.items() if count >= 2]
            
            consensus_outliers[column] = {
                'indices': consensus_indices,
                'agreement_count': len(consensus_indices),
                'individual_counts': dict(outlier_counts),
                'methods_agreed': methods
            }
        
        return consensus_outliers
    
    def create_data_report(self, original_data: pd.DataFrame, processed_data: pd.DataFrame) -> str:
        """データ処理レポート生成"""
        try:
            report = "📊 Data Preprocessing Report\n"
            report += "=" * 50 + "\n\n"
            
            # 基本情報比較
            report += "📋 Basic Information:\n"
            report += f"Original shape: {original_data.shape}\n"
            report += f"Processed shape: {processed_data.shape}\n"
            report += f"Rows changed: {processed_data.shape[0] - original_data.shape[0]}\n"
            report += f"Columns changed: {processed_data.shape[1] - original_data.shape[1]}\n\n"
            
            # 欠損値比較
            orig_missing = original_data.isnull().sum().sum()
            proc_missing = processed_data.isnull().sum().sum()
            
            report += "❌ Missing Values:\n"
            report += f"Original missing: {orig_missing}\n"
            report += f"Processed missing: {proc_missing}\n"
            report += f"Missing values handled: {orig_missing - proc_missing}\n\n"
            
            # データタイプ情報
            report += "🔍 Data Types:\n"
            report += f"Original numeric columns: {len(original_data.select_dtypes(include=[np.number]).columns)}\n"
            report += f"Processed numeric columns: {len(processed_data.select_dtypes(include=[np.number]).columns)}\n\n"
            
            # 使用されたスケーラー/変換器
            if self.scalers:
                report += "⚙️ Applied Transformations:\n"
                for name, scaler in self.scalers.items():
                    report += f"• {name}: {type(scaler).__name__}\n"
                report += "\n"
            
            # メモリ使用量
            orig_memory = original_data.memory_usage(deep=True).sum() / 1024 / 1024
            proc_memory = processed_data.memory_usage(deep=True).sum() / 1024 / 1024
            
            report += "💾 Memory Usage:\n"
            report += f"Original: {orig_memory:.2f} MB\n"
            report += f"Processed: {proc_memory:.2f} MB\n"
            report += f"Change: {proc_memory - orig_memory:.2f} MB\n"
            
            return report
            
        except Exception as e:
            return f"レポート生成エラー: {str(e)}"

# インスタンス作成
data_preprocessor = DataPreprocessor() 