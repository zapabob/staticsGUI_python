import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

class HADStatisticalAnalysis:
    """HAD統計分析クラス - CUDA対応"""
    
    def __init__(self, data=None):
        self.data = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_history = []
        
    def set_data(self, data):
        """データセット"""
        self.data = data
        
    def log_result(self, analysis_type, result):
        """結果履歴記録"""
        self.results_history.append({
            'timestamp': pd.Timestamp.now(),
            'analysis': analysis_type,
            'result': result
        })
    
    def basic_statistics(self, columns=None):
        """基本統計量計算"""
        if self.data is None or self.data.empty:
            return "データが設定されていません。"
        
        try:
            if columns is None:
                numeric_data = self.data.select_dtypes(include=[np.number])
            else:
                numeric_data = self.data[columns].select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return "数値データが見つかりません。"
            
            results = []
            results.append("=" * 60)
            results.append("基本統計量 (Basic Statistics)")
            results.append("=" * 60)
            
            for col in tqdm(numeric_data.columns, desc="Processing columns"):
                col_data = numeric_data[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                # CUDA対応の高速計算
                if torch.cuda.is_available() and len(col_data) > 1000:
                    tensor_data = torch.tensor(col_data.values, device=self.device, dtype=torch.float32)
                    mean_val = torch.mean(tensor_data).cpu().item()
                    std_val = torch.std(tensor_data).cpu().item()
                    var_val = torch.var(tensor_data).cpu().item()
                    min_val = torch.min(tensor_data).cpu().item()
                    max_val = torch.max(tensor_data).cpu().item()
                    median_val = torch.median(tensor_data).cpu().item()
                else:
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    var_val = col_data.var()
                    min_val = col_data.min()
                    max_val = col_data.max()
                    median_val = col_data.median()
                
                # 追加統計量
                skewness = stats.skew(col_data)
                kurtosis = stats.kurtosis(col_data)
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                
                results.append(f"\n変数: {col}")
                results.append("-" * 40)
                results.append(f"観測数 (N)      : {len(col_data):>10.0f}")
                results.append(f"平均 (Mean)     : {mean_val:>10.4f}")
                results.append(f"中央値 (Median) : {median_val:>10.4f}")
                results.append(f"標準偏差 (SD)   : {std_val:>10.4f}")
                results.append(f"分散 (Variance) : {var_val:>10.4f}")
                results.append(f"最小値 (Min)    : {min_val:>10.4f}")
                results.append(f"最大値 (Max)    : {max_val:>10.4f}")
                results.append(f"第1四分位 (Q1)  : {q1:>10.4f}")
                results.append(f"第3四分位 (Q3)  : {q3:>10.4f}")
                results.append(f"四分位範囲(IQR) : {iqr:>10.4f}")
                results.append(f"歪度 (Skewness) : {skewness:>10.4f}")
                results.append(f"尖度 (Kurtosis) : {kurtosis:>10.4f}")
            
            result_text = "\n".join(results)
            self.log_result("Basic Statistics", result_text)
            return result_text
            
        except Exception as e:
            error_msg = f"基本統計量の計算中にエラーが発生しました: {e}"
            self.log_result("Basic Statistics Error", error_msg)
            return error_msg
    
    def frequency_distribution(self, column, bins=10):
        """度数分布分析"""
        if self.data is None or column not in self.data.columns:
            return "指定された列が見つかりません。"
        
        try:
            col_data = self.data[column].dropna()
            
            results = []
            results.append("=" * 60)
            results.append(f"度数分布 - {column}")
            results.append("=" * 60)
            
            if col_data.dtype in ['object', 'category']:
                # カテゴリカルデータの場合
                freq = col_data.value_counts().sort_index()
                percent = (freq / len(col_data) * 100).round(2)
                
                results.append(f"{'値':<15} {'度数':<10} {'割合(%)':<10}")
                results.append("-" * 40)
                
                for value, count in freq.items():
                    results.append(f"{str(value):<15} {count:<10} {percent[value]:<10}")
                
            else:
                # 数値データの場合
                hist, bin_edges = np.histogram(col_data, bins=bins)
                
                results.append(f"{'区間':<20} {'度数':<10} {'割合(%)':<10}")
                results.append("-" * 45)
                
                for i in range(len(hist)):
                    interval = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
                    freq = hist[i]
                    percent = (freq / len(col_data) * 100)
                    results.append(f"{interval:<20} {freq:<10} {percent:<10.2f}")
            
            results.append(f"\n総観測数: {len(col_data)}")
            
            result_text = "\n".join(results)
            self.log_result("Frequency Distribution", result_text)
            return result_text
            
        except Exception as e:
            error_msg = f"度数分布の計算中にエラーが発生しました: {e}"
            self.log_result("Frequency Distribution Error", error_msg)
            return error_msg
    
    def correlation_analysis(self, method='pearson'):
        """相関分析"""
        if self.data is None or self.data.empty:
            return "データが設定されていません。"
        
        try:
            numeric_data = self.data.select_dtypes(include=[np.number])
            
            if numeric_data.shape[1] < 2:
                return "相関分析には少なくとも2つの数値変数が必要です。"
            
            # 相関係数計算
            if method == 'pearson':
                corr_matrix = numeric_data.corr(method='pearson')
                method_name = "Pearson積率相関係数"
            elif method == 'spearman':
                corr_matrix = numeric_data.corr(method='spearman')
                method_name = "Spearman順位相関係数"
            elif method == 'kendall':
                corr_matrix = numeric_data.corr(method='kendall')
                method_name = "Kendall順位相関係数"
            else:
                corr_matrix = numeric_data.corr(method='pearson')
                method_name = "Pearson積率相関係数"
            
            results = []
            results.append("=" * 60)
            results.append(f"相関分析 - {method_name}")
            results.append("=" * 60)
            
            # 相関行列表示
            results.append("\n相関行列:")
            results.append("-" * 50)
            
            # ヘッダー
            header = "変数".ljust(15)
            for col in corr_matrix.columns:
                header += f"{col[:8]:<10}"
            results.append(header)
            
            # 相関係数
            for idx, row in corr_matrix.iterrows():
                row_str = f"{idx[:12]:<15}"
                for val in row:
                    if pd.isna(val):
                        row_str += "     -    "
                    else:
                        row_str += f"{val:>8.3f}  "
                results.append(row_str)
            
            # 有意性検定
            results.append("\n\n有意性検定 (p値):")
            results.append("-" * 50)
            
            n = len(numeric_data.dropna())
            p_values = []
            
            for i, col1 in enumerate(numeric_data.columns):
                p_row = []
                for j, col2 in enumerate(numeric_data.columns):
                    if i == j:
                        p_row.append(1.0)
                    else:
                        x = numeric_data[col1].dropna()
                        y = numeric_data[col2].dropna()
                        
                        # 共通のインデックスのみ使用
                        common_idx = x.index.intersection(y.index)
                        if len(common_idx) > 2:
                            x_common = x[common_idx]
                            y_common = y[common_idx]
                            
                            if method == 'pearson':
                                _, p_val = stats.pearsonr(x_common, y_common)
                            elif method == 'spearman':
                                _, p_val = stats.spearmanr(x_common, y_common)
                            elif method == 'kendall':
                                _, p_val = stats.kendalltau(x_common, y_common)
                            else:
                                _, p_val = stats.pearsonr(x_common, y_common)
                            
                            p_row.append(p_val)
                        else:
                            p_row.append(np.nan)
                p_values.append(p_row)
            
            p_matrix = pd.DataFrame(p_values, 
                                  index=numeric_data.columns, 
                                  columns=numeric_data.columns)
            
            # p値行列表示
            header = "変数".ljust(15)
            for col in p_matrix.columns:
                header += f"{col[:8]:<10}"
            results.append(header)
            
            for idx, row in p_matrix.iterrows():
                row_str = f"{idx[:12]:<15}"
                for val in row:
                    if pd.isna(val):
                        row_str += "     -    "
                    elif val < 0.001:
                        row_str += "   <.001  "
                    else:
                        row_str += f"{val:>8.3f}  "
                results.append(row_str)
            
            results.append(f"\n観測数: {n}")
            results.append("* p < 0.05: 有意, p < 0.01: 高度に有意, p < 0.001: 極めて有意")
            
            result_text = "\n".join(results)
            self.log_result("Correlation Analysis", result_text)
            return result_text
            
        except Exception as e:
            error_msg = f"相関分析中にエラーが発生しました: {e}"
            self.log_result("Correlation Analysis Error", error_msg)
            return error_msg
    
    def t_test(self, column1, column2=None, paired=False, mu=0):
        """t検定"""
        if self.data is None:
            return "データが設定されていません。"
        
        try:
            results = []
            results.append("=" * 60)
            
            if column2 is None:
                # 1標本t検定
                results.append(f"1標本t検定 - {column1}")
                results.append("=" * 60)
                
                data1 = self.data[column1].dropna()
                
                if len(data1) < 2:
                    return "検定には少なくとも2つの観測値が必要です。"
                
                t_stat, p_value = stats.ttest_1samp(data1, mu)
                
                results.append(f"帰無仮説: μ = {mu}")
                results.append(f"対立仮説: μ ≠ {mu}")
                results.append(f"\n観測数: {len(data1)}")
                results.append(f"平均: {data1.mean():.4f}")
                results.append(f"標準偏差: {data1.std():.4f}")
                results.append(f"標準誤差: {data1.std()/np.sqrt(len(data1)):.4f}")
                results.append(f"\nt統計量: {t_stat:.4f}")
                results.append(f"自由度: {len(data1)-1}")
                results.append(f"p値: {p_value:.6f}")
                
            else:
                if paired:
                    # 対応ありt検定
                    results.append(f"対応ありt検定 - {column1} vs {column2}")
                    results.append("=" * 60)
                    
                    data1 = self.data[column1].dropna()
                    data2 = self.data[column2].dropna()
                    
                    # 共通インデックス
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) < 2:
                        return "対応ありt検定には少なくとも2つの対応するデータが必要です。"
                    
                    data1_paired = data1[common_idx]
                    data2_paired = data2[common_idx]
                    
                    t_stat, p_value = stats.ttest_rel(data1_paired, data2_paired)
                    
                    diff = data1_paired - data2_paired
                    
                    results.append(f"帰無仮説: μ_diff = 0")
                    results.append(f"対立仮説: μ_diff ≠ 0")
                    results.append(f"\n対応数: {len(common_idx)}")
                    results.append(f"{column1} 平均: {data1_paired.mean():.4f}")
                    results.append(f"{column2} 平均: {data2_paired.mean():.4f}")
                    results.append(f"差の平均: {diff.mean():.4f}")
                    results.append(f"差の標準偏差: {diff.std():.4f}")
                    
                else:
                    # 対応なしt検定
                    results.append(f"対応なしt検定 - {column1} vs {column2}")
                    results.append("=" * 60)
                    
                    data1 = self.data[column1].dropna()
                    data2 = self.data[column2].dropna()
                    
                    if len(data1) < 2 or len(data2) < 2:
                        return "各群に少なくとも2つの観測値が必要です。"
                    
                    # 等分散性の検定
                    levene_stat, levene_p = stats.levene(data1, data2)
                    equal_var = levene_p > 0.05
                    
                    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
                    
                    results.append(f"帰無仮説: μ1 = μ2")
                    results.append(f"対立仮説: μ1 ≠ μ2")
                    results.append(f"\nLevene等分散性検定:")
                    results.append(f"  統計量: {levene_stat:.4f}, p値: {levene_p:.6f}")
                    results.append(f"  等分散性: {'仮定する' if equal_var else '仮定しない'}")
                    results.append(f"\n{column1}群:")
                    results.append(f"  観測数: {len(data1)}")
                    results.append(f"  平均: {data1.mean():.4f}")
                    results.append(f"  標準偏差: {data1.std():.4f}")
                    results.append(f"\n{column2}群:")
                    results.append(f"  観測数: {len(data2)}")
                    results.append(f"  平均: {data2.mean():.4f}")
                    results.append(f"  標準偏差: {data2.std():.4f}")
                
                results.append(f"\nt統計量: {t_stat:.4f}")
                if paired:
                    results.append(f"自由度: {len(common_idx)-1}")
                else:
                    if equal_var:
                        df = len(data1) + len(data2) - 2
                    else:
                        # Welchのt検定の自由度
                        s1, s2 = data1.std(), data2.std()
                        n1, n2 = len(data1), len(data2)
                        df = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
                    results.append(f"自由度: {df:.2f}")
                results.append(f"p値: {p_value:.6f}")
            
            # 効果サイズ (Cohen's d)
            if column2 is not None and not paired:
                pooled_std = np.sqrt(((len(data1)-1)*data1.var() + (len(data2)-1)*data2.var()) / (len(data1)+len(data2)-2))
                cohens_d = abs(data1.mean() - data2.mean()) / pooled_std
                results.append(f"Cohen's d: {cohens_d:.4f}")
            
            # 結果の解釈
            results.append(f"\n結果の解釈:")
            alpha = 0.05
            if p_value < 0.001:
                results.append(f"p < 0.001 (極めて有意)")
            elif p_value < 0.01:
                results.append(f"p < 0.01 (高度に有意)")
            elif p_value < alpha:
                results.append(f"p < 0.05 (有意)")
            else:
                results.append(f"p ≥ 0.05 (有意差なし)")
            
            result_text = "\n".join(results)
            self.log_result("t-test", result_text)
            return result_text
            
        except Exception as e:
            error_msg = f"t検定中にエラーが発生しました: {e}"
            self.log_result("t-test Error", error_msg)
            return error_msg
    
    def anova_test(self, dependent_var, independent_var, post_hoc=True):
        """一元配置分散分析"""
        if self.data is None:
            return "データが設定されていません。"
        
        try:
            results = []
            results.append("=" * 60)
            results.append(f"一元配置分散分析 (One-way ANOVA)")
            results.append(f"従属変数: {dependent_var}, 独立変数: {independent_var}")
            results.append("=" * 60)
            
            # データ準備
            clean_data = self.data[[dependent_var, independent_var]].dropna()
            groups = clean_data.groupby(independent_var)[dependent_var].apply(list)
            
            if len(groups) < 2:
                return "分散分析には少なくとも2つの群が必要です。"
            
            # 各群の基本統計
            results.append("群別基本統計:")
            results.append("-" * 50)
            results.append(f"{'群':<15} {'観測数':<8} {'平均':<10} {'標準偏差':<10}")
            results.append("-" * 50)
            
            total_n = 0
            for group_name, group_data in groups.items():
                n = len(group_data)
                mean = np.mean(group_data)
                std = np.std(group_data, ddof=1)
                results.append(f"{str(group_name):<15} {n:<8} {mean:<10.4f} {std:<10.4f}")
                total_n += n
            
            # ANOVA実行
            group_values = [group_data for group_data in groups.values()]
            f_stat, p_value = stats.f_oneway(*group_values)
            
            # 効果量 (eta squared)
            grand_mean = clean_data[dependent_var].mean()
            ss_between = sum([len(group) * (np.mean(group) - grand_mean)**2 for group in group_values])
            ss_total = sum([(x - grand_mean)**2 for group in group_values for x in group])
            eta_squared = ss_between / ss_total
            
            results.append(f"\n分散分析結果:")
            results.append("-" * 30)
            results.append(f"F統計量: {f_stat:.4f}")
            results.append(f"p値: {p_value:.6f}")
            results.append(f"効果量 (η²): {eta_squared:.4f}")
            
            # 結果解釈
            results.append(f"\n結果の解釈:")
            if p_value < 0.001:
                results.append("p < 0.001 (極めて有意): 群間に有意差があります")
            elif p_value < 0.01:
                results.append("p < 0.01 (高度に有意): 群間に有意差があります")
            elif p_value < 0.05:
                results.append("p < 0.05 (有意): 群間に有意差があります")
            else:
                results.append("p ≥ 0.05 (有意差なし): 群間に有意差は認められません")
            
            result_text = "\n".join(results)
            self.log_result("ANOVA", result_text)
            return result_text
            
        except Exception as e:
            error_msg = f"分散分析中にエラーが発生しました: {e}"
            self.log_result("ANOVA Error", error_msg)
            return error_msg
    
    def regression_analysis(self, dependent_var, independent_vars, standardize=False):
        """重回帰分析"""
        if self.data is None:
            return "データが設定されていません。"
        
        try:
            results = []
            results.append("=" * 60)
            results.append("重回帰分析 (Multiple Regression Analysis)")
            results.append("=" * 60)
            
            # データ準備
            if isinstance(independent_vars, str):
                independent_vars = [independent_vars]
            
            all_vars = [dependent_var] + independent_vars
            clean_data = self.data[all_vars].dropna()
            
            if len(clean_data) < len(independent_vars) + 2:
                return "回帰分析には十分なデータが必要です。"
            
            X = clean_data[independent_vars]
            y = clean_data[dependent_var]
            
            # 標準化
            if standardize:
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
                
                model = LinearRegression()
                model.fit(X_scaled, y_scaled)
                
                results.append("標準化回帰分析結果:")
                coefficients = model.coef_
                intercept = model.intercept_
            else:
                model = LinearRegression()
                model.fit(X, y)
                coefficients = model.coef_
                intercept = model.intercept_
            
            # 予測と残差
            y_pred = model.predict(X_scaled if standardize else X)
            residuals = (y_scaled if standardize else y) - y_pred
            
            # モデル適合度
            r2 = model.score(X_scaled if standardize else X, y_scaled if standardize else y)
            n = len(clean_data)
            k = len(independent_vars)
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
            
            # 標準誤差
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            
            results.append(f"\nモデル適合度:")
            results.append("-" * 30)
            results.append(f"観測数: {n}")
            results.append(f"決定係数 (R²): {r2:.4f}")
            results.append(f"調整済み決定係数: {adjusted_r2:.4f}")
            results.append(f"標準誤差 (RMSE): {rmse:.4f}")
            
            # 回帰係数
            results.append(f"\n回帰係数:")
            results.append("-" * 50)
            results.append(f"{'変数':<15} {'係数':<12} {'標準誤差':<12} {'t値':<10} {'p値':<12}")
            results.append("-" * 65)
            
            # 切片
            results.append(f"{'(定数)':<15} {intercept:<12.4f}")
            
            # 各独立変数の係数
            for i, var in enumerate(independent_vars):
                coef = coefficients[i]
                
                # t検定（簡易版）
                se_coef = rmse / np.sqrt(np.sum((X.iloc[:, i] - X.iloc[:, i].mean())**2))
                t_value = coef / se_coef
                p_value = 2 * (1 - stats.t.cdf(abs(t_value), n - k - 1))
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                
                results.append(f"{var:<15} {coef:<12.4f} {se_coef:<12.4f} {t_value:<10.4f} {p_value:<12.6f} {significance}")
            
            result_text = "\n".join(results)
            self.log_result("Regression Analysis", result_text)
            return result_text
            
        except Exception as e:
            error_msg = f"回帰分析中にエラーが発生しました: {e}"
            self.log_result("Regression Analysis Error", error_msg)
            return error_msg
    
    def chi_square_test(self, var1, var2=None, expected_freq=None):
        """カイ二乗検定"""
        if self.data is None:
            return "データが設定されていません。"
        
        try:
            results = []
            results.append("=" * 60)
            
            if var2 is None:
                # 適合度検定
                results.append(f"カイ二乗適合度検定 - {var1}")
                results.append("=" * 60)
                
                observed = self.data[var1].value_counts().sort_index()
                
                if expected_freq is None:
                    # 等確率を仮定
                    expected_freq = [len(self.data[var1]) / len(observed)] * len(observed)
                
                chi2, p_value = stats.chisquare(observed, expected_freq)
                
                results.append("観測度数 vs 期待度数:")
                results.append("-" * 40)
                results.append(f"{'カテゴリ':<15} {'観測度数':<12} {'期待度数':<12}")
                results.append("-" * 40)
                
                for i, (cat, obs) in enumerate(observed.items()):
                    exp = expected_freq[i] if i < len(expected_freq) else expected_freq[0]
                    results.append(f"{str(cat):<15} {obs:<12} {exp:<12.2f}")
                
            else:
                # 独立性検定
                results.append(f"カイ二乗独立性検定 - {var1} × {var2}")
                results.append("=" * 60)
                
                # クロス表作成
                contingency_table = pd.crosstab(self.data[var1], self.data[var2])
                
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                
                results.append("クロス表 (観測度数):")
                results.append("-" * 50)
                results.append(contingency_table.to_string())
                
                results.append(f"\n期待度数:")
                results.append("-" * 30)
                expected_df = pd.DataFrame(expected, 
                                         index=contingency_table.index, 
                                         columns=contingency_table.columns)
                results.append(expected_df.round(2).to_string())
                
                # Cramer's V (効果量)
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                
                results.append(f"\nCramer's V: {cramers_v:.4f}")
                results.append(f"自由度: {dof}")
            
            results.append(f"\nカイ二乗統計量: {chi2:.4f}")
            results.append(f"p値: {p_value:.6f}")
            
            # 結果解釈
            results.append(f"\n結果の解釈:")
            if p_value < 0.001:
                results.append("p < 0.001 (極めて有意)")
            elif p_value < 0.01:
                results.append("p < 0.01 (高度に有意)")
            elif p_value < 0.05:
                results.append("p < 0.05 (有意)")
            else:
                results.append("p ≥ 0.05 (有意差なし)")
            
            result_text = "\n".join(results)
            self.log_result("Chi-square test", result_text)
            return result_text
            
        except Exception as e:
            error_msg = f"カイ二乗検定中にエラーが発生しました: {e}"
            self.log_result("Chi-square test Error", error_msg)
            return error_msg 