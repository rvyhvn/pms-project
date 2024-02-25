import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


class MultipleLinearRegression:
    def __init__(self, file_path):
        self.data_frame = pd.read_csv(file_path)
        self.data_y = self.data_frame["Y"].values.astype(float)
        self.data_x = self.data_frame.drop("Y", axis=1).values.astype(float)
        self.n = len(self.data_x)
        self.num_features = len(self.data_x[0]) + 1

    def _add_intercept(self):
        return np.column_stack((np.ones(self.n), self.data_x))

    def fit(self):
        x = self._add_intercept()
        coefficients = np.linalg.inv(x.T @ x) @ x.T @ self.data_y
        self.b0 = coefficients[0]
        self.b = coefficients[1:]

    def predict(self):
        x = self._add_intercept()
        return x @ np.concatenate(([self.b0], self.b))

    def calculate_error(self):
        y_estimasi = self.predict()
        return self.data_y - y_estimasi

    def plot_regression(self, file_path):
        # Mengambil kolom pertama dari data_x sebagai variabel X
        x = self.data_x[:, 0]
        y_actual = self.data_y
        y_estimated = self.b0 + np.dot(self.data_x, self.b)

        plt.scatter(y_actual, y_estimated, label='Data')
        plt.plot([min(y_actual), max(y_actual)], [min(y_actual),
                 max(y_actual)], color='red', label='Regression Line')
        plt.xlabel('Actual Y')
        plt.ylabel('Estimated Y')
        plt.title(f'Scatter Plot with Linear Regression {file_path}')
        plt.legend()
        plt.grid(True)
        plt.show()


class DataAnalyzer:
    def __init__(self, file_path):
        self.data_frame = pd.read_csv(file_path)

    def calculate_correlation(self):
        correlation = {}
        data_y = self.data_frame["Y"].values.astype(float)

        for column in self.data_frame.columns:
            if column != "Y":
                data_x = self.data_frame[column].values.astype(float)
                correlation[column] = np.corrcoef(data_y, data_x)[0, 1]
                if np.var(data_x) == 0:
                    correlation[column] = 0

        return correlation

    def calculate_statistics(self):
        statistics = {}

        for column in self.data_frame.columns:
            data = self.data_frame[column].values.astype(float)
            statistics[column] = {
                "Mean": np.mean(data),
                "Median": np.median(data),
                "Maximum": np.max(data),
                "Minimum": np.min(data),
                "Range": np.ptp(data),
                "Std. Deviasi": np.std(data),
                "Variansi": np.var(data, ddof=1),
                "Std. Error": stats.sem(data),
                "Mode": stats.mode(data, keepdims=True).mode[0],
                "Kurtosis": stats.kurtosis(data),
                "Skewness": stats.skew(data),
                "Sum": np.sum(data),
                "Count": len(data),
            }

        return statistics

    def calculate_anova(self):
        anova = {}

        for column in self.data_frame.columns:
            if column != "Y":
                data = self.data_frame[column].values.astype(float)
                anova_result = stats.f_oneway(data, self.data_frame["Y"])
                anova[column] = {
                    "F-value": anova_result.statistic,
                    "p-value": anova_result.pvalue,
                }
        y_mean = np.mean(self.data_frame["Y"].values.astype(float))
        ss_total = np.sum(
            np.square(self.data_frame["Y"].values.astype(float) - y_mean))
        df_total = len(self.data_frame["Y"].values.astype(float)) - 1

        mse = ss_total / df_total

        for column, result in anova.items():
            df_group = 1
            df_residual = df_total - df_group

            ss_group = result["F-value"] * df_group
            ss_residual = ss_total - ss_group

            msr = ss_group / df_group
            mse = ss_residual / df_residual

            anova[column].update({
                "DF Group": df_group,
                "DF Residual": df_residual,
                "SS Group": ss_group,
                "SS Residual": ss_residual,
                "MSE": mse,
                "MSR": msr
            })

        return anova


# Contoh penggunaan kelas-kelas di atas
file_paths = ["data2020.csv", "data2021.csv", "data2022.csv"]

for file_path in file_paths:
    print(f"Calculate statistics untuk {file_path}:")
    data_analyzer = DataAnalyzer(file_path)
    statistics = data_analyzer.calculate_statistics()

    for column, stat in statistics.items():
        print(f"Statistik untuk variabel {column}")
        print(f"Mean: {stat['Mean']:.3f}")
        print(f"Median: {stat['Median']:.3f}")
        print(f"Maximum: {stat['Maximum']:.3f}")
        print(f"Minimum: {stat['Minimum']:.3f}")
        print(f"Range: {stat['Range']:.3f}")
        print(f"Std. Deviasi: {stat['Std. Deviasi']:.3f}")
        print(f"Variansi: {stat['Variansi']:.3f}")
        print(f"Std. Error: {stat['Std. Error']:.3f}")
        print(f"Mode: {stat['Mode']:.3f}")
        print(f"Kurtosis: {stat['Kurtosis']:.3f}")
        print(f"Skewness: {stat['Skewness']:.3f}")
        print(f"Sum: {stat['Sum']:.3f}")
        print(f"Count: {stat['Count']}")
        print()

    print(f"Calculate correlation untuk {file_path}:")
    correlation = data_analyzer.calculate_correlation()
    for column, corr in correlation.items():
        print(f"Korelasi antara Y dan {column}: {corr:.3f}")
        data_x = data_analyzer.data_frame[column].values.astype(float)
        data_y = data_analyzer.data_frame["Y"].values.astype(float)
        print()

    print(f"Mencetak hasil regresi linear berganda untuk {file_path}:")
    regression_model = MultipleLinearRegression(file_path)
    regression_model.fit()
    print("Nilai b0 (intercept):", round(regression_model.b0, 3))
    print("Nilai b (koefisien):", np.round(regression_model.b, 3))
    print("Y estimasi:", regression_model.predict())
    print("Nilai error:", regression_model.calculate_error())
    print()
    regression_model.plot_regression(file_path)

    print(f"Calculate ANOVA untuk {file_path}:")
    anova = data_analyzer.calculate_anova()
    for column, result in anova.items():
        print(f"Hasil ANOVA untuk variabel {column}")
        print(f"F-value: {result['F-value']:.3f}")
        print(f"p-value: {result['p-value']:.3f}")
        print(f"DF Group: {result['DF Group']}")
        print(f"DF Residual: {result['DF Residual']}")
        print(f"SS Group: {result['SS Group']:.3f}")
        print(f"SS Residual: {result['SS Residual']:.3f}")
        print(f"MSE: {result['MSE']:.3f}")
        print(f"MSR: {result['MSR']:.3f}")

        if result['p-value'] < 0.05:
            print("H0 ditolak. Terdapat perbedaan yang signifikan di antara kelompok.")
        else:
            print(
                "H0 diterima. Tidak terdapat perbedaan yang signifikan di antara kelompok.")
    print()
