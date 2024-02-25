import numpy as np
import math
import pandas as pd

def hitung_standar_deviasi(data):
    n = len(data)
    mean = sum(data) / n
    varian = sum([((x - mean) ** 2) for x in data]) / n
    standar_deviasi = math.sqrt(varian)
    return standar_deviasi

def hitung_regresi_linier_sederhana(data_x, data_y):
    if len(data_x) != len(data_y):
        raise ValueError("Panjang data tidak sama!")

    mean_x = np.mean(data_x)
    mean_y = np.mean(data_y)

    numerator = sum([(data_x[i] - mean_x) * (data_y[i] - mean_y) for i in range(len(data_x))])
    denominator = sum([(data_x[i] - mean_x)**2 for i in range(len(data_x))])
    b1 = numerator / denominator

    b0 = mean_y - b1 * mean_x

    return (b0, b1)

def hitung_error(data_x, data_y, b0, b1):
    y_estimasi = [b0 + b1*x for x in data_x]
    error = [data_y[i] - y_estimasi[i] for i in range(len(data_x))]
    return error, y_estimasi

# Implementasi menu
while True:
    print("1. Hitung standar deviasi")
    print("2. Hitung regresi linier sederhana")
    print("3. Keluar")
    choice = input("Masukkan pilihan (1/2/3): ")

    if choice == "1":
        data = [1, 2, 3, 4, 5]  # Contoh data, sesuaikan dengan data yang ingin dihitung standar deviasinya
        sd = hitung_standar_deviasi(data)
        print("Standar deviasi data: ", sd)

    elif choice == "2":
        file_path = "regresi.csv"  # Ubah dengan path file CSV Anda
        data_frame = pd.read_csv(file_path)
        data_x = data_frame["X"].values.astype(float)
        data_y = data_frame["Y"].values.astype(float)
        b0, b1 = hitung_regresi_linier_sederhana(data_x, data_y)
        print("Nilai b0 (intercept) = ", round(b0, 3))
        print("Nilai b1 (slope) = ", round(b1, 3))

        error, y_estimasi = hitung_error(data_x, data_y, b0, b1)
        print("Error: ", error)
        print("Y estimasi: ", y_estimasi)

    elif choice == "3":
        break

    else:
        print("Pilihan tidak valid. Silakan pilih 1, 2, atau 3.")
