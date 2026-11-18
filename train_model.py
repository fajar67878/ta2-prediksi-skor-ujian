import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib

# 1. Memuat Dataset
df = pd.read_csv('dataset/student_scores.csv')
print("--- 5 Baris Pertama Dataset ---")
print(df.head())
print("\n--- Informasi Missing Values ---")
print(df.isnull().sum())

# 2. Eksplorasi Data (EDA) - Validasi Tren Linear
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Hours', y='Scores', data=df, color='blue', s=70)
plt.title('Hubungan Jumlah Jam Belajar vs Skor Ujian')
plt.xlabel('Jumlah Jam Belajar (Hours)')
plt.ylabel('Skor Ujian (Scores)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('eda_scatterplot.png') # Simpan grafik untuk lampiran laporan
plt.show()

# 3. Persiapan Fitur (X) dan Target (y)
X = df[['Hours']]
y = df['Scores']

# Membagi data: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Pelatihan Model Regresi Linear Sederhana
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prediksi & Evaluasi Model
y_pred = model.predict(X_test)

mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print("\n--- Parameter Model ---")
print(f"Intercept (B0) : {model.intercept_:.4f}")
print(f"Koefisien (B1) : {model.coef_[0]:.4f}")

print("\n--- Evaluasi Performa ---")
print(f"RMSE           : {rmse:.4f}")
print(f"R-squared (R2) : {r2:.4f}")

# Visualisasi Garis Regresi (Best-Fit Line)
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Data Aktual (Test)')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Garis Regresi (Best-Fit)')
plt.title('Hasil Prediksi Regresi Linear pada Data Uji')
plt.xlabel('Jumlah Jam Belajar (Hours)')
plt.ylabel('Skor Ujian (Scores)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('hasil_regresi.png') # Simpan grafik untuk lampiran laporan
plt.show()

# 6. Simpan Model dengan Joblib
joblib.dump(model, 'model_regresi.pkl')
print("\nModel berhasil disimpan ke 'model_regresi.pkl'")