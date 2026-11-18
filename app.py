from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Memuat model dan dataset untuk visualisasi grafik interaktif
model = joblib.load('model_regresi.pkl')
df = pd.read_csv('dataset/student_scores.csv')

# Mengambil parameter model untuk ditampilkan di UI
intercept = round(float(model.intercept_), 2)
slope = round(float(model.coef_[0]), 2)

# Menyiapkan data titik aktual dan garis regresi untuk Chart.js
scatter_data = [{'x': float(row['Hours']), 'y': float(row['Scores'])} for _, row in df.iterrows()]
x_min, x_max = float(df['Hours'].min()), float(df['Hours'].max())
line_data = [
    {'x': 0, 'y': round(float(model.predict([[0]])[0]), 2)},
    {'x': 10, 'y': round(float(model.predict([[10]])[0]), 2)}
]

def get_kategori(skor):
    if skor >= 85:
        return ("Sangat Baik (Grade A)", "success", "Pertahankan konsistensi belajar Anda!")
    elif skor >= 70:
        return ("Baik (Grade B)", "primary", "Hasil sudah bagus, sedikit tambahan waktu belajar bisa meraih nilai maksimal.")
    elif skor >= 55:
        return ("Cukup (Grade C)", "warning", "Disarankan menambah durasi belajar dan latihan soal secara rutin.")
    else:
        return ("Perlu Ditingkatkan (Grade D)", "danger", "Durasi belajar masih minim. Coba tingkatkan fokus dan jam belajar harian.")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediksi_skor = None
    jam_belajar = None
    error_msg = None
    kategori = None

    if request.method == 'POST':
        try:
            jam_belajar = float(request.form['hours'])
            if jam_belajar < 0 or jam_belajar > 24:
                error_msg = "Masukkan durasi jam belajar yang logis (antara 0 hingga 24 jam)."
            else:
                input_data = np.array([[jam_belajar]])
                hasil = model.predict(input_data)[0]
                prediksi_skor = round(min(max(hasil, 0), 100), 2)
                kategori = get_kategori(prediksi_skor)
        except ValueError:
            error_msg = "Input tidak valid. Harap masukkan angka."

    return render_template(
        'index.html',
        prediksi=prediksi_skor,
        jam=jam_belajar,
        error=error_msg,
        kategori=kategori,
        intercept=intercept,
        slope=slope,
        scatter_data=scatter_data,
        line_data=line_data
    )

if __name__ == '__main__':
    app.run(debug=True)