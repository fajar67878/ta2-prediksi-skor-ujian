import os
import csv
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'student_scores.csv')

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Parameter hasil pelatihan model Regresi Linear dari train_model.py
INTERCEPT = 2.83
SLOPE = 9.68

# Membaca dataset CSV menggunakan modul bawaan Python
scatter_data = []
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scatter_data.append({
                'x': float(row['Hours']),
                'y': float(row['Scores'])
            })

line_data = [
    {'x': 0, 'y': round(INTERCEPT, 2)},
    {'x': 10, 'y': round(INTERCEPT + (SLOPE * 10), 2)}
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
@app.route('/<path:path>', methods=['GET', 'POST'])
def index(path=None):
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
                hasil = INTERCEPT + (SLOPE * jam_belajar)
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
        intercept=INTERCEPT,
        slope=SLOPE,
        scatter_data=scatter_data,
        line_data=line_data
    )

if __name__ == '__main__':
    app.run(debug=True)