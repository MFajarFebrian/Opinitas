# Opinitas

Dashboard analisa ulasan sentimen dan aspek aplikasi AI pada Google Play Store menggunakan **Logistic Regression**.

## Fitur

- Scraping ulasan langsung dari Google Play Store
- Klasifikasi sentimen (positif / negatif) dengan Logistic Regression
- Klasifikasi aspek (UI/UX, Kualitas Konten, Fitur, Performa) dengan Logistic Regression
- Visualisasi: donut chart, stacked bar per aspek, tren sentimen bulanan
- Top 10 kata kunci per aspek
- Tabel detail ulasan dengan filter dan pagination
- Download hasil analisis dalam CSV

## Quick Start

```bash
# Clone
git clone https://github.com/MFajarFebrian/Opinitas.git
cd Opinitas

# Install dependencies
pip install -r dashboard/requirements.txt

# Jalankan dashboard
streamlit run dashboard/app.py
```

Buka `http://localhost:8501` di browser.

## Struktur Repo

```
Opinitas/
├── dashboard/
│   ├── app.py                    # Entry point Streamlit
│   ├── requirements.txt          # Python dependencies
│   ├── .streamlit/
│   │   └── config.toml           # Theme config
│   └── utils/
│       ├── __init__.py
│       ├── scraper.py            # Google Play scraping
│       ├── pipeline.py           # Preprocessing + LR inference
│       └── charts.py             # Plotly + Wordcloud helpers
├── models/
│   ├── lr_sentiment.pkl          # Logistic Regression (sentimen)
│   └── lr_aspect_classifier.pkl  # Logistic Regression (aspek)
├── data/
│   ├── tfidf_vectorizer_sent.pkl     # TF-IDF vectorizer (5.000 fitur)
│   ├── tfidf_vectorizer_aspect.pkl   # TF-IDF vectorizer (15.000 fitur)
│   └── roberta_labeled_final.csv     # Dataset training (8.208 baris)
└── README.md
```

## Model

| Model | Task | Accuracy | Macro-F1 | Params |
|-------|------|----------|----------|--------|
| Logistic Regression | Sentimen | 91.96% | 89.72% | C=10.0, lbfgs |
| Logistic Regression | Aspek | 89.77% | 83.47% | C=20.0, lbfgs |

### Aspek (dari BERTopic NB03)

| Aspek | Deskripsi |
|-------|-----------|
| UI/UX | Pengalaman pengguna, antarmuka, kemudahan |
| Kualitas Konten | Kualitas jawaban, akurasi AI |
| Fitur | Fiturgambar, video, edit, canvas |
| Performa | Error, crash, lemot, bug |

## Pipeline

```
Ulasan mentah
  → clean_strict(): case folding → noise removal → slang normalization
    → tokenize → stopword removal (Sastrawi) → stemming (Sastrawi)
  → TF-IDF transform (sentimen 5K fitur, aspek 15K fitur)
  → LR predict (sentimen + aspek)
  → Output: (aspek, sentimen) pair per ulasan
```

## Teknologi

- Python 3.13
- Streamlit (dashboard)
- scikit-learn (Logistic Regression, TF-IDF)
- Sastrawi + indoNLP (preprocessing Bahasa Indonesia)
- Plotly (visualisasi)
- google-play-scraper (scraping ulasan)
