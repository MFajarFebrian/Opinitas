"""Preprocessing pipeline + LR inference for the Opinitas dashboard.

Replicates the ``clean_strict`` preprocessing from NB02 and loads the
Logistic Regression models from NB05 (sentimen + aspek).
"""

import re
import os
import pathlib
import joblib
import pandas as pd
import numpy as np
from functools import lru_cache

ROOT = pathlib.Path(__file__).resolve().parents[2]

# ── Preprocessing constants (copied verbatim from NB02) ──

PROTECTED_WORDS = [
    "lemot",
    "berikan",
    "mengerti", "memuaskan", "langganan", "berlangganan", "percakapan",
    "pengalaman",
    "pembelajaran", "pengguna",
    "belajar", "perbaiki", "diperbaiki", "perbaikan", "pengetahuan",
    "penjelasan",
    "kekurangan", "tampilan", "jaringan",
    "terbatas"
]

CUSTOM_SLANG = {
    "enggak": "tidak", "banget": "sangat", "bgt": "sangat", "gue": "saya", "gw": "saya",
    "lu": "kamu", "kalo": "kalau", "eror": "error", "baguss": "bagus", "bgus": "bagus",
    "bagu": "bagus", "tuga": "tugas", "trimakasih": "terima kasih", "appk": "aplikasi",
    "apk": "aplikasi", "bikin": "buat", "ngedit": "edit", "mantul": "mantap", "good": "bagus",
    "nice": "bagus", "best": "terbaik", "ok": "baik", "oke": "baik", "thanks": "terima kasih",
    "thx": "terima kasih", "aplikasih": "aplikasi", "sangant": "sangat"
}

MIN_WORDS = 3


def _normalize_repeated_chars(text: str) -> str:
    """Collapse elongated character runs (e.g. 'bagusss' -> 'bagus')."""
    return re.sub(r"(.)\1{2,}", r"\1", text)


def clean_strict(text: str) -> str:
    """Full strict preprocessing pipeline — replicates NB02 ``clean_strict``.

    Steps: case-fold -> noise removal -> slang normalization ->
    tokenize -> stopword removal (Sastrawi) -> stemming (Sastrawi)
    with protected-word shielding.
    """
    import emoji
    from indoNLP.preprocessing import replace_slang
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = emoji.replace_emoji(text, replace="")
    text = _normalize_repeated_chars(text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Slang normalization (indoNLP + custom)
    text = replace_slang(text)
    text = " ".join([CUSTOM_SLANG.get(w, w) for w in text.split()])

    # Tokenize
    tokens = text.split()

    # --- Protected word shielding via placeholders ---
    protected_map = {}
    processed_words = []
    for i, word in enumerate(tokens):
        if word in PROTECTED_WORDS:
            placeholder = f"protectword{i}"
            protected_map[placeholder] = word
            processed_words.append(placeholder)
        else:
            processed_words.append(word)

    # Stopword Removal
    text_joined = " ".join(processed_words)
    text_no_stop = _get_stopword_remover().remove(text_joined)

    # Stemming
    text_stemmed = _get_stemmer().stem(text_no_stop)

    # Restore protected words
    stemmed_words = text_stemmed.split()
    final_words = [protected_map.get(w, w) for w in stemmed_words]
    text = " ".join(final_words)
    text = " ".join(text.split())
    return text


@lru_cache(maxsize=1)
def _get_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    return StemmerFactory().create_stemmer()


@lru_cache(maxsize=1)
def _get_stopword_remover():
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    return StopWordRemoverFactory().create_stop_word_remover()


@lru_cache(maxsize=1)
def _load_models():
    """Lazily load LR models + TF-IDF vectorizers."""
    models_dir = ROOT / "models"
    data_dir = ROOT / "data"

    lr_sent = joblib.load(models_dir / "lr_sentiment.pkl")
    lr_asp = joblib.load(models_dir / "lr_aspect_classifier.pkl")
    vec_sent = joblib.load(data_dir / "tfidf_vectorizer_sent.pkl")
    vec_asp = joblib.load(data_dir / "tfidf_vectorizer_aspect.pkl")

    return {
        "lr_sentiment": lr_sent,
        "lr_aspect": lr_asp,
        "vec_sentiment": vec_sent,
        "vec_aspect": vec_asp,
    }


def predict_batch_lr(texts: list[str], progress_callback=None) -> pd.DataFrame:
    """Run full ABSA pipeline on a list of raw review texts.

    Returns DataFrame with columns:
        text_original, text_clean, sentimen_label, sentimen_conf,
        aspek, aspek_conf

    ``progress_callback``: optional callable(step_name, pct) where
        step_name is a human-readable string and pct is 0-100.
    """
    if not texts:
        return pd.DataFrame()

    def _cb(step, pct):
        if progress_callback:
            progress_callback(step, pct)

    _cb("Memuat model Logistic Regression...", 5)
    m = _load_models()

    # Preprocess each text
    _cb("Preprocessing teks (case folding, noise removal, slang normalization, stopword, stemming)...", 10)
    cleaned = []
    total = len(texts)
    for i, t in enumerate(texts):
        cleaned.append(clean_strict(t))
        if total > 0 and i % max(1, total // 20) == 0:
            pct = 10 + int((i / total) * 35)  # 10% -> 45%
            _cb("Preprocessing teks (case folding, noise removal, slang normalization, stopword, stemming)...", pct)

    _cb("Preprocessing teks selesai", 45)

    # Filter empty results
    results = []
    for orig, clean in zip(texts, cleaned):
        if clean.strip():
            results.append((orig, clean))
        else:
            results.append((orig, ""))

    clean_texts = [r[1] if r[1] else "kosong" for r in results]
    originals = [r[0] for r in results]

    # TF-IDF transform
    _cb("Vektorisasi TF-IDF (sentimen + aspek)...", 50)
    X_sent = m["vec_sentiment"].transform(clean_texts)
    X_asp = m["vec_aspect"].transform(clean_texts)
    _cb("Vektorisasi TF-IDF selesai", 60)

    # Predict sentimen
    _cb("Klasifikasi sentimen (Logistic Regression)...", 65)
    sent_labels = m["lr_sentiment"].predict(X_sent)
    sent_proba = m["lr_sentiment"].predict_proba(X_sent)
    sent_classes = m["lr_sentiment"].classes_
    _cb("Klasifikasi sentimen selesai", 80)

    # Predict aspek
    _cb("Klasifikasi aspek (Logistic Regression)...", 85)
    asp_labels = m["lr_aspect"].predict(X_asp)
    asp_proba = m["lr_aspect"].predict_proba(X_asp)
    asp_classes = m["lr_aspect"].classes_
    _cb("Klasifikasi aspek selesai", 95)

    # Build confidence (max prob)
    _cb("Menggabungkan hasil ABSA...", 97)
    rows = []
    for i in range(len(originals)):
        sent_idx = list(sent_classes).index(sent_labels[i])
        asp_idx = list(asp_classes).index(asp_labels[i])

        rows.append({
            "text_original": originals[i],
            "text_clean": clean_texts[i],
            "sentimen_label": sent_labels[i],
            "sentimen_conf": float(sent_proba[i][sent_idx]),
            "aspek": asp_labels[i],
            "aspek_conf": float(asp_proba[i][asp_idx]),
        })

    _cb("Pipeline ABSA selesai", 100)
    return pd.DataFrame(rows)
