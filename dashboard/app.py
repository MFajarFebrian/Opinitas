import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Opinitas",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.scraper import scrape_reviews, search_apps
from utils.pipeline import predict_batch_lr
from utils.charts import donut_sentiment, bar_aspect, trend_chart


def _render_kpi(df):
    """KPI cards for end-users: total, avg rating, % positive, top aspect."""
    valid = df[df["sentimen_label"].isin(["Positif", "Negatif"])]

    avg_rating = df["score"].mean() if "score" in df.columns and len(df) else 0
    pos_pct = (len(valid[valid["sentimen_label"] == "Positif"]) / len(valid) * 100) if len(valid) else 0

    pos_sub = df[(df["sentimen_label"] == "Positif") & df["aspek"].notna()]
    top_praise = pos_sub["aspek"].value_counts().idxmax() if len(pos_sub) else "-"

    # Aspek dengan keluhan terbanyak (proporsi negatif tertinggi, min 5 ulasan)
    neg_sub = df[df["aspek"].notna() & df["sentimen_label"].isin(["Positif", "Negatif"])]
    if len(neg_sub) > 0:
        asp_counts = neg_sub.groupby("aspek")["sentimen_label"].count()
        valid_aspects = asp_counts[asp_counts >= 5].index
        if len(valid_aspects) > 0:
            neg_pct = (neg_sub[neg_sub["aspek"].isin(valid_aspects)]
                       .groupby("aspek")["sentimen_label"]
                       .apply(lambda x: (x == "Negatif").mean() * 100))
            worst_aspect = neg_pct.idxmax() if len(neg_pct) else "-"
        else:
            worst_aspect = "-"
    else:
        worst_aspect = "-"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Ulasan", f"{len(df):,}")
    c2.metric("Rata-rata Rating", f"{avg_rating:.1f} / 5")
    c3.metric("Sentimen Positif", f"{pos_pct:.1f}%")
    c4.metric("Aspek Terpopuler",
              (top_praise or "-").replace("_", " ").title())
    c5.metric("Aspek Bermasalah",
              (worst_aspect or "-").replace("_", " ").title(),
              help="Aspek dengan proporsi keluhan (sentimen negatif) tertinggi. "
                   "Hanya aspek dengan minimal 5 ulasan dipertimbangkan.")


def _stacked_aspect_bar(df):
    """Stacked horizontal bar: sentiment proportion per aspect."""
    d = df[df["aspek"].notna() & df["sentimen_label"].isin(["Positif", "Negatif"])].copy()
    if d.empty:
        return go.Figure(layout=dict(template="plotly_dark", title="Tidak ada data aspek"))
    grp = d.groupby(["aspek", "sentimen_label"]).size().reset_index(name="n")
    tot = grp.groupby("aspek")["n"].transform("sum")
    grp["pct"] = grp["n"] / tot * 100
    fig = px.bar(
        grp, y="aspek", x="pct", color="sentimen_label", orientation="h",
        color_discrete_map={"Positif": "#2ecc71", "Negatif": "#e74c3c"},
        template="plotly_dark", title="Proporsi Sentimen per Aspek",
        labels={"aspek": "", "pct": "%", "sentimen_label": ""},
    )
    fig.update_layout(barmode="stack", xaxis=dict(range=[0, 100]),
                      margin=dict(l=150, r=20, t=50, b=40), height=400)
    return fig


# ── HEADER ──
st.markdown("""<div style="text-align:center;padding:1rem 0;">
<h1 style="font-size:2.5rem;color:#E2E8F0;">
<span style="color:#7C3AED;">Opinitas</span>
</h1>
<p style="color:#94A3B8;">
Dashboard Analisa Ulasan Sentimen dan Aspek Aplikasi AI pada Google Play Store
</p>
</div>""", unsafe_allow_html=True)

# ── SESSION STATE ──
for key in ("df_result", "app_id", "n_reviews", "search_results", "selected_app_id"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── SIDEBAR ──
with st.sidebar:
    st.header("Pengambilan Data")

    AI_APPS = [
        ("Gemini", "com.google.android.apps.bard"),
        ("ChatGPT", "com.openai.chatgpt"),
        ("DeepSeek", "com.deepseek.chat"),
        ("Copilot", "com.microsoft.copilot"),
        ("Perplexity", "ai.perplexity.android"),
        ("Grok", "ai.x.grok"),
    ]
    st.markdown("**Aplikasi AI Populer**")
    ai_cols = st.columns(3)
    for i, (name, aid) in enumerate(AI_APPS):
        with ai_cols[i % 3]:
            if st.button(name, key=f"ai_{aid}", use_container_width=True):
                st.session_state["selected_app_id"] = aid
                st.session_state["search_results"] = None
                st.rerun()

    st.divider()

    search_q = st.text_input("Cari Aplikasi", placeholder="Ketik: gemini, chatgpt, whatsapp...")
    if st.button("Cari", use_container_width=True):
        if search_q.strip():
            with st.spinner("Mencari di Google Play..."):
                st.session_state["search_results"] = search_apps(search_q.strip())

    results = st.session_state.get("search_results") or []
    app_id = None
    if results:
        opts = [f"{r['title']} ({r['appId']})" for r in results]
        selected = st.selectbox("Pilih Aplikasi", range(len(opts)),
                                format_func=lambda i: opts[i])
        if selected is not None and selected < len(results):
            app_id = results[selected]["appId"]

    prefill = st.session_state.get("selected_app_id", "")
    if not app_id:
        app_id = st.text_input("Atau masukkan App ID manual",
                                value=prefill, placeholder="com.contoh.aplikasi")

    count = st.number_input("Jumlah Ulasan", min_value=50, max_value=100000, value=500, step=100)
    run = st.button("Tarik & Analisis", use_container_width=True)

# ── SCRAPE & PREDICT ──
if run:
    if not app_id:
        st.error("Masukkan App ID atau cari aplikasi terlebih dahulu.")
        st.stop()

    lang, country = ("id", "id")

    with st.spinner("Scraping ulasan dari Google Play..."):
        df_raw, msg = scrape_reviews(app_id, count=count, lang=lang, country=country)

    if df_raw is None or df_raw.empty:
        st.error(msg or "Tidak ada ulasan yang berhasil diambil.")
    else:
        st.success(f"{len(df_raw)} ulasan berhasil diambil.")

        progress_bar = st.progress(0, text="Memulai pipeline ABSA...")
        status_text = st.empty()

        def _on_progress(step_name, pct):
            progress_bar.progress(pct / 100, text=step_name)
            status_text.markdown(f"**{pct}%** — {step_name}")

        try:
            df_pred = predict_batch_lr(df_raw["content"].tolist(),
                                       progress_callback=_on_progress)
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Gagal menjalankan model: {e}")
            st.stop()

        progress_bar.empty()
        status_text.empty()

        if df_pred.empty:
            st.error("Model gagal menghasilkan prediksi.")
        else:
            df_pred["sentimen_label"] = df_pred["sentimen_label"].str.title()

            # ⚠️ Positional attach, BUKAN merge on content.
            # Merge pada kolom teks akan meledak secara kartesian ketika
            # ada ulasan dengan konten duplikat (umum di Google Play),
            # menghasilkan baris > jumlah yang diminta (misal 500 -> 650).
            df_result = df_raw.copy()

            # Jumlah baris prediksi bisa lebih pendek dari df_raw jika ada
            # teks kosong; ambil baris df_raw yang berpasangan dengan prediksi.
            n_pred = len(df_pred)
            n_raw = len(df_raw)
            if n_pred < n_raw:
                df_result = df_result.iloc[:n_pred].copy()

            pred_cols = ["sentimen_label", "sentimen_conf", "aspek",
                         "aspek_conf", "text_clean"]
            for col in pred_cols:
                if col in df_pred.columns:
                    df_result[col] = df_pred[col].values[: len(df_result)]

            df_result["at"] = pd.to_datetime(df_result["at"], errors="coerce")
            st.session_state["df_result"] = df_result
            st.session_state["app_id"] = app_id
            st.session_state["n_reviews"] = len(df_result)
            st.rerun()

# ── MAIN BODY ──
df_res = st.session_state["df_result"]

if df_res is None or df_res.empty:
    st.info("Masukkan App ID di sidebar lalu klik **Tarik & Analisis**.")
    st.stop()

st.caption(f"App: `{st.session_state['app_id']}` | "
           f"Ulasan: {len(df_res)} | Model: Logistic Regression")

_render_kpi(df_res)
st.divider()

a, b = st.columns([2, 3])
with a:
    st.plotly_chart(donut_sentiment(df_res, title=""), use_container_width=True)
with b:
    st.plotly_chart(_stacked_aspect_bar(df_res), use_container_width=True)
st.plotly_chart(bar_aspect(df_res, title=""), use_container_width=True)

# ── Tren Sentimen per Bulan ──
st.plotly_chart(trend_chart(df_res), use_container_width=True)

# ── Top 10 Keywords per Aspek ──
st.markdown("#### Top 10 Kata Kunci per Aspek")
from collections import Counter

stop = {"di", "ke", "dari", "yang", "dan", "ini", "itu", "saya", "aku",
        "ada", "aja", "sudah", "udah", "belum", "bisa", "tidak", "ga",
        "enggak", "gak", "gk", "buat", "untuk", "dengan", "atau", "juga",
        "kalau", "kalo", "klo", "karena", "tapi", "tp", "lagi", "lg",
        "lebih", "banget", "bgt", "sama", "ama", "jadi", "jd", "nya",
        "dong", "deh", "sih", "nih", "lah", "kok", "kan", "ya", "nah",
        "loh", "wah", "ih", "eh", "oh", "hai", "halo"}

aspect_kw = {}
for asp in sorted(df_res["aspek"].dropna().unique()):
    texts = df_res[df_res["aspek"] == asp]["content"].dropna()
    words = " ".join(texts).lower().split()
    counter = Counter(w for w in words if w not in stop and len(w) > 2)
    aspect_kw[asp] = [w for w, _ in counter.most_common(10)]

kw_rows = []
for asp, words in aspect_kw.items():
    kw_rows.append({"Aspek": asp.replace("_", " ").title(), **{f"#{i+1}": w for i, w in enumerate(words)}})
kw_df = pd.DataFrame(kw_rows).set_index("Aspek")
st.dataframe(kw_df, use_container_width=True)

# ── Filter + Paginated Data Table ──
st.markdown("#### Detail Ulasan per Aspek & Sentimen")

f1, f2, f3 = st.columns(3)
with f1:
    tbl_search = st.text_input("Cari kata kunci", "", placeholder="crash, bagus...", key="tbl_search")
with f2:
    all_aspects = sorted(df_res["aspek"].dropna().unique().tolist())
    tbl_aspek = st.multiselect("Filter Aspek", all_aspects, default=[], key="tbl_aspek",
                               format_func=lambda x: x.replace("_", " ").title())
with f3:
    tbl_sent = st.multiselect("Filter Sentimen", ["Positif", "Negatif"], default=[], key="tbl_sent")

df_filtered = df_res.copy()
if tbl_search.strip():
    df_filtered = df_filtered[df_filtered["content"].str.contains(tbl_search, case=False, na=False)]
if tbl_aspek:
    df_filtered = df_filtered[df_filtered["aspek"].isin(tbl_aspek)]
if tbl_sent:
    df_filtered = df_filtered[df_filtered["sentimen_label"].isin(tbl_sent)]

n_per_page = 100
total = len(df_filtered)
total_pages = max(1, (total + n_per_page - 1) // n_per_page)
page = st.selectbox(
    "Pilih halaman", range(1, total_pages + 1),
    format_func=lambda p: f"Halaman {p} dari {total_pages}",
    key="page_select"
)

start = (page - 1) * n_per_page
end = start + n_per_page
df_page = df_filtered.iloc[start:end].copy()

table_cols = []
for col, label in [("at", "Tanggal"), ("userName", "Username"),
                    ("content", "Ulasan"), ("score", "Rating"),
                    ("aspek", "Aspek"), ("sentimen_label", "Sentimen"),
                    ("sentimen_conf", "Probabilitas")]:
    if col not in df_page.columns:
        continue
    series = df_page[col]
    if col == "at":
        series = series.dt.strftime("%Y-%m-%d")
    elif col == "sentimen_conf":
        series = series.apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    elif col == "aspek":
        series = series.fillna("-").str.replace("_", " ").str.title()
    table_cols.append((label, series))

df_display = pd.DataFrame({label: s.values for label, s in table_cols})
st.dataframe(df_display, use_container_width=True, height=450,
             column_config={"Ulasan": st.column_config.TextColumn(width="large"),
                            "Tanggal": st.column_config.TextColumn(width="small"),
                            "Rating": st.column_config.NumberColumn(width="small")},
             hide_index=True)

# ── DOWNLOAD CSV ──
st.divider()
st.subheader("Download Dataset Lengkap")
csv = df_res.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"{st.session_state.get('app_id', 'ulasan')}_hasil_absa.csv",
    mime="text/csv",
    use_container_width=True,
)
