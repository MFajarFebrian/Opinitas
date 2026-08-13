"""Chart helpers for the Opinitas dashboard — Plotly + Wordcloud."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Dark theme palette
COLORS = {
    "Positif": "#2ecc71",
    "Negatif": "#e74c3c",
    "Netral": "#f39c12",
    "bg": "#0E1117",
    "card": "#1E1E2E",
    "accent": "#7C3AED",
    "text": "#E2E8F0",
    "muted": "#94A3B8",
}


def _dark_layout(**kwargs):
    base = dict(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    base.update(kwargs)
    return base


def donut_sentiment(df, title="Distribusi Sentimen"):
    """Donut chart of sentiment distribution."""
    d = df[df["sentimen_label"].isin(["Positif", "Negatif"])].copy()
    if d.empty:
        fig = go.Figure(layout=_dark_layout(title=title or "Tidak ada data"))
        fig.update_layout(annotations=[dict(text="Tidak ada data", showarrow=False)])
        return fig

    counts = d["sentimen_label"].value_counts().reset_index()
    counts.columns = ["sentimen", "jumlah"]
    total = counts["jumlah"].sum()
    counts["persen"] = counts["jumlah"] / total * 100

    fig = go.Figure(go.Pie(
        labels=counts["sentimen"],
        values=counts["jumlah"],
        hole=0.55,
        marker=dict(colors=[COLORS.get(s, "#7C3AED") for s in counts["sentimen"]]),
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>Persen: %{percent}<extra></extra>",
    ))
    fig.update_layout(_dark_layout(title=title, showlegend=False, height=350))
    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:11px;color:#94A3B8'>Ulasan</span>",
        x=0.5, y=0.5, font=dict(size=18, color=COLORS["text"]), showarrow=False,
    )
    return fig


def bar_aspect(df, title="Distribusi Aspek"):
    """Horizontal bar chart of aspect distribution."""
    d = df[df["aspek"].notna()].copy()
    if d.empty:
        fig = go.Figure(layout=_dark_layout(title=title or "Tidak ada data"))
        return fig

    counts = d["aspek"].value_counts().reset_index()
    counts.columns = ["aspek", "jumlah"]
    counts["persen"] = counts["jumlah"] / counts["jumlah"].sum() * 100

    fig = px.bar(
        counts, y="aspek", x="jumlah", orientation="h",
        color="jumlah", color_continuous_scale=["#7C3AED", "#2ecc71"],
        template="plotly_dark",
    )
    fig.update_layout(
        _dark_layout(title=title, height=350, yaxis=dict(categoryorder="total ascending")),
        coloraxis_showscale=False,
        xaxis_title="Jumlah Ulasan",
        yaxis_title="",
        margin=dict(l=150, r=20, t=50, b=40),
    )
    fig.update_traces(
        texttemplate="%{x}",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Jumlah: %{x}<extra></extra>",
    )
    # Fix trace name
    fig.data[0].name = "Jumlah"
    return fig


def trend_chart(df, title="Tren Sentimen per Bulan"):
    """Line chart of sentiment trend over time."""
    d = df[df["sentimen_label"].isin(["Positif", "Negatif"])].copy()
    if d.empty or "at" not in d.columns:
        fig = go.Figure(layout=_dark_layout(title=title or "Tidak ada data"))
        return fig

    d["at"] = pd.to_datetime(d["at"], errors="coerce")
    d = d.dropna(subset=["at"])
    if d.empty:
        fig = go.Figure(layout=_dark_layout(title=title or "Tidak ada data"))
        return fig

    d["period"] = d["at"].dt.to_period("M").astype(str)
    monthly = d.groupby(["period", "sentimen_label"]).size().reset_index(name="jumlah")
    monthly = monthly.sort_values("period")

    fig = go.Figure()
    for sentiment in ["Positif", "Negatif"]:
        sub = monthly[monthly["sentimen_label"] == sentiment]
        fig.add_trace(go.Scatter(
            x=sub["period"], y=sub["jumlah"],
            mode="lines+markers",
            name=sentiment,
            line=dict(color=COLORS[sentiment], width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{sentiment}</b><br>%{{x}}<br>Jumlah: %{{y}}<extra></extra>",
        ))

    fig.update_layout(_dark_layout(title=title, height=400, xaxis_title="", yaxis_title="Jumlah Ulasan"))
    return fig
