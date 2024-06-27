import altair as alt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

import streamlit as st

import nltk

# Setup
nltk.download("stopwords")
stop_words = list(nltk.corpus.stopwords.words("indonesian"))


@st.cache_data
def generate_wordcloud(all_text, circle_mask):
    return WordCloud(
        width=800,
        height=800,
        background_color="white",
        mask=circle_mask,
        contour_width=1,
        contour_color="steelblue",
        min_font_size=10,
        colormap="viridis",
        prefer_horizontal=1.0,
        scale=3,
        max_words=50000000,
        relative_scaling=0.5,
        normalize_plurals=False,
        stopwords=stop_words,
    ).generate(all_text)


def get_top_ngrams(corpus, ngram_range=(1, 1), stop_words=None, n=None):
    if corpus.empty:
        return []
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def visualize_data(df):
    # Pie Chart untuk Distribusi Label
    st.subheader("Distribusi Label")
    label_counts = df["Label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]
    total_count = label_counts["Count"].sum()
    label_counts["Percentage"] = ((label_counts["Count"] / total_count) * 100).round(2)

    pie_chart = (
        alt.Chart(label_counts)
        .mark_arc()
        .encode(
            theta=alt.Theta(field="Percentage", type="quantitative"),
            color=alt.Color(field="Label", type="nominal"),
            tooltip=[
                "Label",
                "Count",
                alt.Tooltip(field="Percentage", type="quantitative", format=".2f"),
            ],
        )
    )
    with st.spinner("Creating pie chart..."):
        st.altair_chart(pie_chart, use_container_width=True)

    # Frekuensi of Words
    st.subheader("Frekuensi Kata")
    positive_texts = df[df["Label"] == "Positif"]["Processed"]
    negative_texts = df[df["Label"] == "Negatif"]["Processed"]
    top_positive_words = get_top_ngrams(positive_texts, ngram_range=(1, 1), n=30)
    top_negative_words = get_top_ngrams(negative_texts, ngram_range=(1, 1), n=30)
    positive_df = pd.DataFrame(
        top_positive_words, columns=["Kata", "Frekuensi"]
    ).sort_values(by="Frekuensi", ascending=False)
    negative_df = pd.DataFrame(
        top_negative_words, columns=["Kata", "Frekuensi"]
    ).sort_values(by="Frekuensi", ascending=False)

    positive_chart = (
        alt.Chart(positive_df)
        .mark_bar()
        .encode(x=alt.X("Kata", sort=None), y="Frekuensi")
        .properties(title="Kata-Kata Teratas yang Dilabeli Positif")
    )
    negative_chart = (
        alt.Chart(negative_df)
        .mark_bar()
        .encode(x=alt.X("Kata", sort=None), y="Frekuensi")
        .properties(title="Kata-Kata Teratas yang Dilabeli Negatif")
    )
    with st.spinner("Creating bar chart..."):
        st.altair_chart(positive_chart, use_container_width=True)
        st.altair_chart(negative_chart, use_container_width=True)

    # Bigram Analysis
    st.subheader("Frekuensi Bigram")
    top_positive_bigrams = get_top_ngrams(positive_texts, ngram_range=(2, 2), n=30)
    top_negative_bigrams = get_top_ngrams(negative_texts, ngram_range=(2, 2), n=30)
    positive_bigram_df = pd.DataFrame(
        top_positive_bigrams, columns=["Bigram", "Frekuensi"]
    ).sort_values(by="Frekuensi", ascending=False)
    negative_bigram_df = pd.DataFrame(
        top_negative_bigrams, columns=["Bigram", "Frekuensi"]
    ).sort_values(by="Frekuensi", ascending=False)

    positive_bigram_chart = (
        alt.Chart(positive_bigram_df)
        .mark_bar()
        .encode(x=alt.X("Bigram", sort=None), y="Frekuensi")
        .properties(title="Bigram-Bigram Teratas yang Dilabeli Positif")
    )
    negative_bigram_chart = (
        alt.Chart(negative_bigram_df)
        .mark_bar()
        .encode(x=alt.X("Bigram", sort=None), y="Frekuensi")
        .properties(title="Bigram-Bigram Teratas yang Dilabeli Negatif")
    )
    with st.spinner("Creating bar chart..."):
        st.altair_chart(positive_bigram_chart, use_container_width=True)
        st.altair_chart(negative_bigram_chart, use_container_width=True)

    # Histogram
    st.subheader("Distribusi Panjang Karakter")
    df["Text Length"] = df["Processed"].apply(len)
    hist_chart = (
        alt.Chart(df)
        .transform_bin(
            "Panjang Karakter",
            field="Text Length",
            bin=alt.Bin(maxbins=50),
        )
        .transform_aggregate(Frekuensi="count()", groupby=["Panjang Karakter", "Label"])
        .mark_bar()
        .encode(x="Panjang Karakter:Q", y="Frekuensi:Q", color="Label:N")
        .properties(width=600, height=400, title="Histogram")
    )
    with st.spinner("Creating histogram..."):
        st.altair_chart(hist_chart, use_container_width=True)

    # Wordcloud
    st.subheader("Word Cloud")
    all_texts = df["Processed"]
    all_text = " ".join(all_texts.astype(str).tolist())
    circle_mask = np.array(Image.open("assets/mask.png"))
    wordcloud = generate_wordcloud(all_text, circle_mask)
    with st.spinner("Creating word cloud..."):
        st.image(wordcloud.to_array(), use_column_width=True)
