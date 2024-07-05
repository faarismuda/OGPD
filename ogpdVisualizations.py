import altair as alt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import streamlit as st

import nltk

# Setup
nltk.download("stopwords")
stop_words = list(nltk.corpus.stopwords.words("indonesian"))

def get_top_ngrams(corpus, ngram_range=(1, 1), stop_words=None, n=None):
    if corpus.empty:
        return []
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

def ogpd_visualize_data(df):
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
    st.write(
        "Visualisasi pie chart menunjukkan bahwa sebagian besar unggahan, yaitu sekitar 79,7% tidak terkait dengan promosi judi daring, sementara 20,3% sisanya memang mempromosikan judi daring. Ini menunjukkan bahwa meskipun mayoritas konten tidak mempromosikan judi, proporsi yang mempromosikan cukup signifikan dan memerlukan perhatian khusus."
    )

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
    st.write(
        "Histogram memperlihatkan distribusi panjang unggahan dalam karakter terhadap frekuensi kemunculannya. Dari data yang terlihat, pada rentang panjang unggahan yang lebih pendek, yaitu sekitar 0 hingga 50 karakter, unggahan negatif tampak memiliki frekuensi lebih tinggi dibandingkan dengan unggahan positif. Namun, ketika panjang unggahan bertambah, terjadi penurunan frekuensi untuk kedua kategori tersebut. Menariknya, penurunan ini terjadi lebih lambat pada unggahan negatif. Ini menunjukkan bahwa unggahan promosi cenderung singkat dan langsung, sedangkan unggahan penolakan lebih panjang dan deskriptif."
    )

    # Wordcloud
    st.subheader("Word Cloud")
    st.image("assets/ogpd-wordcloud.png", use_column_width=True)
    st.write(
        "Analisis word cloud menunjukkan dominasi kata “judi online”, yang menegaskan fokus dataset pada judi daring. Kata-kata seperti “main”, “situs”, “toto gelap”, “bola”, “poker”, dan “slot” sering muncul, menandakan diskusi seputar berbagai jenis permainan judi online dan situs yang menyediakannya. Kata “uang” dan “menang” berkaitan dengan iming-iming yang didapat ketika bermain judi, sementara “bandar” dan “agen” terkait dengan penyelenggara atau bisa juga memberitahukan secara langsung penyelenggara judi mana yang dipromosikan. Kata-kata seperti “kalah”, “tangkap”, “haram”, “blokir”, dan “larang” mengindikasikan risiko dan konsekuensi negatif dari judi daring."
    )