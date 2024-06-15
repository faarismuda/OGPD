import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from PIL import Image
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from streamlit_navigation_bar import st_navbar
import altair as alt
import datetime
import joblib  # pastikan joblib diimpor dari modul yang benar
import os
import time

# Setup
nltk.download("stopwords")
stop_words = list(stopwords.words("indonesian"))  # Convert to list


@st.cache_data
def load_data():
    return pd.read_csv("labeled_final.csv")


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


def get_top_ngrams(corpus, ngram_range=(1, 1), n=None):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


# Load Data
data = load_data()

# Navigation
pages = ["OGPD", "Explorer", "About"]
styles = {
    "nav": {
        "background-color": "rgb(116, 185, 255)",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(255, 255, 255)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(45, 52, 54, 0.25)",
    },
    "hover": {
        "background-color": "rgba(223, 230, 233, 0.35)",
    },
}
page = st_navbar(pages, styles=styles)

# Display content based on selected page
if page == "OGPD":
    st.title("OGPD")
    st.write("Penjelasan singkat aplikasi.")

    # Pie Chart untuk Distribusi Label
    st.subheader("Distribusi Label")
    label_counts = data["label"].value_counts().reset_index()
    label_counts.columns = ["label", "count"]

    # Menghitung persentase untuk setiap label dan membatasi desimalnya
    total_count = label_counts["count"].sum()
    label_counts["percentage"] = ((label_counts["count"] / total_count) * 100).round(2)

    # Membuat pie chart dengan Altair untuk distribusi label
    pie_chart = (
        alt.Chart(label_counts)
        .mark_arc()
        .encode(
            theta=alt.Theta(field="percentage", type="quantitative"),
            color=alt.Color(field="label", type="nominal"),
            tooltip=[
                "label",
                "count",
                alt.Tooltip(field="percentage", type="quantitative", format=".2f"),
            ],
        )
    )

    st.altair_chart(pie_chart, use_container_width=True)
    st.write(
        "Sebagian besar unggahan dari 2019 hingga 2023 dengan kata kunci “judi” tidak terkait dengan promosi judi daring. Meskipun demikian, proporsi 27,9% yang mempromosikan judi online masih cukup besar, menandakan perlunya pengawasan konten tersebut."
    )

    # Wordcloud
    st.subheader("Word Cloud")
    all_texts = data["text"]
    all_text = " ".join(all_texts.astype(str).tolist())
    circle_mask = np.array(Image.open("mask.png"))
    wordcloud = generate_wordcloud(all_text, circle_mask)
    st.image(wordcloud.to_array(), use_column_width=True)
    st.write(
        "Analisis word cloud menunjukkan dominasi kata “judi” dan “online”, yang menegaskan fokus dataset pada judi daring. Kata-kata seperti “main”, “situs”, “toto gelap”, “bola”, “poker”, dan “slot” sering muncul, menandakan diskusi seputar berbagai jenis permainan judi online dan situs yang menyediakannya. Kata “uang” dan “menang” berkaitan dengan insentif finansial judi, sementara “bandar” dan “agen” terkait dengan penyelenggara. Kata-kata seperti “kalah”, “tangkap”, “haram”, “blokir”, dan “larang” mengindikasikan risiko dan konsekuensi negatif dari judi daring."
    )

    # Frequency of Words
    st.subheader("Frekuensi Kata")
    positive_texts = data[data["label"] == "Positif"]["text"]
    negative_texts = data[data["label"] == "Negatif"]["text"]
    top_positive_words = get_top_ngrams(positive_texts, ngram_range=(1, 1), n=20)
    top_negative_words = get_top_ngrams(negative_texts, ngram_range=(1, 1), n=20)
    positive_df = pd.DataFrame(
        top_positive_words, columns=["word", "frequency"]
    ).sort_values(by="frequency", ascending=False)
    negative_df = pd.DataFrame(
        top_negative_words, columns=["word", "frequency"]
    ).sort_values(by="frequency", ascending=False)

    # Create bar charts with Altair
    positive_chart = (
        alt.Chart(positive_df)
        .mark_bar()
        .encode(x=alt.X("word", sort=None), y="frequency")
        .properties(title="Top Positive Words")
    )

    negative_chart = (
        alt.Chart(negative_df)
        .mark_bar()
        .encode(x=alt.X("word", sort=None), y="frequency")
        .properties(title="Top Negative Words")
    )

    st.altair_chart(positive_chart, use_container_width=True)
    st.altair_chart(negative_chart, use_container_width=True)

    # Bigram Analysis
    st.subheader("Analisis Bigram")
    top_positive_bigrams = get_top_ngrams(positive_texts, ngram_range=(2, 2), n=20)
    top_negative_bigrams = get_top_ngrams(negative_texts, ngram_range=(2, 2), n=20)
    positive_bigram_df = pd.DataFrame(
        top_positive_bigrams, columns=["bigram", "frequency"]
    ).sort_values(by="frequency", ascending=False)
    negative_bigram_df = pd.DataFrame(
        top_negative_bigrams, columns=["bigram", "frequency"]
    ).sort_values(by="frequency", ascending=False)

    # Create bigram bar charts with Altair
    positive_bigram_chart = (
        alt.Chart(positive_bigram_df)
        .mark_bar()
        .encode(x=alt.X("bigram", sort=None), y="frequency")
        .properties(title="Top Positive Bigrams")
    )

    negative_bigram_chart = (
        alt.Chart(negative_bigram_df)
        .mark_bar()
        .encode(x=alt.X("bigram", sort=None), y="frequency")
        .properties(title="Top Negative Bigrams")
    )

    st.altair_chart(positive_bigram_chart, use_container_width=True)
    st.altair_chart(negative_bigram_chart, use_container_width=True)

    # Histogram
    st.subheader("Distribusi Panjang Tweet")
    data["tweet_length"] = data["text"].apply(len)

    # Create histogram with Altair
    hist_chart = (
        alt.Chart(data)
        .transform_bin("binned_length", field="tweet_length", bin=alt.Bin(maxbins=50))
        .transform_aggregate(count="count()", groupby=["binned_length", "label"])
        .mark_bar()
        .encode(x="binned_length:Q", y="count:Q", color="label:N")
        .properties(width=600, height=400, title="Distribusi Panjang Tweet")
    )

    st.altair_chart(hist_chart, use_container_width=True)

elif page == "Explorer":
    st.title("Explorer")
    search_keyword = st.text_input("Masukkan kata kunci pencarian:", value="judi")
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    start_date = st.date_input("Tanggal mulai:", value=yesterday)
    end_date = st.date_input("Tanggal akhir:", value=today)

    if st.button("Crawl dan Prediksi"):
        # Format search keyword
        search_keyword_formatted = (
            f"{search_keyword} lang:id since:{start_date} until:{end_date}"
        )
        filename = "x_explorer.csv"
        limit = 5000
        x_auth_token = '0cbdc3f18f9ab2ad2bb8e5dc6bf36041126203c9'

        # Cek apakah file sudah ada, jika ya, tambahkan timestamp pada nama file
        if os.path.isfile(f"tweets-data/{filename}"):
            timestamp = int(time.time())
            filename = f"x_explorer_{timestamp}.csv"

        # Crawling data
        os.system(
            f'npx --yes tweet-harvest@latest -o "{filename}" -s "{search_keyword_formatted}" -l {limit} --token "{x_auth_token}"'
        )

        # Load data
        file_path = f"tweets-data/{filename}"
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except pd.errors.EmptyDataError:
            st.error("Unggahan tidak ditemukan.")
            st.stop()

        # Prediksi menggunakan model SVM
        model_path = "svm_model.pkl"  # Ganti dengan path model Anda
        model = joblib.load(model_path)
        X = df["full_text"]  # Menggunakan kolom 'full_text' untuk prediksi
        predictions = model.predict(X)

        # Simpan hasil prediksi ke CSV baru
        df["label"] = predictions
        df = df.rename(columns={"full_text": "Text", "tweet_url": "URL", "label": "Label"})

        # Tambahkan nomor urut
        df.insert(0, 'No', range(1, 1 + len(df)))

        # Mem-wrap text di kolom "Text"
        df["Text"] = df["Text"].apply(lambda x: '\n'.join(x[i:i+50] for i in range(0, len(x), 50)))

        output_filename = f"predicted_{filename}"
        df[["No", "Text", "URL", "Label"]].to_csv(output_filename, index=False)

        st.success(f"Crawling dan prediksi selesai! Hasil disimpan di {output_filename}")
        st.table(df[["No", "Text", "URL", "Label"]])  # Menggunakan st.table untuk menghilangkan index bawaan



elif page == "About":
    st.title("About")
    st.write("This is an about page.")
