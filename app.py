from datetime import datetime, timedelta
import os
import re
import subprocess

import altair as alt
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from apify_client import ApifyClient
from wordcloud import WordCloud

import streamlit as st
from streamlit_navigation_bar import st_navbar

import nltk
from nltk.corpus import stopwords

from dotenv import load_dotenv

# Setup
nltk.download("stopwords")
stop_words = list(nltk.corpus.stopwords.words("indonesian"))

load_dotenv()


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


def get_top_ngrams(corpus, ngram_range=(1, 1), stop_words=None, n=None):
    if corpus.empty:
        return []
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

today = datetime.today()
yesterday = today - timedelta(days=1)
tomorrow = today + timedelta(days=1)

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
    st.title("Online Gambling Promotion Detector")
    st.write(
        "Selamat datang di Online Gambling Promotion Detector. Sistem ini dirancang untuk membantu Anda dalam mengklasifikasikan konten yang terkait dengan promosi judi daring di platform media sosial."
    )
    st.write(
        "Sistem ini menggunakan algoritma Support Vector Machine (SVM) untuk memproses dan menganalisis data yang dikumpulkan dari media sosial X. Dengan mengumpulkan dan memproses sekitar puluhan ribu unggahan yang telah diberi label positif untuk unggahan yang mengandung promosi dan negatif untuk unggahan yang tidak mengandung promosi, dibuat suatu model yang kemudian dilatih agar sistem ini mampu mendeteksi apakah suatu unggahan di media sosial mempromosikan judi daring atau tidak."
    )

    st.header("Analisis Data")
    st.write(
        "Dari analisis terhadap sekitar 38.000 unggahan dengan kata kunci “judi” yang telah dikumpulkan dalam rentang awal tahun 2019 hingga akhir tahun 2023, berikut adalah beberapa temuan yang dapat dilihat melalui visualisasi data:"
    )

    # Pie Chart untuk Distribusi Label
    st.subheader("Distribusi Label")
    label_counts = data["label"].value_counts().reset_index()
    label_counts.columns = ["label", "Count"]

    # Menghitung persentase untuk setiap label dan membatasi desimalnya
    total_count = label_counts["Count"].sum()
    label_counts["Percentage"] = ((label_counts["Count"] / total_count) * 100).round(2)

    # Membuat pie chart dengan Altair untuk distribusi label
    pie_chart = (
        alt.Chart(label_counts)
        .mark_arc()
        .encode(
            theta=alt.Theta(field="Percentage", type="quantitative"),
            color=alt.Color(field="label", type="nominal"),
            tooltip=[
                "label",
                "Count",
                alt.Tooltip(field="Percentage", type="quantitative", format=".2f"),
            ],
        )
    )

    st.altair_chart(pie_chart, use_container_width=True)
    st.write(
        "Visualisasi pie chart menunjukkan bahwa sebagian besar unggahan, yaitu sekitar 72,1% tidak terkait dengan promosi judi daring, sementara 27,9% sisanya memang mempromosikan judi daring. Ini menunjukkan bahwa meskipun mayoritas konten tidak mempromosikan judi, proporsi yang mempromosikan cukup signifikan dan memerlukan perhatian khusus."
    )

    # Frekuensi of Words
    st.subheader("Frekuensi Kata")
    positive_texts = data[data["label"] == "Positif"]["text"]
    negative_texts = data[data["label"] == "Negatif"]["text"]
    top_positive_words = get_top_ngrams(positive_texts, ngram_range=(1, 1), n=30)
    top_negative_words = get_top_ngrams(negative_texts, ngram_range=(1, 1), n=30)
    positive_df = pd.DataFrame(
        top_positive_words, columns=["Kata", "Frekuensi"]
    ).sort_values(by="Frekuensi", ascending=False)
    negative_df = pd.DataFrame(
        top_negative_words, columns=["Kata", "Frekuensi"]
    ).sort_values(by="Frekuensi", ascending=False)

    # Create bar charts with Altair
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

    # Create bigram bar charts with Altair
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

    st.altair_chart(positive_bigram_chart, use_container_width=True)
    st.altair_chart(negative_bigram_chart, use_container_width=True)

    # Histogram
    st.subheader("Distribusi Panjang Karakter")
    data["Text Length"] = data["text"].apply(len)

    # Create histogram with Altair
    hist_chart = (
        alt.Chart(data)
        .transform_bin("Panjang Karakter", field="Text Length", bin=alt.Bin(maxbins=50))
        .transform_aggregate(Frekuensi="count()", groupby=["Panjang Karakter", "label"])
        .mark_bar()
        .encode(x="Panjang Karakter:Q", y="Frekuensi:Q", color="label:N")
        .properties(width=600, height=400, title="Histogram")
    )

    st.altair_chart(hist_chart, use_container_width=True)
    st.write(
        "Histogram memperlihatkan distribusi panjang unggahan dalam karakter terhadap frekuensi kemunculannya. Unggahan positif, ditandai biru muda, lebih sering muncul pada rentang 0-50 karakter dibandingkan unggahan negatif, yang ditandai biru tua. Frekuensi kedua kategori menurun seiring bertambahnya panjang unggahan, namun penurunan pada unggahan negatif lebih lambat. Ini menunjukkan bahwa unggahan promosi cenderung singkat dan langsung, sedangkan unggahan penolakan lebih panjang dan deskriptif."
    )

    # Wordcloud
    st.subheader("Word Cloud")
    all_texts = data["text"]
    all_text = " ".join(all_texts.astype(str).tolist())
    circle_mask = np.array(Image.open("mask.png"))
    wordcloud = generate_wordcloud(all_text, circle_mask)
    st.image(wordcloud.to_array(), use_column_width=True)
    st.write(
        "Analisis word cloud menunjukkan dominasi kata “judi online”, yang menegaskan fokus dataset pada judi daring. Kata-kata seperti “main”, “situs”, “toto gelap”, “bola”, “poker”, dan “slot” sering muncul, menandakan diskusi seputar berbagai jenis permainan judi online dan situs yang menyediakannya. Kata “uang” dan “menang” berkaitan dengan iming-iming yang didapat ketika bermain judi, sementara “bandar” dan “agen” terkait dengan penyelenggara atau bisa juga memberitahukan secara langsung penyelenggara judi mana yang dipromosikan. Kata-kata seperti “kalah”, “tangkap”, “haram”, “blokir”, dan “larang” mengindikasikan risiko dan konsekuensi negatif dari judi daring."
    )

elif page == "Explorer":
    st.title("Explorer")

    explorer_option = st.selectbox("Pilih Explorer:", ("Facebook", "Instagram", "X"))

    slang_df = pd.read_csv("Kata_Baku_Final.csv")
    slang_dict = dict(zip(slang_df.iloc[:, 0], slang_df.iloc[:, 1]))

    stopwords_df = pd.read_csv("Stopwords.csv")
    stopwords = stopwords_df.iloc[:, 0].tolist()

    def cleaning(text):
        text = str(text).lower()
        text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)
        text = re.sub(r"@[^\s]+", "", text)
        text = re.sub(r"#[^\s]+", "", text)
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\bamp\b", "", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\b[a-zA-Z]\b", " ", text)
        text = re.sub(r"(.)\1+", r"\1\1", text)
        text = re.sub(r"\b(\w+)(?:\W\1\b)+", r"\1", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def normalize_slang(text):
        return " ".join(slang_dict.get(word, word) for word in text.split())

    def remove_stopwords(text):
        return " ".join(word for word in text.split() if word not in stopwords)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemming(text):
        return " ".join(stemmer.stem(word) for word in text.split())

    if explorer_option == "Facebook":
        st.header("Facebook Explorer")

        APIFY_TOKEN = os.getenv("APIFY_TOKEN_FACEBOOK")
        client = ApifyClient(APIFY_TOKEN)

        choice = st.selectbox(
            "Pilih opsi:",
            options=[
                "Unggahan Pribadi atau Halaman",
                "Unggahan Grup",
                "Unggahan dengan Hashtag",
                "Komentar dalam Unggahan",
            ],
        )

        directory = "facebook-data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        if choice == "Unggahan Pribadi atau Halaman":
            col1, col2 = st.columns(2)
            with col1:
                username = st.text_input(
                    "Masukkan username Facebook:", value="humansofnewyork"
                )
            with col2:
                resultsLimit = st.number_input(
                    "Masukkan batas maksimum unggahan:", min_value=1, value=20
                )

            col3, col4 = st.columns(2)
            with col3:
                start_date = st.date_input(
                    "Hanya unggahan sejak tanggal:", value=yesterday
                )
            with col4:
                end_date = st.date_input(
                    "Hanya unggahan sampai tanggal:", value=tomorrow
                )

            if st.button("Crawl dan Klasifikasi"):
                # Prepare the Actor input for Facebook account
                run_input_account = {
                    "startUrls": [{"url": f"https://www.facebook.com/{username}/"}],
                    "resultsLimit": resultsLimit,
                    "onlyPostsNewerThan": start_date.strftime("%Y-%m-%d"),
                    "onlyPostsOlderThan": end_date.strftime("%Y-%m-%d"),
                }

                with st.spinner("Crawling data..."):
                    # Run the Actor and wait for it to finish
                    run_account = client.actor("KoJrdxJCTtpon81KY").call(
                        run_input=run_input_account
                    )

                # Fetch and print Actor results from the run's dataset (if there are any)
                data = []
                for item in client.dataset(
                    run_account["defaultDatasetId"]
                ).iterate_items():
                    data.append(item)
                df = pd.DataFrame(data)

                filename = f"facebook-data/fe_unggahan_{username}_{timestamp}.csv"
                df.to_csv(filename, index=False)
        elif choice == "Unggahan Grup":
            col1, col2 = st.columns(2)
            with col1:
                group_url = st.text_input(
                    "Masukkan URL grup Facebook:",
                    value="https://www.facebook.com/groups/874728723021553",
                )
            with col2:
                resultsLimit = st.number_input(
                    "Masukkan batas maksimum unggahan:", min_value=1, value=20
                )
            col3, col4 = st.columns(2)
            with col3:
                viewOption = st.selectbox(
                    "Pilih opsi tampilan:",
                    options=["CHRONOLOGICAL", "RECENT_ACTIVITY", "TOP_POSTS"],
                )
            with col4:
                onlyPostsNewerThan = st.date_input(
                    "Hanya unggahan sejak tanggal:", value=yesterday
                )

            if st.button("Crawl dan Klasifikasi"):
                # Prepare the Actor input for Facebook group
                run_input_group = {
                    "startUrls": [{"url": group_url}],
                    "resultsLimit": resultsLimit,
                    "viewOption": viewOption,
                    "onlyPostsNewerThan": onlyPostsNewerThan.strftime("%Y-%m-%d"),
                }

                with st.spinner("Crawling data..."):
                    # Run the Actor and wait for it to finish
                    run_group = client.actor("2chN8UQcH1CfxLRNE").call(
                        run_input=run_input_group
                    )

                # Fetch and print Actor results from the run's dataset (if there are any)
                data = []
                for item in client.dataset(
                    run_group["defaultDatasetId"]
                ).iterate_items():
                    data.append(item)
                df = pd.DataFrame(data)

                filename = f"facebook-data/fe_unggahan_group_{timestamp}.csv"
                df.to_csv(filename, index=False)

        elif choice == "Unggahan dengan Hashtag":
            col1, col2 = st.columns(2)
            with col1:
                hashtag = st.text_input("Masukkan hashtag:", value="judi")
            with col2:
                resultsLimit = st.number_input(
                    "Masukkan batas maksimum unggahan:", min_value=1, value=20
                )

            if st.button("Crawl dan Klasifikasi"):
                # Prepare the Actor input for Facebook hashtag
                run_input_hashtag = {
                    "keywordList": [hashtag],
                    "resultsLimit": resultsLimit,
                }

                with st.spinner("Crawling data..."):
                    # Run the Actor and wait for it to finish
                    run_hashtag = client.actor("qgl7gVMdjLUUrMI5P").call(
                        run_input=run_input_hashtag
                    )

                # Fetch and print Actor results from the run's dataset (if there are any)
                data = []
                for item in client.dataset(
                    run_hashtag["defaultDatasetId"]
                ).iterate_items():
                    data.append(item)
                df = pd.DataFrame(data)

                filename = f"facebook-data/fe_unggahan_{hashtag}_{timestamp}.csv"
                df.to_csv(filename, index=False)

        elif choice == "Komentar dalam Unggahan":
            startUrls = st.text_input(
                "Masukkan URL unggahan Facebook:",
                value="https://www.facebook.com/humansofnewyork/posts/pfbid0BbKbkisExKGSKuhee9a7i86RwRuMKFC8NSkKStB7CsM3uXJuAAfZLrkcJMXxhH4Yl",
            )

            col1, col2 = st.columns(2)
            with col1:
                resultsLimit = st.number_input(
                    "Masukkan batas maksimum komentar:", min_value=1, value=20
                )
            with col2:
                viewOption = st.selectbox(
                    "Pilih opsi tampilan:",
                    options=["RANKED_UNFILTERED", "RANKED_THREADED", "RECENT_ACTIVITY"],
                )

            if st.button("Crawl dan Klasifikasi"):
                # Prepare the Actor input for Facebook post
                run_input_post = {
                    "startUrls": [{"url": startUrls}],
                    "resultsLimit": resultsLimit,
                    "includeNestedComments": False,
                    "viewOption": viewOption,
                }

                with st.spinner("Crawling data..."):
                    # Run the Actor and wait for it to finish
                    run_post = client.actor("us5srxAYnsrkgUv2v").call(
                        run_input=run_input_post
                    )

                # Fetch and print Actor results from the run's dataset (if there are any)
                data = []
                for item in client.dataset(
                    run_post["defaultDatasetId"]
                ).iterate_items():
                    data.append(item)
                df = pd.DataFrame(data)

                filename = f"facebook-data/fe_komentar_{timestamp}.csv"
                df.to_csv(filename, index=False)

    elif explorer_option == "Instagram":
        st.header("Instagram Explorer")

        APIFY_TOKEN = os.getenv("APIFY_TOKEN_INSTAGRAM")
        client = ApifyClient(APIFY_TOKEN)

        # Add new input for choice
        choice = st.selectbox(
            "Pilih jenis crawling:",
            options=[
                "Unggahan Pribadi",
                "Unggahan dengan Hashtag",
                "Komentar dalam Unggahan",
            ],
        )

        directory = "instagram-data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        if choice == "Unggahan Pribadi":
            col1, col2 = st.columns(2)
            with col1:
                username = st.text_input(
                    "Masukkan username Instagram:", value="rpl_upi"
                )
            with col2:
                resultsLimit = st.number_input(
                    "Masukkan batas maksimum unggahan:", min_value=1, value=20
                )

            if not username:
                st.error("Username Instagram tidak boleh kosong.")
            elif resultsLimit is None:
                st.error("Batas maksimum unggahan tidak boleh kosong.")
            else:
                if st.button("Crawl dan Klasifikasi"):
                    # Prepare the Actor input for Instagram account
                    run_input_account = {
                        "username": [username],
                        "resultsLimit": resultsLimit,
                    }

                    with st.spinner("Crawling data..."):
                        # Run the Actor and wait for it to finish
                        run_account = client.actor("nH2AHrwxeTRJoN5hX").call(
                            run_input=run_input_account
                        )

                    # Fetch and print Actor results from the run's dataset (if there are any)
                    data = []
                    for item in client.dataset(
                        run_account["defaultDatasetId"]
                    ).iterate_items():
                        data.append(item)
                    df = pd.DataFrame(data)

                    filename = f"instagram-data/ie_unggahan_{username}_{timestamp}.csv"
                    df.to_csv(filename, index=False)

                    # Load data
                    file_path = filename
                    try:
                        df = pd.read_csv(file_path, encoding="latin1")
                    except pd.errors.EmptyDataError:
                        st.error("Unggahan tidak ditemukan.")
                        st.stop()

                    df["Processed"] = df["caption"].apply(cleaning)
                    df["Processed"] = df["Processed"].apply(normalize_slang)
                    df["Processed"] = df["Processed"].apply(remove_stopwords)
                    df["Processed"] = df["Processed"].apply(stemming)

                    df.to_csv(file_path, index=False)

                    # Klasifikasi menggunakan model SVM
                    model_path = "svm_model.pkl"  # Ganti dengan path model Anda
                    try:
                        model = joblib.load(model_path)
                    except FileNotFoundError:
                        st.error("Model tidak ditemukan.")
                        st.stop()

                    X = df[
                        "Processed"
                    ]  # Menggunakan kolom 'processed' untuk klasifikasi
                    predictions = model.predict(X)

                    # Simpan hasil klasifikasi ke CSV baru
                    df["label"] = predictions
                    df = df.rename(
                        columns={
                            "caption": "Text",
                            "url": "URL",
                            "label": "Label",
                        }
                    )

                    # Mengatur ulang index dimulai dari 1
                    df.index = np.arange(1, len(df) + 1)

                    output_filename = f"{filename.replace('.csv', '')}_predicted"
                    df[["Text", "URL", "Label"]].to_csv(
                        f"{output_filename}.csv", index=False
                    )

                    st.success("Crawling dan klasifikasi selesai!")
                    st.dataframe(df[["Text", "URL", "Label"]])

                    # Pie Chart untuk Distribusi Label
                    st.subheader("Distribusi Label")
                    label_counts = df["Label"].value_counts().reset_index()
                    label_counts.columns = ["Label", "Count"]

                    # Menghitung persentase untuk setiap label dan membatasi desimalnya
                    total_count = label_counts["Count"].sum()
                    label_counts["Percentage"] = (
                        (label_counts["Count"] / total_count) * 100
                    ).round(2)

                    # Membuat pie chart dengan Altair untuk distribusi label
                    pie_chart = (
                        alt.Chart(label_counts)
                        .mark_arc()
                        .encode(
                            theta=alt.Theta(field="Percentage", type="quantitative"),
                            color=alt.Color(field="Label", type="nominal"),
                            tooltip=[
                                "Label",
                                "Count",
                                alt.Tooltip(
                                    field="Percentage",
                                    type="quantitative",
                                    format=".2f",
                                ),
                            ],
                        )
                    )

                    st.altair_chart(pie_chart, use_container_width=True)

                    # Frekuensi of Words
                    st.subheader("Frekuensi Kata")
                    positive_texts = df[df["Label"] == "Positif"]["Processed"]
                    negative_texts = df[df["Label"] == "Negatif"]["Processed"]
                    top_positive_words = get_top_ngrams(
                        positive_texts, ngram_range=(1, 1), n=30
                    )
                    top_negative_words = get_top_ngrams(
                        negative_texts, ngram_range=(1, 1), n=30
                    )
                    positive_df = pd.DataFrame(
                        top_positive_words, columns=["Kata", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)
                    negative_df = pd.DataFrame(
                        top_negative_words, columns=["Kata", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)

                    # Create bar charts with Altair
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

                    st.altair_chart(positive_chart, use_container_width=True)
                    st.altair_chart(negative_chart, use_container_width=True)

                    # Bigram Analysis
                    st.subheader("Frekuensi Bigram")
                    top_positive_bigrams = get_top_ngrams(
                        positive_texts, ngram_range=(2, 2), n=30
                    )
                    top_negative_bigrams = get_top_ngrams(
                        negative_texts, ngram_range=(2, 2), n=30
                    )
                    positive_bigram_df = pd.DataFrame(
                        top_positive_bigrams, columns=["Bigram", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)
                    negative_bigram_df = pd.DataFrame(
                        top_negative_bigrams, columns=["Bigram", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)

                    # Create bigram bar charts with Altair
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

                    st.altair_chart(positive_bigram_chart, use_container_width=True)
                    st.altair_chart(negative_bigram_chart, use_container_width=True)

                    # Histogram
                    st.subheader("Distribusi Panjang Karakter")
                    df["Text Length"] = df["Processed"].apply(len)

                    # Create histogram with Altair
                    hist_chart = (
                        alt.Chart(df)
                        .transform_bin(
                            "Panjang Karakter",
                            field="Text Length",
                            bin=alt.Bin(maxbins=50),
                        )
                        .transform_aggregate(
                            Frekuensi="count()", groupby=["Panjang Karakter", "Label"]
                        )
                        .mark_bar()
                        .encode(
                            x="Panjang Karakter:Q", y="Frekuensi:Q", color="Label:N"
                        )
                        .properties(width=600, height=400, title="Histogram")
                    )

                    st.altair_chart(hist_chart, use_container_width=True)

                    # Wordcloud
                    st.subheader("Word Cloud")
                    all_texts = df["Processed"]
                    all_text = " ".join(all_texts.astype(str).tolist())
                    circle_mask = np.array(Image.open("mask.png"))
                    wordcloud = generate_wordcloud(all_text, circle_mask)
                    st.image(wordcloud.to_array(), use_column_width=True)

        elif choice == "Unggahan dengan Hashtag":
            # Add new input for hashtag
            hashtags = st.text_input("Masukkan hashtag:", value="judi")

            col1, col2 = st.columns(2)
            with col1:
                resultsLimit = st.number_input(
                    "Masukkan batas maksimum unggahan:", min_value=1, value=20
                )
            with col2:
                onlyPostsNewerThan = st.date_input(
                    "Hanya unggahan sejak tanggal:", value=yesterday
                )

            if not hashtags:
                st.error("Hashtag tidak boleh kosong.")
            elif resultsLimit is None:
                st.error("Batas maksimum unggahan tidak boleh kosong.")
            elif onlyPostsNewerThan is None:
                st.error("Tanggal tidak boleh kosong.")
            else:
                if st.button("Crawl dan Klasifikasi"):
                    # Prepare the Actor input for hashtag
                    run_input_hashtag = {
                        "hashtags": [hashtags],
                        "resultsLimit": resultsLimit,
                        "onlyPostsNewerThan": onlyPostsNewerThan.strftime("%Y-%m-%d"),
                    }

                    with st.spinner("Crawling data..."):
                        # Run the Actor and wait for it to finish
                        run_hashtag = client.actor("reGe1ST3OBgYZSsZJ").call(
                            run_input=run_input_hashtag
                        )

                    # Fetch and print Actor results from the run's dataset (if there are any)
                    data = []
                    for item in client.dataset(
                        run_hashtag["defaultDatasetId"]
                    ).iterate_items():
                        data.append(item)
                    df = pd.DataFrame(data)

                    filename = f"instagram-data/ie_unggahan_{hashtags}_{timestamp}.csv"
                    df.to_csv(filename, index=False)

                    # Load data
                    file_path = filename
                    try:
                        df = pd.read_csv(file_path, encoding="latin1")
                    except pd.errors.EmptyDataError:
                        st.error("Unggahan tidak ditemukan.")
                        st.stop()

                    df["Processed"] = df["caption"].apply(cleaning)
                    df["Processed"] = df["Processed"].apply(normalize_slang)
                    df["Processed"] = df["Processed"].apply(remove_stopwords)
                    df["Processed"] = df["Processed"].apply(stemming)

                    df.to_csv(file_path, index=False)

                    # Klasifikasi menggunakan model SVM
                    model_path = "svm_model.pkl"  # Ganti dengan path model Anda
                    try:
                        model = joblib.load(model_path)
                    except FileNotFoundError:
                        st.error("Model tidak ditemukan.")
                        st.stop()

                    X = df[
                        "Processed"
                    ]  # Menggunakan kolom 'processed' untuk klasifikasi
                    predictions = model.predict(X)

                    # Simpan hasil klasifikasi ke CSV baru
                    df["label"] = predictions
                    df = df.rename(
                        columns={
                            "caption": "Text",
                            "url": "URL",
                            "ownerUsername": "Username",
                            "label": "Label",
                        }
                    )

                    # Mengatur ulang index dimulai dari 1
                    df.index = np.arange(1, len(df) + 1)

                    output_filename = f"{filename.replace('.csv', '')}_predicted"
                    df[["Text", "URL", "Username", "Label"]].to_csv(
                        f"{output_filename}.csv", index=False
                    )

                    st.success("Crawling dan klasifikasi selesai!")
                    st.dataframe(df[["Text", "URL", "Username", "Label"]])
                    # Pie Chart untuk Distribusi Label
                    st.subheader("Distribusi Label")
                    label_counts = df["Label"].value_counts().reset_index()
                    label_counts.columns = ["Label", "Count"]

                    # Menghitung persentase untuk setiap label dan membatasi desimalnya
                    total_count = label_counts["Count"].sum()
                    label_counts["Percentage"] = (
                        (label_counts["Count"] / total_count) * 100
                    ).round(2)

                    # Membuat pie chart dengan Altair untuk distribusi label
                    pie_chart = (
                        alt.Chart(label_counts)
                        .mark_arc()
                        .encode(
                            theta=alt.Theta(field="Percentage", type="quantitative"),
                            color=alt.Color(field="Label", type="nominal"),
                            tooltip=[
                                "Label",
                                "Count",
                                alt.Tooltip(
                                    field="Percentage",
                                    type="quantitative",
                                    format=".2f",
                                ),
                            ],
                        )
                    )

                    st.altair_chart(pie_chart, use_container_width=True)

                    # Frekuensi of Words
                    st.subheader("Frekuensi Kata")
                    positive_texts = df[df["Label"] == "Positif"]["Processed"]
                    negative_texts = df[df["Label"] == "Negatif"]["Processed"]
                    top_positive_words = get_top_ngrams(
                        positive_texts, ngram_range=(1, 1), n=30
                    )
                    top_negative_words = get_top_ngrams(
                        negative_texts, ngram_range=(1, 1), n=30
                    )
                    positive_df = pd.DataFrame(
                        top_positive_words, columns=["Kata", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)
                    negative_df = pd.DataFrame(
                        top_negative_words, columns=["Kata", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)

                    # Create bar charts with Altair
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

                    st.altair_chart(positive_chart, use_container_width=True)
                    st.altair_chart(negative_chart, use_container_width=True)

                    # Bigram Analysis
                    st.subheader("Frekuensi Bigram")
                    top_positive_bigrams = get_top_ngrams(
                        positive_texts, ngram_range=(2, 2), n=30
                    )
                    top_negative_bigrams = get_top_ngrams(
                        negative_texts, ngram_range=(2, 2), n=30
                    )
                    positive_bigram_df = pd.DataFrame(
                        top_positive_bigrams, columns=["Bigram", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)
                    negative_bigram_df = pd.DataFrame(
                        top_negative_bigrams, columns=["Bigram", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)

                    # Create bigram bar charts with Altair
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

                    st.altair_chart(positive_bigram_chart, use_container_width=True)
                    st.altair_chart(negative_bigram_chart, use_container_width=True)

                    # Histogram
                    st.subheader("Distribusi Panjang Karakter")
                    df["Text Length"] = df["Processed"].apply(len)

                    # Create histogram with Altair
                    hist_chart = (
                        alt.Chart(df)
                        .transform_bin(
                            "Panjang Karakter",
                            field="Text Length",
                            bin=alt.Bin(maxbins=50),
                        )
                        .transform_aggregate(
                            Frekuensi="count()", groupby=["Panjang Karakter", "Label"]
                        )
                        .mark_bar()
                        .encode(
                            x="Panjang Karakter:Q", y="Frekuensi:Q", color="Label:N"
                        )
                        .properties(width=600, height=400, title="Histogram")
                    )

                    st.altair_chart(hist_chart, use_container_width=True)

                    # Wordcloud
                    st.subheader("Word Cloud")
                    all_texts = df["Processed"]
                    all_text = " ".join(all_texts.astype(str).tolist())
                    circle_mask = np.array(Image.open("mask.png"))
                    wordcloud = generate_wordcloud(all_text, circle_mask)
                    st.image(wordcloud.to_array(), use_column_width=True)

        elif choice == "Komentar dalam Unggahan":
            # Add new input for hashtag
            directUrls = st.text_input(
                "Masukkan link unggahan Instagram:",
                value="https://www.instagram.com/p/C8Y52QPPmib/",
            )
            directUrls = directUrls.split("?")[0]
            resultsLimit = st.number_input(
                "Masukkan batas maksimum komentar:", min_value=1, value=20
            )

            if not directUrls:
                st.error("URL tidak boleh kosong.")
            elif "?" in directUrls:
                directUrls = directUrls.split("?")[0]
            elif not directUrls.startswith("https://www.instagram.com/p/"):
                st.error(
                    "URL tidak valid. URL harus diawali dengan https://www.instagram.com/p/."
                )
            elif (
                len(directUrls.split("https://www.instagram.com/p/")[1].rstrip("/"))
                != 11
            ):
                st.error(
                    "URL tidak valid. Periksa kembali bagian setelah https://www.instagram.com/p/."
                )
            elif resultsLimit is None:
                st.error("Batas maksimum unggahan tidak boleh kosong.")
            else:
                if st.button("Crawl dan Klasifikasi"):
                    # Prepare the Actor input for hashtag
                    run_input_post = {
                        "directUrls": [directUrls],
                        "resultsLimit": resultsLimit,
                    }

                    with st.spinner("Crawling data..."):
                        # Run the Actor and wait for it to finish
                        run_comment = client.actor("SbK00X0JYCPblD2wp").call(
                            run_input=run_input_post
                        )

                    # Fetch and print Actor results from the run's dataset (if there are any)
                    data = []
                    for item in client.dataset(
                        run_comment["defaultDatasetId"]
                    ).iterate_items():
                        data.append(item)
                    df = pd.DataFrame(data)

                    filename = f"instagram-data/ie_komentar_{timestamp}.csv"
                    df.to_csv(filename, index=False)

                    # Load data
                    file_path = filename
                    try:
                        df = pd.read_csv(file_path, encoding="latin1")
                    except pd.errors.EmptyDataError:
                        st.error("Unggahan tidak ditemukan.")
                        st.stop()

                    df["Processed"] = df["text"].apply(cleaning)
                    df["Processed"] = df["Processed"].apply(normalize_slang)
                    df["Processed"] = df["Processed"].apply(remove_stopwords)
                    df["Processed"] = df["Processed"].apply(stemming)

                    df.to_csv(file_path, index=False)

                    # Klasifikasi menggunakan model SVM
                    model_path = "svm_model.pkl"  # Ganti dengan path model Anda
                    try:
                        model = joblib.load(model_path)
                    except FileNotFoundError:
                        st.error("Model tidak ditemukan.")
                        st.stop()

                    X = df[
                        "Processed"
                    ]  # Menggunakan kolom 'processed' untuk klasifikasi
                    predictions = model.predict(X)

                    # Simpan hasil klasifikasi ke CSV baru
                    df["label"] = predictions
                    df = df.rename(
                        columns={
                            "text": "Text",
                            "ownerUsername": "Username",
                            "label": "Label",
                        }
                    )

                    # Mengatur ulang index dimulai dari 1
                    df.index = np.arange(1, len(df) + 1)

                    output_filename = f"{filename.replace('.csv', '')}_predicted"
                    df[["Text", "Username", "Label"]].to_csv(
                        f"{output_filename}.csv", index=False
                    )

                    st.success("Crawling dan klasifikasi selesai!")
                    st.dataframe(df[["Text", "Username", "Label"]])

                    # Pie Chart untuk Distribusi Label
                    st.subheader("Distribusi Label")
                    label_counts = df["Label"].value_counts().reset_index()
                    label_counts.columns = ["Label", "Count"]

                    # Menghitung persentase untuk setiap label dan membatasi desimalnya
                    total_count = label_counts["Count"].sum()
                    label_counts["Percentage"] = (
                        (label_counts["Count"] / total_count) * 100
                    ).round(2)

                    # Membuat pie chart dengan Altair untuk distribusi label
                    pie_chart = (
                        alt.Chart(label_counts)
                        .mark_arc()
                        .encode(
                            theta=alt.Theta(field="Percentage", type="quantitative"),
                            color=alt.Color(field="Label", type="nominal"),
                            tooltip=[
                                "Label",
                                "Count",
                                alt.Tooltip(
                                    field="Percentage",
                                    type="quantitative",
                                    format=".2f",
                                ),
                            ],
                        )
                    )

                    st.altair_chart(pie_chart, use_container_width=True)

                    # Frekuensi of Words
                    st.subheader("Frekuensi Kata")
                    positive_texts = df[df["Label"] == "Positif"]["Processed"]
                    negative_texts = df[df["Label"] == "Negatif"]["Processed"]
                    top_positive_words = get_top_ngrams(
                        positive_texts, ngram_range=(1, 1), n=30
                    )
                    top_negative_words = get_top_ngrams(
                        negative_texts, ngram_range=(1, 1), n=30
                    )
                    positive_df = pd.DataFrame(
                        top_positive_words, columns=["Kata", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)
                    negative_df = pd.DataFrame(
                        top_negative_words, columns=["Kata", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)

                    # Create bar charts with Altair
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

                    st.altair_chart(positive_chart, use_container_width=True)
                    st.altair_chart(negative_chart, use_container_width=True)

                    # Bigram Analysis
                    st.subheader("Frekuensi Bigram")
                    top_positive_bigrams = get_top_ngrams(
                        positive_texts, ngram_range=(2, 2), n=30
                    )
                    top_negative_bigrams = get_top_ngrams(
                        negative_texts, ngram_range=(2, 2), n=30
                    )
                    positive_bigram_df = pd.DataFrame(
                        top_positive_bigrams, columns=["Bigram", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)
                    negative_bigram_df = pd.DataFrame(
                        top_negative_bigrams, columns=["Bigram", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)

                    # Create bigram bar charts with Altair
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

                    st.altair_chart(positive_bigram_chart, use_container_width=True)
                    st.altair_chart(negative_bigram_chart, use_container_width=True)

                    # Histogram
                    st.subheader("Distribusi Panjang Karakter")
                    df["Text Length"] = df["Processed"].apply(len)

                    # Create histogram with Altair
                    hist_chart = (
                        alt.Chart(df)
                        .transform_bin(
                            "Panjang Karakter",
                            field="Text Length",
                            bin=alt.Bin(maxbins=50),
                        )
                        .transform_aggregate(
                            Frekuensi="count()", groupby=["Panjang Karakter", "Label"]
                        )
                        .mark_bar()
                        .encode(
                            x="Panjang Karakter:Q", y="Frekuensi:Q", color="Label:N"
                        )
                        .properties(width=600, height=400, title="Histogram")
                    )

                    st.altair_chart(hist_chart, use_container_width=True)

                    # Wordcloud
                    st.subheader("Word Cloud")
                    all_texts = df["Processed"]
                    all_text = " ".join(all_texts.astype(str).tolist())
                    circle_mask = np.array(Image.open("mask.png"))
                    wordcloud = generate_wordcloud(all_text, circle_mask)
                    st.image(wordcloud.to_array(), use_column_width=True)
    elif explorer_option == "X":
        st.header("X Explorer")
        st.write(
            "Aplikasi ini memungkinkan Anda untuk melakukan crawling unggahan X dan mengklasifikasikannya menggunakan model SVM."
        )

        x_auth_token = os.getenv("X_AUTH_TOKEN")

        x_option = st.selectbox(
            "Pilih opsi:",
            ("Unggahan Pribadi", "Pencari Unggahan"),
        )

        if x_option == "Unggahan Pribadi":
            # Tambahkan kode untuk menangani unggahan pribadi di sini
            col1, col2 = st.columns(2)
            with col1:
                search_keyword = st.text_input(
                    "Masukkan username X:", value="KomisiJudi"
                )
            with col2:
                limit = st.number_input(
                    "Masukkan batas maksimum unggahan:", min_value=20, value=200
                )

            col3, col4 = st.columns(2)
            with col3:
                start_date = st.date_input("Hanya unggahan sejak tanggal:", value=today)
            with col4:
                end_date = st.date_input(
                    "Hanya unggahan sampai tanggal:", value=tomorrow
                )

            if not search_keyword:
                st.error("Username X tidak boleh kosong.")
            elif limit is None:
                st.error("Batas maksimum unggahan tidak boleh kosong.")
            elif start_date is None or end_date is None:
                st.error("Tanggal tidak boleh kosong.")
            elif start_date > end_date:
                st.error("Silakan masukkan rentang tanggal yang valid.")
            else:
                if st.button("Crawl dan Klasifikasi"):
                    # Format search keyword
                    search_keyword_formatted = (
                        f"from:{search_keyword} since:{start_date} until:{end_date}"
                    )

                    filename = f"xe_{search_keyword}_{timestamp}.csv"

                    with st.spinner("Crawling data..."):
                        process = subprocess.Popen(
                            f'npx --yes tweet-harvest@latest -o "{filename}" -s "{search_keyword_formatted}" -l {limit} --token "{x_auth_token}"',
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                        )

                        while True:
                            output = process.stdout.readline()
                            if output == "" and process.poll() is not None:
                                break
                            if output:
                                line = output.strip().decode("utf-8")
                                print(line)
                                if (
                                    "No more tweets found" in line
                                    or "done scrolling" in line
                                ):
                                    print(
                                        "No more tweets found or done scrolling, stopping..."
                                    )
                                    process.terminate()
                                    break
                        rc = process.poll()

                    # Load data
                    file_path = f"tweets-data/{filename}"
                    try:
                        df = pd.read_csv(file_path, encoding="latin1")
                    except pd.errors.EmptyDataError:
                        st.error("Unggahan tidak ditemukan.")
                        st.stop()

                    df["Processed"] = df["full_text"].apply(cleaning)
                    df["Processed"] = df["Processed"].apply(normalize_slang)
                    df["Processed"] = df["Processed"].apply(remove_stopwords)
                    df["Processed"] = df["Processed"].apply(stemming)

                    df.to_csv(file_path, index=False)

                    # Klasifikasi menggunakan model SVM
                    model_path = "svm_model.pkl"  # Ganti dengan path model Anda
                    try:
                        model = joblib.load(model_path)
                    except FileNotFoundError:
                        st.error("Model tidak ditemukan.")
                        st.stop()

                    X = df[
                        "Processed"
                    ]  # Menggunakan kolom 'processed' untuk klasifikasi
                    predictions = model.predict(X)

                    # Simpan hasil klasifikasi ke CSV baru
                    df["label"] = predictions
                    df = df.rename(
                        columns={
                            "full_text": "Text",
                            "tweet_url": "URL",
                            "username": "Username",
                            "label": "Label",
                        }
                    )

                    # Mengatur ulang index dimulai dari 1
                    df.index = np.arange(1, len(df) + 1)

                    output_filename = f"{filename.replace('.csv', '')}_predicted"
                    df[["Text", "URL", "Username", "Label"]].to_csv(
                        f"tweets-data/{output_filename}.csv", index=False
                    )

                    st.success("Crawling dan klasifikasi selesai!")
                    st.dataframe(df[["Text", "URL", "Username", "Label"]])

                    # Pie Chart untuk Distribusi Label
                    st.subheader("Distribusi Label")
                    label_counts = df["Label"].value_counts().reset_index()
                    label_counts.columns = ["Label", "Count"]

                    # Menghitung persentase untuk setiap label dan membatasi desimalnya
                    total_count = label_counts["Count"].sum()
                    label_counts["Percentage"] = (
                        (label_counts["Count"] / total_count) * 100
                    ).round(2)

                    # Membuat pie chart dengan Altair untuk distribusi label
                    pie_chart = (
                        alt.Chart(label_counts)
                        .mark_arc()
                        .encode(
                            theta=alt.Theta(field="Percentage", type="quantitative"),
                            color=alt.Color(field="Label", type="nominal"),
                            tooltip=[
                                "Label",
                                "Count",
                                alt.Tooltip(
                                    field="Percentage",
                                    type="quantitative",
                                    format=".2f",
                                ),
                            ],
                        )
                    )

                    st.altair_chart(pie_chart, use_container_width=True)

                    # Frekuensi of Words
                    st.subheader("Frekuensi Kata")
                    positive_texts = df[df["Label"] == "Positif"]["Processed"]
                    negative_texts = df[df["Label"] == "Negatif"]["Processed"]
                    top_positive_words = get_top_ngrams(
                        positive_texts, ngram_range=(1, 1), n=30
                    )
                    top_negative_words = get_top_ngrams(
                        negative_texts, ngram_range=(1, 1), n=30
                    )
                    positive_df = pd.DataFrame(
                        top_positive_words, columns=["Kata", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)
                    negative_df = pd.DataFrame(
                        top_negative_words, columns=["Kata", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)

                    # Create bar charts with Altair
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

                    st.altair_chart(positive_chart, use_container_width=True)
                    st.altair_chart(negative_chart, use_container_width=True)

                    # Bigram Analysis
                    st.subheader("Frekuensi Bigram")
                    top_positive_bigrams = get_top_ngrams(
                        positive_texts, ngram_range=(2, 2), n=30
                    )
                    top_negative_bigrams = get_top_ngrams(
                        negative_texts, ngram_range=(2, 2), n=30
                    )
                    positive_bigram_df = pd.DataFrame(
                        top_positive_bigrams, columns=["Bigram", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)
                    negative_bigram_df = pd.DataFrame(
                        top_negative_bigrams, columns=["Bigram", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)

                    # Create bigram bar charts with Altair
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

                    st.altair_chart(positive_bigram_chart, use_container_width=True)
                    st.altair_chart(negative_bigram_chart, use_container_width=True)

                    # Histogram
                    st.subheader("Distribusi Panjang Karakter")
                    df["Text Length"] = df["Processed"].apply(len)

                    # Create histogram with Altair
                    hist_chart = (
                        alt.Chart(df)
                        .transform_bin(
                            "Panjang Karakter",
                            field="Text Length",
                            bin=alt.Bin(maxbins=50),
                        )
                        .transform_aggregate(
                            Frekuensi="count()", groupby=["Panjang Karakter", "Label"]
                        )
                        .mark_bar()
                        .encode(
                            x="Panjang Karakter:Q", y="Frekuensi:Q", color="Label:N"
                        )
                        .properties(width=600, height=400, title="Histogram")
                    )

                    st.altair_chart(hist_chart, use_container_width=True)

                    # Wordcloud
                    st.subheader("Word Cloud")
                    all_texts = df["Processed"]
                    all_text = " ".join(all_texts.astype(str).tolist())
                    circle_mask = np.array(Image.open("mask.png"))
                    wordcloud = generate_wordcloud(all_text, circle_mask)
                    st.image(wordcloud.to_array(), use_column_width=True)
        elif x_option == "Pencari Unggahan":
            # Tambahkan kode untuk menangani pencarian unggahan di sini
            col1, col2 = st.columns(2)
            with col1:
                search_keyword = st.text_input(
                    "Masukkan kata kunci pencarian:", value="judi"
                )
            with col2:
                limit = st.number_input(
                    "Masukkan batas maksimum unggahan:", min_value=20, value=200
                )

            col3, col4 = st.columns(2)
            with col3:
                start_date = st.date_input("Hanya unggahan sejak tanggal:", value=today)
            with col4:
                end_date = st.date_input(
                    "Hanya unggahan sampai tanggal:", value=tomorrow
                )
            if not search_keyword:
                st.error("Kata kunci pencarian tidak boleh kosong.")
            elif limit is None:
                st.error("Batas maksimum unggahan tidak boleh kosong.")
            elif start_date is None:
                st.error("Tanggal mulai tidak boleh kosong.")
            elif end_date is None:
                st.error("Tanggal akhir tidak boleh kosong.")
            elif start_date > end_date:
                st.error("Tanggal mulai tidak boleh lebih besar dari tanggal akhir.")
            else:
                if st.button("Crawl dan Klasifikasi"):
                    # Format search keyword
                    search_keyword_formatted = (
                        f"{search_keyword} lang:id since:{start_date} until:{end_date}"
                    )

                    filename = f"xe_{search_keyword}_{timestamp}.csv"

                    # Crawling data
                    with st.spinner("Crawling data..."):
                        process = subprocess.Popen(
                            f'npx --yes tweet-harvest@latest -o "{filename}" -s "{search_keyword_formatted}" -l {limit} --token "{x_auth_token}"',
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                        )

                        while True:
                            output = process.stdout.readline()
                            if output == "" and process.poll() is not None:
                                break
                            if output:
                                line = output.strip().decode("utf-8")
                                print(line)
                                if (
                                    "No more tweets found" in line
                                    or "done scrolling" in line
                                ):
                                    print(
                                        "No more tweets found or done scrolling, stopping..."
                                    )
                                    process.terminate()
                                    break
                        rc = process.poll()

                    # Load data
                    file_path = f"tweets-data/{filename}"
                    try:
                        df = pd.read_csv(file_path, encoding="latin1")
                    except pd.errors.EmptyDataError:
                        st.error("Unggahan tidak ditemukan.")
                        st.stop()

                    df["Processed"] = df["full_text"].apply(cleaning)
                    df["Processed"] = df["Processed"].apply(normalize_slang)
                    df["Processed"] = df["Processed"].apply(remove_stopwords)
                    df["Processed"] = df["Processed"].apply(stemming)

                    df.to_csv(file_path, index=False)

                    # Klasifikasi menggunakan model SVM
                    model_path = "svm_model.pkl"  # Ganti dengan path model Anda
                    try:
                        model = joblib.load(model_path)
                    except FileNotFoundError:
                        st.error("Model tidak ditemukan.")
                        st.stop()

                    X = df[
                        "Processed"
                    ]  # Menggunakan kolom 'processed' untuk klasifikasi
                    predictions = model.predict(X)

                    # Simpan hasil klasifikasi ke CSV baru
                    df["label"] = predictions
                    df = df.rename(
                        columns={
                            "full_text": "Text",
                            "tweet_url": "URL",
                            "username": "Username",
                            "label": "Label",
                        }
                    )

                    # Mengatur ulang index dimulai dari 1
                    df.index = np.arange(1, len(df) + 1)

                    output_filename = f"{filename.replace('.csv', '')}_predicted"
                    df[["Text", "URL", "Username", "Label"]].to_csv(
                        f"tweets-data/{output_filename}.csv", index=False
                    )

                    st.success("Crawling dan klasifikasi selesai!")
                    st.dataframe(df[["Text", "URL", "Username", "Label"]])

                    # Pie Chart untuk Distribusi Label
                    st.subheader("Distribusi Label")
                    label_counts = df["Label"].value_counts().reset_index()
                    label_counts.columns = ["Label", "Count"]

                    # Menghitung persentase untuk setiap label dan membatasi desimalnya
                    total_count = label_counts["Count"].sum()
                    label_counts["Percentage"] = (
                        (label_counts["Count"] / total_count) * 100
                    ).round(2)

                    # Membuat pie chart dengan Altair untuk distribusi label
                    pie_chart = (
                        alt.Chart(label_counts)
                        .mark_arc()
                        .encode(
                            theta=alt.Theta(field="Percentage", type="quantitative"),
                            color=alt.Color(field="Label", type="nominal"),
                            tooltip=[
                                "Label",
                                "Count",
                                alt.Tooltip(
                                    field="Percentage",
                                    type="quantitative",
                                    format=".2f",
                                ),
                            ],
                        )
                    )

                    st.altair_chart(pie_chart, use_container_width=True)

                    # Frekuensi of Words
                    st.subheader("Frekuensi Kata")
                    positive_texts = df[df["Label"] == "Positif"]["Processed"]
                    negative_texts = df[df["Label"] == "Negatif"]["Processed"]
                    top_positive_words = get_top_ngrams(
                        positive_texts, ngram_range=(1, 1), n=30
                    )
                    top_negative_words = get_top_ngrams(
                        negative_texts, ngram_range=(1, 1), n=30
                    )
                    positive_df = pd.DataFrame(
                        top_positive_words, columns=["Kata", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)
                    negative_df = pd.DataFrame(
                        top_negative_words, columns=["Kata", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)

                    # Create bar charts with Altair
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

                    st.altair_chart(positive_chart, use_container_width=True)
                    st.altair_chart(negative_chart, use_container_width=True)

                    # Bigram Analysis
                    st.subheader("Frekuensi Bigram")
                    top_positive_bigrams = get_top_ngrams(
                        positive_texts, ngram_range=(2, 2), n=30
                    )
                    top_negative_bigrams = get_top_ngrams(
                        negative_texts, ngram_range=(2, 2), n=30
                    )
                    positive_bigram_df = pd.DataFrame(
                        top_positive_bigrams, columns=["Bigram", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)
                    negative_bigram_df = pd.DataFrame(
                        top_negative_bigrams, columns=["Bigram", "Frekuensi"]
                    ).sort_values(by="Frekuensi", ascending=False)

                    # Create bigram bar charts with Altair
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

                    st.altair_chart(positive_bigram_chart, use_container_width=True)
                    st.altair_chart(negative_bigram_chart, use_container_width=True)

                    # Histogram
                    st.subheader("Distribusi Panjang Karakter")
                    df["Text Length"] = df["Processed"].apply(len)

                    # Create histogram with Altair
                    hist_chart = (
                        alt.Chart(df)
                        .transform_bin(
                            "Panjang Karakter",
                            field="Text Length",
                            bin=alt.Bin(maxbins=50),
                        )
                        .transform_aggregate(
                            Frekuensi="count()", groupby=["Panjang Karakter", "Label"]
                        )
                        .mark_bar()
                        .encode(
                            x="Panjang Karakter:Q", y="Frekuensi:Q", color="Label:N"
                        )
                        .properties(width=600, height=400, title="Histogram")
                    )

                    st.altair_chart(hist_chart, use_container_width=True)

                    # Wordcloud
                    st.subheader("Word Cloud")
                    all_texts = df["Processed"]
                    all_text = " ".join(all_texts.astype(str).tolist())
                    circle_mask = np.array(Image.open("mask.png"))
                    wordcloud = generate_wordcloud(all_text, circle_mask)
                    st.image(wordcloud.to_array(), use_column_width=True)


elif page == "About":
    st.title("About")
    st.write(
        "This application is designed to perform social media data crawling and classification."
    )

    st.subheader("Our Mission")
    st.write(
        "Our mission is to provide a user-friendly interface for social media data analysis. We aim to make data analysis accessible to everyone, regardless of their technical skills. Additionally, we are committed to assisting authorities in combating online gambling promotions."
    )

    st.subheader("Contact Us")
    st.write("For any inquiries, please contact us at: faarismudawork@gmail.com")
