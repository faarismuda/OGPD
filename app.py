import streamlit as st
import pandas as pd
import re
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
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
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
    st.title("Online Gambling Promotion Detector")
    st.write(
        "Selamat datang di Online Gambling Promotion Detector. Sistem ini dirancang untuk membantu Anda dalam mengklasifikasikan konten yang terkait dengan promosi judi daring di platform media sosial."
    )
    st.write(
        "Sistem ini menggunakan algoritma Support Vector Machine (SVM) untuk memproses dan menganalisis data yang dikumpulkan dari media sosial X. Dengan mengumpulkan dan memproses sekitar 38.000 unggahan yang telah diberi label positif untuk unggahan yang mengandung promosi dan negatif untuk unggahan yang tidak mengandung promosi, dibuat suatu model yang kemudian dilatih agar sistem ini mampu mendeteksi apakah suatu unggahan di media sosial mempromosikan judi daring atau tidak."
    )

    st.header("Analisis Data")
    st.write(
        "Dari analisis terhadap sekitar 38.000 unggahan yang telah dikumpulkan dalam rentang awal tahun 2019 hingga akhir tahun 2023, berikut adalah beberapa temuan yang dapat dilihat melalui visualisasi data:"
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
    st.subheader("Analisis Bigram")
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

    if explorer_option == "Facebook":
        st.header("Facebook Explorer")
    elif explorer_option == "Instagram":
        st.header("Instagram Explorer")
    elif explorer_option == "X":
        st.header("X Explorer")
        st.write(
            "Aplikasi ini memungkinkan Anda untuk melakukan crawling unggahan X dan mengklasifikasikan menggunakan model SVM."
        )
        col1, col2 = st.columns(2)
        with col1:
            search_keyword = st.text_input(
                "Masukkan kata kunci pencarian:", value="judi"
            )
        with col2:
            limit = st.number_input(
                "Masukkan batas maksimum unggahan:", min_value=20, value=200
            )
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        tomorrow = today + datetime.timedelta(days=1)
        col3, col4 = st.columns(2)
        with col3:
            start_date = st.date_input("Tanggal mulai:", value=today)
        with col4:
            end_date = st.date_input("Tanggal akhir:", value=tomorrow)
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
            if st.button("Crawl dan Prediksi"):
                # Format search keyword
                search_keyword_formatted = (
                    f"{search_keyword} lang:id since:{start_date} until:{end_date}"
                )
                
                x_auth_token = os.getenv("X_AUTH_TOKEN")

                timestamp = int(time.time())
                
                filename = f"x_explorer_{timestamp}.csv"

                # Crawling data
                with st.spinner("Crawling data..."):
                    os.system(
                        f'npx --yes tweet-harvest@latest -o "{filename}" -s "{search_keyword_formatted}" -l {limit} --token "{x_auth_token}"'
                    )

                # Load data
                file_path = f"tweets-data/{filename}"
                try:
                    df = pd.read_csv(file_path, encoding="latin1")
                except pd.errors.EmptyDataError:
                    st.error("Unggahan tidak ditemukan.")
                    st.stop()

                slang_df = pd.read_csv("Kata_Baku_Final.csv")
                slang_dict = dict(zip(slang_df.iloc[:, 0], slang_df.iloc[:, 1]))

                stopwords_df = pd.read_csv("Stopwords.csv")
                stopwords = stopwords_df.iloc[:, 0].tolist()

                def cleaning(text):
                    # Case Folding
                    text = str(text).lower()

                    # Menghapus URL
                    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)

                    # Menghapus Tag User
                    text = re.sub(r"@[^\s]+", "", text)

                    # Menghapus Hashtag
                    text = re.sub(r"#[^\s]+", "", text)

                    # Mengganti Tag HTML dengan Spasi
                    text = re.sub(r"<.*?>", " ", text)

                    # Mengganti Tanda Baca/Karakter Khusus dengan Spasi
                    text = re.sub(r"[^\w\s]", " ", text)

                    # Mengganti Karakter non-ASCII atau Karakter Unicode dengan Spasi
                    text = re.sub(r"[^\x00-\x7F]+", " ", text)

                    # Menghapus Angka
                    text = re.sub(r"\d+", "", text)

                    # Menghapus Kata "amp"
                    text = re.sub(r"\bamp\b", "", text)

                    # Mengganti Line Baru dengan Spasi
                    text = re.sub(r"\n", " ", text)

                    # Menghapus Single Char
                    text = re.sub(r"\b[a-zA-Z]\b", " ", text)

                    # Mengganti semua urutan karakter yang berulang lebih dari dua kali dalam string text menjadi dua pengulangan karakter. Haiii > Haii
                    text = re.sub(r"(.)\1+", r"\1\1", text)

                    # Menghapus kata-kata yang berulang. Halo halo apa kabar? > Halo apa kabar?
                    text = re.sub(
                        r"\b(\w+)(?:\W\1\b)+", r"\1", text, flags=re.IGNORECASE
                    )

                    # Menghapus Spasi Ekstra
                    text = re.sub(r"\s+", " ", text)

                    # Menghapus Whitespace di Awal dan Akhir Teks
                    text = text.strip()

                    return text

                def normalize_slang(text):
                    return " ".join(slang_dict.get(word, word) for word in text.split())

                def remove_stopwords(text):
                    return " ".join(
                        word for word in text.split() if word not in stopwords
                    )

                factory = StemmerFactory()
                stemmer = factory.create_stemmer()

                def stemming(text):
                    return " ".join(stemmer.stem(word) for word in text.split())

                df["Processed"] = df["full_text"].apply(cleaning)
                df["Processed"] = df["Processed"].apply(normalize_slang)
                df["Processed"] = df["Processed"].apply(remove_stopwords)
                df["Processed"] = df["Processed"].apply(stemming)

                df.to_csv(file_path, index=False)

                # Prediksi menggunakan model SVM
                model_path = "svm_model.pkl"  # Ganti dengan path model Anda
                try:
                    model = joblib.load(model_path)
                except FileNotFoundError:
                    st.error("Model tidak ditemukan.")
                    st.stop()

                X = df["Processed"]  # Menggunakan kolom 'processed' untuk klasifikasi
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
                                field="Percentage", type="quantitative", format=".2f"
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
                st.subheader("Analisis Bigram")
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
                        "Panjang Karakter", field="Text Length", bin=alt.Bin(maxbins=50)
                    )
                    .transform_aggregate(
                        Frekuensi="count()", groupby=["Panjang Karakter", "Label"]
                    )
                    .mark_bar()
                    .encode(x="Panjang Karakter:Q", y="Frekuensi:Q", color="Label:N")
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
    st.write("This is an about page.")
