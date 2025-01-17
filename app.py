from datetime import datetime, timedelta
import time
import csv
import os
import re
import subprocess

from visualizations import visualize_data
from ogpdVisualizations import ogpd_visualize_data
from recommendations import recommendations

import joblib
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from apify_client import ApifyClient

import streamlit as st
from streamlit_navigation_bar import st_navbar

from nltk.corpus import stopwords

from dotenv import load_dotenv


load_dotenv()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
today = datetime.today()
yesterday = today - timedelta(days=1)
tomorrow = today + timedelta(days=1)

st.set_page_config(
    page_title="OGPD - Online Gambling Promotion Detector",
    page_icon="🔶",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Navigation
pages = ["OGPD", "Explorer", "Archive"]
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
        "padding": "0.3275rem 0.625rem",
        "font-family": "Product Sans, sans-serif",
        "font-size": "1.2rem",
    },
    "active": {
        "background-color": "rgba(45, 52, 54, 0.25)",
    },
    "hover": {
        "background-color": "rgba(223, 230, 233, 0.35)",
    },
}

page = st_navbar(pages, styles=styles)

with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# Display content based on selected page
if page == "OGPD":
    st.title("Online Gambling Promotion Detector")
    st.write(
        "Selamat datang di Online Gambling Promotion Detector. Sistem ini dirancang untuk membantu Anda dalam mengklasifikasikan konten yang terkait dengan promosi judi daring di platform media sosial."
    )
    st.write(
        "Sistem ini menggunakan algoritma Support Vector Machine (SVM) untuk memproses dan menganalisis data yang dikumpulkan dari media sosial X. Dengan mengumpulkan dan memproses sekitar puluhan ribu unggahan yang telah diberi label positif untuk unggahan yang mengandung promosi dan negatif untuk unggahan yang tidak mengandung promosi, dibuat suatu model yang kemudian dilatih agar sistem ini mampu mengklasifikasikan apakah suatu unggahan di media sosial mempromosikan judi daring atau tidak."
    )

    st.header("Analisis Data")
    st.write(
        "Dari analisis terhadap sekitar 42.000 unggahan dengan kata kunci “judi” yang telah dikumpulkan dalam rentang awal tahun 2019 hingga pertengahan tahun 2024, berikut adalah beberapa temuan yang dapat dilihat melalui visualisasi data:"
    )

    # Load Data
    @st.cache_data
    def load_data():
        return pd.read_csv("data/labeled_final.csv")

    df = load_data()

    df = df.rename(
        columns={
            "text": "Processed",
            "label": "Label",
        }
    )

    ogpd_visualize_data(df)


elif page == "Explorer":
    st.title("Explorer")
    explorer_option = st.selectbox(
        "Pilih Explorer:", ("Plain Text", "Facebook", "Instagram", "X")
    )

    model_path = "models/improved_svm_model.pkl"

    slang_df = pd.read_csv("data/Kata_Baku_Final.csv")
    slang_dict = dict(zip(slang_df.iloc[:, 0], slang_df.iloc[:, 1]))

    stopwords_df = pd.read_csv("data/Stopwords.csv")
    stopwords = stopwords_df.iloc[:, 0].tolist()

    def cleaning(text):
        text = str(text).lower()
        text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)
        text = re.sub(r"@[^\s]+", "", text)
        # text = re.sub(r"#[^\s]+", "", text)
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

    if explorer_option == "Plain Text":
        st.header("Plain Text")
        st.write(
            "Plain Text Explorer memungkinkan Anda untuk memasukkan teks secara manual atau menggunggah file yang berisi teks untuk melakukan klasifikasi"
        )

        directory = "plain-text-data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        container = st.container(border=True)
        # Pilihan input: Teks atau CSV
        input_type = container.radio("Pilih jenis input:", ("Teks", "Unggah File"))

        if input_type == "Teks":
            # Inputan teks
            user_input = container.text_area(
                "Masukkan teks:",
                value="Judi online merupakan tindakan yang tidak bermoral dan merugikan masyarakat.",
            )
            lines = user_input.split("\n")

            # Hapus baris kosong
            lines = [line for line in lines if line.strip() != ""]

            # Menggabungkan teks input menjadi dataframe
            if lines:
                df_input = pd.DataFrame(lines, columns=["Text"])
            else:
                st.error("Teks tidak boleh kosong.")

        elif input_type == "Unggah File":
            # Unggah file CSV
            uploaded_file = container.file_uploader(
                "Unggah file", type=["csv", "xlsx", "xls"]
            )

            if uploaded_file is not None:
                file_name = uploaded_file.name
                file_extension = os.path.splitext(file_name)[1]

                # Memilih metode pembacaan berdasarkan ekstensi file
                if file_extension.lower() == ".csv":
                    df_uploaded = pd.read_csv(uploaded_file)
                elif file_extension.lower() in [".xlsx", ".xls"]:
                    df_uploaded = pd.read_excel(uploaded_file)

                # Mengubah nama kolom pertama menjadi 'Text'
                first_column_name = df_uploaded.columns[0]
                df_uploaded.rename(columns={first_column_name: "Text"}, inplace=True)

                # Mengatur ulang index dimulai dari 1
                df_uploaded.index = range(1, len(df_uploaded) + 1)
                container.caption("Data dari file yang diunggah:")
                container.dataframe(df_uploaded["Text"], use_container_width=True)
            else:
                container.info(
                    "Pastikan file CSV atau Excel memiliki header atau column."
                )

        # Tombol untuk memulai klasifikasi
        if container.button("Klasifikasi"):
            if input_type == "Teks" and lines:
                df = df_input
            elif input_type == "Unggah File" and uploaded_file is not None:
                df = df_uploaded
            else:
                st.error("Tidak ada data untuk diklasifikasikan.")
                st.stop()

            filename = f"plain-text-data/pte_unggahan_text_{timestamp}.csv"
            df.to_csv(filename, index=False)

            # Load data
            file_path = filename
            if file_path is not None:
                try:
                    df = pd.read_csv(file_path, encoding="latin1")
                except pd.errors.EmptyDataError:
                    st.error("Unggahan tidak ditemukan.")
                    st.stop()
            else:
                st.error("Unggahan tidak ditemukan.")
                st.stop()

            with st.spinner("Pre-processing..."):
                df["Processed"] = df["Text"].apply(cleaning)
                df["Processed"] = df["Processed"].apply(normalize_slang)
                df["Processed"] = df["Processed"].apply(remove_stopwords)
                df["Processed"] = df["Processed"].apply(stemming)

                df.to_csv(file_path, index=False)

            # Klasifikasi menggunakan model SVM

            try:
                model = joblib.load(model_path)
            except FileNotFoundError:
                st.error("Model tidak ditemukan.")
                st.stop()

            X = df["Processed"]

            with st.spinner(
                "Classifying data..."
            ):  # Menggunakan kolom 'processed' untuk klasifikasi
                predictions = model.predict(X)

            # Simpan hasil klasifikasi ke CSV baru
            df["Label"] = predictions

            # Mengatur ulang index dimulai dari 1
            df.index = range(1, len(df) + 1)

            output_filename = f"{filename.replace('.csv', '')}_predicted"
            df[["Text", "Label"]].to_csv(f"{output_filename}.csv", index=False)

            st.success("Klasifikasi selesai!")
            st.dataframe(df[["Text", "Label"]], use_container_width=True)

            visualize_data(df)

    elif explorer_option == "Facebook":
        st.header("Facebook")
        st.write(
            "Facebook Explorer memungkinkan Anda untuk crawling unggahan pribadi, halaman, atau grup di Facebook dan mengklasifikasikannya"
        )

        container = st.container(border=True)

        APIFY_TOKEN = os.getenv("APIFY_TOKEN")
        client = ApifyClient(APIFY_TOKEN)

        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.edge.service import Service
        from selenium.webdriver.edge.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from bs4 import BeautifulSoup

        choice = container.selectbox(
            "Pilih jenis crawling:",
            options=[
                "Unggahan Pribadi atau Halaman",
                "Unggahan Grup",
                # "Unggahan dengan Hashtag",
                # "Komentar dalam Unggahan",
            ],
        )

        directory = "facebook-data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        if choice == "Unggahan Pribadi atau Halaman":

            def init_driver():
                options = Options()
                options.use_chromium = True
                options.add_argument("--headless")  # Run in headless mode
                options.add_argument("--disable-gpu")  # Disable GPU acceleration
                options.add_argument("--window-size=1920x1080")  # Set window size for headless mode
                options.add_argument("--no-sandbox")  # Bypass OS security model, required for Docker
                options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
                options.add_argument("--disable-software-rasterizer")  # Disable software rasterizer

                # Disable images loading
                prefs = {"profile.managed_default_content_settings.images": 2}
                options.add_experimental_option("prefs", prefs)

                driver_service = None
                try:
                    # Coba path pertama
                    driver_service = Service("assets/msedgedriver")
                    driver_service.start()
                except Exception as e1:
                    print(f"Failed to start service with assets/msedgedriver: {e1}")
                    try:
                        # Coba path kedua
                        driver_service = Service("/usr/local/bin/msedgedriver")
                        driver_service.start()
                    except Exception as e2:
                        print(f"Failed to start service with /usr/local/bin/msedgedriver: {e2}")
                        raise e2  # Gagal kedua-duanya, lempar exception

                driver = webdriver.Edge(service=driver_service, options=options)
                return driver

            def close_popup(driver):
                try:
                    close_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, ".x92rtbv"))
                    )
                    close_button.click()
                    time.sleep(5)  # Wait a bit for the pop-up to close
                except Exception as e:
                    print(f"Popup not found or already closed: {e}")

            def click_see_more_buttons(driver):
                try:
                    see_more_buttons = driver.find_elements(
                        By.XPATH,
                        "//div[text()='See more' or text()='Lihat selengkapnya']",
                    )
                    for button in see_more_buttons:
                        driver.execute_script("arguments[0].click();", button)
                        time.sleep(1)  # Wait a bit for the content to expand
                except Exception as e:
                    print(f"No 'See more' buttons found or failed to click: {e}")

            def scrape_posts(driver, username, limit):
                driver.get(f"https://web.facebook.com/{username}/")
                time.sleep(5)  # Wait a bit for the page to load

                # Close authentication pop-up
                close_popup(driver)

                posts = []
                post_set = set()  # To keep track of unique posts
                css_selectors = [
                    ".x1iorvi4.x1pi30zi.x1swvt13.xjkvuk6",
                    ".x1iorvi4.x1pi30zi.x1l90r2v.x1swvt13",
                    ".x1swvt13.x1pi30zi.xexx8yu.x18d9i69",
                    ".x1yx25j4.x13crsa5.x6x52a7.x1rxj1xn.xxpdul3",
                ]

                last_height = driver.execute_script("return document.body.scrollHeight")

                while len(posts) < limit:
                    click_see_more_buttons(driver)  # Click 'See more' buttons

                    for selector in css_selectors:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        for element in elements:
                            text = element.text.replace(
                                "\n", " "
                            )  # Replace newline characters with space
                            if text not in post_set:
                                posts.append(text)
                                post_set.add(text)
                            if len(posts) >= limit:
                                break

                    if len(posts) >= limit:
                        break

                    # Scroll down to load more posts
                    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
                    time.sleep(
                        5
                    )  # Increased delay to 5 seconds to allow more time for posts to load

                    new_height = driver.execute_script(
                        "return document.body.scrollHeight"
                    )
                    if new_height == last_height:
                        break
                    last_height = new_height

                return posts[:limit]

            def save_to_csv(posts, filename="facebook_posts.csv"):
                with open(filename, mode="w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Text"])
                    for post in posts:
                        writer.writerow([post])

            def main(username, limit):
                driver = init_driver()
                try:
                    posts = scrape_posts(driver, username, limit)
                    if not posts:
                        return None  # Return None if posts list is empty
                    filename = f"facebook-data/fe_unggahan_{username}_{timestamp}.csv"
                    save_to_csv(posts, filename)
                    return filename
                finally:
                    driver.quit()

            col1, col2 = container.columns(2)
            with col1:
                username = st.text_input(
                    "Masukkan username Facebook:", value="facebook"
                )
            with col2:
                resultsLimit = st.number_input(
                    "Masukkan batas maksimum unggahan:", min_value=20, value=50
                )
            if not username:
                st.error("Username Facebook tidak boleh kosong.")
            elif " " in username:
                st.error("Username Facebook tidak boleh mengandung spasi.")
            elif resultsLimit is None:
                st.error("Batas maksimum unggahan tidak boleh kosong.")
            elif resultsLimit < 20:
                st.error("Batas unggahan minimal adalah 20.")
            else:
                if container.button("Crawl dan Klasifikasi"):
                    with st.spinner("Crawling data..."):
                        filename = main(username, resultsLimit)

                    # Load data
                    file_path = filename
                    if file_path is not None:
                        try:
                            df = pd.read_csv(file_path, encoding="latin1")
                        except pd.errors.EmptyDataError:
                            st.error("Unggahan tidak ditemukan.")
                            st.stop()
                    else:
                        st.error("Unggahan tidak ditemukan.")
                        st.stop()

                    with st.spinner("Pre-processing..."):
                        df["Processed"] = df["Text"].apply(cleaning)
                        df["Processed"] = df["Processed"].apply(normalize_slang)
                        df["Processed"] = df["Processed"].apply(remove_stopwords)
                        df["Processed"] = df["Processed"].apply(stemming)

                        df.to_csv(file_path, index=False)

                    # Klasifikasi menggunakan model SVM

                    try:
                        model = joblib.load(model_path)
                    except FileNotFoundError:
                        st.error("Model tidak ditemukan.")
                        st.stop()

                    X = df[
                        "Processed"
                    ]  # Menggunakan kolom 'processed' untuk klasifikasi
                    with st.spinner("Classifying data..."):
                        predictions = model.predict(X)

                    # Simpan hasil klasifikasi ke CSV baru
                    df["label"] = predictions
                    df = df.rename(
                        columns={
                            "label": "Label",
                        }
                    )

                    # Mengatur ulang index dimulai dari 1
                    df.index = range(1, len(df) + 1)

                    output_filename = f"{filename.replace('.csv', '')}_predicted"
                    df[["Text", "Label"]].to_csv(f"{output_filename}.csv", index=False)

                    st.success("Crawling dan klasifikasi selesai!")
                    st.dataframe(df[["Text", "Label"]], use_container_width=True)

                    visualize_data(df)

        elif choice == "Unggahan Grup":

            def init_driver():
                options = Options()
                options.use_chromium = True
                options.add_argument("--headless")  # Run in headless mode
                options.add_argument("--disable-gpu")  # Disable GPU acceleration
                options.add_argument("--window-size=1920x1080")  # Set window size for headless mode
                options.add_argument("--no-sandbox")  # Bypass OS security model, required for Docker
                options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
                options.add_argument("--disable-software-rasterizer")  # Disable software rasterizer

                # Disable images loading
                prefs = {"profile.managed_default_content_settings.images": 2}
                options.add_experimental_option("prefs", prefs)

                driver_service = None
                try:
                    # Coba path pertama
                    driver_service = Service("assets/msedgedriver")
                    driver_service.start()
                except Exception as e1:
                    print(f"Failed to start service with assets/msedgedriver: {e1}")
                    try:
                        # Coba path kedua
                        driver_service = Service("/usr/local/bin/msedgedriver")
                        driver_service.start()
                    except Exception as e2:
                        print(f"Failed to start service with /usr/local/bin/msedgedriver: {e2}")
                        raise e2  # Gagal kedua-duanya, lempar exception

                driver = webdriver.Edge(service=driver_service, options=options)
                return driver

            def close_popup(driver):
                try:
                    close_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, ".x92rtbv"))
                    )
                    close_button.click()
                    time.sleep(5)  # Wait a bit for the pop-up to close
                except Exception as e:
                    print(f"Popup not found or already closed: {e}")

            def click_see_more_buttons(driver):
                try:
                    see_more_buttons = driver.find_elements(
                        By.XPATH,
                        "//div[text()='See more' or text()='Lihat selengkapnya']",
                    )
                    for button in see_more_buttons:
                        driver.execute_script("arguments[0].click();", button)
                        time.sleep(1)  # Wait a bit for the content to expand
                except Exception as e:
                    print(f"No 'See more' buttons found or failed to click: {e}")

            def is_valid_profile_url(url):
                # Check if the URL is a valid Facebook profile URL
                return (
                    "/people/" in url or "facebook.com/" in url and "/photo/" not in url
                )

            def clean_url(url):
                return url.split("?")[0]

            def scroll_until_limit(driver, limit, extra=30):
                body = driver.find_element(By.TAG_NAME, "body")
                extended_limit = limit + extra
                while True:
                    elements = driver.find_elements(
                        By.CSS_SELECTOR, "div.x1yztbdb.x1n2onr6.xh8yej3.x1ja2u2z"
                    )
                    if len(elements) > extended_limit:
                        break
                    body.send_keys(Keys.PAGE_DOWN)
                    time.sleep(
                        1
                    )  # Increase time sleep to ensure elements load properly

            def scrape_posts(driver, group_url, limit):
                driver.get(f"{group_url}")
                time.sleep(5)  # Wait a bit for the page to load

                # Close authentication pop-up
                close_popup(driver)

                posts = []
                post_set = set()  # To keep track of unique posts
                css_selectors = [
                    ".x1iorvi4.x1pi30zi.x1swvt13.xjkvuk6",
                    ".x1iorvi4.x1pi30zi.x1l90r2v.x1swvt13",
                    ".x1swvt13.x1pi30zi.xexx8yu.x18d9i69",
                    ".x1yx25j4.x13crsa5.x6x52a7.x1rxj1xn.xxpdul3",
                ]

                last_height = driver.execute_script("return document.body.scrollHeight")

                while len(posts) < limit:
                    click_see_more_buttons(driver)  # Click 'See more' buttons

                    page_source = driver.page_source
                    soup = BeautifulSoup(page_source, "html.parser")

                    for post_wrapper in soup.find_all(
                        "div", class_="x1yztbdb x1n2onr6 xh8yej3 x1ja2u2z"
                    ):
                        if len(posts) >= limit:
                            break

                        post_data = {}

                        # Extracting the name and username
                        try:
                            name_element = post_wrapper.find("a", {"aria-label": True})
                            if name_element:
                                name = name_element["aria-label"]
                                url = name_element["href"]
                                if not is_valid_profile_url(url):
                                    continue
                                post_data["name"] = name
                                post_data["username"] = clean_url(url)
                            else:
                                post_data["name"] = "Unknown"
                                post_data["username"] = "Unknown"
                        except Exception as e:
                            print(f"Exception when finding user element: {e}")
                            post_data["name"] = "Unknown"
                            post_data["username"] = "Unknown"

                        # Extracting the post text
                        post_text = ""
                        for selector in css_selectors:
                            text_elements = post_wrapper.select(selector)
                            for element in text_elements:
                                post_text += (
                                    element.get_text(separator=" ", strip=True) + " "
                                )
                        post_data["text"] = post_text.strip()

                        # Skip duplicate posts
                        if post_data["text"] in post_set:
                            continue
                        post_set.add(post_data["text"])

                        posts.append(post_data)

                    posts = [post for post in posts if post["text"]]
                    # Scroll down to load more posts
                    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
                    time.sleep(
                        5
                    )  # Increased delay to 5 seconds to allow more time for posts to load

                    new_height = driver.execute_script(
                        "return document.body.scrollHeight"
                    )
                    if new_height == last_height:
                        break
                    last_height = new_height

                return posts[:limit]

            def save_to_csv(posts, filename="facebook_posts.csv"):
                with open(filename, mode="w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(
                        file, fieldnames=["name", "username", "text"]
                    )
                    writer.writeheader()
                    for post in posts:
                        writer.writerow(post)

            def main(group_url, limit):
                driver = init_driver()
                try:
                    posts = scrape_posts(driver, group_url, limit)
                    if not posts:
                        return None  # Return None if posts list is empty
                    filename = f"facebook-data/fe_unggahan_grup_{timestamp}.csv"
                    save_to_csv(posts, filename)
                    return filename
                finally:
                    driver.quit()

            col1, col2 = container.columns(2)
            with col1:
                group_url = st.text_input(
                    "Masukkan URL grup Facebook:",
                    value="https://web.facebook.com/groups/1765762043450939/",
                )
            with col2:
                resultsLimit = st.number_input(
                    "Masukkan batas maksimum unggahan:", min_value=20, value=50
                )

            if not group_url:
                st.error("URL tidak boleh kosong.")
            else:
                prefixes = [
                    "https://facebook.com/groups/",
                    "https://www.facebook.com/groups/",
                    "https://m.facebook.com/groups/",
                    "https://mbasic.facebook.com/groups/",
                    "https://facebook.com/groups/",
                    "https://web.facebook.com/groups/",
                ]
                valid_prefix = False
                for prefix in prefixes:
                    if group_url.startswith(prefix):
                        group_url = group_url.replace(
                            prefix, "https://web.facebook.com/groups/"
                        )
                        valid_prefix = True
                        break
                if not valid_prefix:
                    st.error("URL grup Facebook tidak valid.")
                else:
                    if "?" in group_url:
                        group_url = group_url.split("?")[0]
                    group_id = group_url.split("https://web.facebook.com/groups/")[
                        1
                    ].rstrip("/")
                    if group_id == "":
                        st.error(
                            "URL grup tidak boleh kosong setelah 'https://web.facebook.com/groups/'."
                        )
                    elif resultsLimit is None:
                        st.error("Batas maksimum unggahan tidak boleh kosong.")
                    elif resultsLimit < 20:
                        st.error("Batas unggahan minimal adalah 20.")
                    else:
                        if container.button("Crawl dan Klasifikasi"):
                            # Prepare the Actor input for Facebook group

                            with st.spinner("Crawling data..."):
                                filename = main(group_url, resultsLimit)

                            # Load data
                            file_path = filename
                            if file_path is not None:
                                try:
                                    df = pd.read_csv(file_path, encoding="latin1")
                                except pd.errors.EmptyDataError:
                                    st.error("Unggahan tidak ditemukan.")
                                    st.stop()
                            else:
                                st.error("Unggahan tidak ditemukan.")
                                st.stop()

                            with st.spinner("Pre-processing..."):
                                df["Processed"] = df["text"].apply(cleaning)
                                df["Processed"] = df["Processed"].apply(normalize_slang)
                                df["Processed"] = df["Processed"].apply(
                                    remove_stopwords
                                )
                                df["Processed"] = df["Processed"].apply(stemming)

                                df.to_csv(file_path, index=False)

                            # Klasifikasi menggunakan model SVM

                            try:
                                model = joblib.load(model_path)
                            except FileNotFoundError:
                                st.error("Model tidak ditemukan.")
                                st.stop()

                            X = df[
                                "Processed"
                            ]  # Menggunakan kolom 'processed' untuk klasifikasi
                            with st.spinner("Classifying data..."):
                                predictions = model.predict(X)

                            # Simpan hasil klasifikasi ke CSV baru
                            df["label"] = predictions
                            df = df.rename(
                                columns={
                                    "label": "Label",
                                    "text": "Text",
                                    "name": "Name",
                                    "username": "Username",
                                }
                            )

                            # Mengatur ulang index dimulai dari 1
                            df.index = range(1, len(df) + 1)

                            output_filename = (
                                f"{filename.replace('.csv', '')}_predicted"
                            )
                            df[["Text", "Name", "Username", "Label"]].to_csv(
                                f"{output_filename}.csv", index=False
                            )

                            st.success("Crawling dan klasifikasi selesai!")
                            st.dataframe(
                                df[["Text", "Name", "Username", "Label"]],
                                column_config={
                                    "Username": st.column_config.LinkColumn()
                                },
                                use_container_width=True,
                            )

                            visualize_data(df)

        # elif choice == "Unggahan dengan Hashtag":
        #     col1, col2 = st.columns(2)
        #     with col1:
        #         hashtag = st.text_input("Masukkan hashtag:", value="judi")
        #     with col2:
        #         resultsLimit = st.number_input(
        #             "Masukkan batas maksimum unggahan:", min_value=1, value=20
        #         )

        #     if st.button("Crawl dan Klasifikasi"):
        #         # Prepare the Actor input for Facebook hashtag
        #         run_input_hashtag = {
        #             "keywordList": [hashtag],
        #             "resultsLimit": resultsLimit,
        #         }

        #         with st.spinner("Crawling data..."):
        #             # Run the Actor and wait for it to finish
        #             run_hashtag = client.actor("apify/facebook-hashtag-scraper").call(
        #                 run_input=run_input_hashtag
        #             )

        #         # Fetch and print Actor results from the run's dataset (if there are any)
        #         data = []
        #         for item in client.dataset(
        #             run_hashtag["defaultDatasetId"]
        #         ).iterate_items():
        #             data.append(item)
        #         df = pd.DataFrame(data)

        #         filename = f"facebook-data/fe_unggahan_{hashtag}_{timestamp}.csv"
        #         df.to_csv(filename, index=False)

        # elif choice == "Komentar dalam Unggahan":
        #     startUrls = st.text_input(
        #         "Masukkan URL unggahan Facebook:",
        #         value="https://www.facebook.com/humansofnewyork/posts/pfbid0BbKbkisExKGSKuhee9a7i86RwRuMKFC8NSkKStB7CsM3uXJuAAfZLrkcJMXxhH4Yl",
        #     )

        #     col1, col2 = st.columns(2)
        #     with col1:
        #         resultsLimit = st.number_input(
        #             "Masukkan batas maksimum komentar:", min_value=1, value=20
        #         )
        #     with col2:
        #         viewOption = st.selectbox(
        #             "Pilih jenis crawling tampilan:",
        #             options=["RANKED_UNFILTERED", "RANKED_THREADED", "RECENT_ACTIVITY"],
        #         )

        #     if st.button("Crawl dan Klasifikasi"):
        #         # Prepare the Actor input for Facebook post
        #         run_input_post = {
        #             "startUrls": [{"url": startUrls}],
        #             "resultsLimit": resultsLimit,
        #             "includeNestedComments": False,
        #             "viewOption": viewOption,
        #         }

        #         with st.spinner("Crawling data..."):
        #             # Run the Actor and wait for it to finish
        #             run_post = client.actor("us5srxAYnsrkgUv2v").call(
        #                 run_input=run_input_post
        #             )

        #         # Fetch and print Actor results from the run's dataset (if there are any)
        #         data = []
        #         for item in client.dataset(
        #             run_post["defaultDatasetId"]
        #         ).iterate_items():
        #             data.append(item)
        #         df = pd.DataFrame(data)

        #         filename = f"facebook-data/fe_komentar_{timestamp}.csv"
        #         df.to_csv(filename, index=False)

    elif explorer_option == "Instagram":
        st.header("Instagram")

        st.write(
            "Instagram Explorer memungkinkan Anda untuk crawling unggahan pribadi, unggahan berdasarkan hashtag, atau komentar dalam unggahan di Instagram dan mengklasifikasikannya."
        )

        container = st.container(border=True)

        APIFY_TOKEN = os.getenv("APIFY_TOKEN")
        client = ApifyClient(APIFY_TOKEN)

        # Add new input for choice
        choice = container.selectbox(
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
            col1, col2 = container.columns(2)
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
            elif " " in username:
                st.error("Username Instagram tidak boleh mengandung spasi.")
            elif resultsLimit < 1:
                st.error("Batas unggahan minimal adalah 1.")
            elif resultsLimit is None:
                st.error("Batas maksimum unggahan tidak boleh kosong.")
            else:
                if container.button("Crawl dan Klasifikasi"):
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
                    if file_path is not None:
                        try:
                            df = pd.read_csv(file_path, encoding="latin1")
                        except pd.errors.EmptyDataError:
                            st.error("Unggahan tidak ditemukan.")
                            st.stop()
                    else:
                        st.error("Unggahan tidak ditemukan.")
                        st.stop()

                    with st.spinner("Pre-processing..."):
                        df["Processed"] = df["caption"].apply(cleaning)
                        df["Processed"] = df["Processed"].apply(normalize_slang)
                        df["Processed"] = df["Processed"].apply(remove_stopwords)
                        df["Processed"] = df["Processed"].apply(stemming)

                        df.to_csv(file_path, index=False)

                    # Klasifikasi menggunakan model SVM

                    try:
                        model = joblib.load(model_path)
                    except FileNotFoundError:
                        st.error("Model tidak ditemukan.")
                        st.stop()

                    X = df[
                        "Processed"
                    ]  # Menggunakan kolom 'processed' untuk klasifikasi
                    with st.spinner("Classifying data..."):
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
                    df.index = range(1, len(df) + 1)

                    output_filename = f"{filename.replace('.csv', '')}_predicted"
                    df[["Text", "URL", "Label"]].to_csv(
                        f"{output_filename}.csv", index=False
                    )

                    st.success("Crawling dan klasifikasi selesai!")
                    st.dataframe(
                        df[["Text", "URL", "Label"]],
                        column_config={"URL": st.column_config.LinkColumn()},
                        use_container_width=True,
                    )

                    visualize_data(df)

        elif choice == "Unggahan dengan Hashtag":
            hashtags = container.text_input("Masukkan hashtag:", value="judi")

            col1, col2 = container.columns(2)
            with col1:
                resultsLimit = st.number_input(
                    "Masukkan batas maksimum unggahan:", min_value=1, value=20
                )
            with col2:
                onlyPostsNewerThan = st.date_input(
                    "Hanya unggahan sejak tanggal:", value=yesterday
                )
            recommendations()
            if not hashtags:
                st.error("Hashtag tidak boleh kosong.")
            elif " " in hashtags:
                st.error("Hashtag tidak boleh mengandung spasi.")
            elif resultsLimit is None:
                st.error("Batas maksimum unggahan tidak boleh kosong.")
            elif resultsLimit < 1:
                st.error("Batas unggahan minimal adalah 1.")
            elif onlyPostsNewerThan is None:
                st.error("Tanggal tidak boleh kosong.")
            else:
                if container.button("Crawl dan Klasifikasi"):
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
                    if file_path is not None:
                        try:
                            df = pd.read_csv(file_path, encoding="latin1")
                        except pd.errors.EmptyDataError:
                            st.error("Unggahan tidak ditemukan.")
                            st.stop()
                    else:
                        st.error("Unggahan tidak ditemukan.")
                        st.stop()

                    with st.spinner("Pre-processing..."):
                        df["Processed"] = df["caption"].apply(cleaning)
                        df["Processed"] = df["Processed"].apply(normalize_slang)
                        df["Processed"] = df["Processed"].apply(remove_stopwords)
                        df["Processed"] = df["Processed"].apply(stemming)

                        df.to_csv(file_path, index=False)

                    # Klasifikasi menggunakan model SVM

                    try:
                        model = joblib.load(model_path)
                    except FileNotFoundError:
                        st.error("Model tidak ditemukan.")
                        st.stop()

                    X = df[
                        "Processed"
                    ]  # Menggunakan kolom 'processed' untuk klasifikasi
                    with st.spinner("Classifying data..."):
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
                    df.index = range(1, len(df) + 1)

                    output_filename = f"{filename.replace('.csv', '')}_predicted"
                    df[["Text", "URL", "Username", "Label"]].to_csv(
                        f"{output_filename}.csv", index=False
                    )

                    st.success("Crawling dan klasifikasi selesai!")
                    st.dataframe(
                        df[["Text", "URL", "Username", "Label"]],
                        column_config={"URL": st.column_config.LinkColumn()},
                        use_container_width=True,
                    )
                    visualize_data(df)

        elif choice == "Komentar dalam Unggahan":
            # Add new input for hashtag
            directUrls = container.text_input(
                "Masukkan link unggahan Instagram:",
                value="https://www.instagram.com/p/C8Y52QPPmib/",
            )
            directUrls = directUrls.split("?")[0]
            resultsLimit = container.number_input(
                "Masukkan batas maksimum komentar:", min_value=1, value=20
            )
            if not directUrls:
                st.error("URL tidak boleh kosong.")
            else:
                prefixes = [
                    "https://instagram.com/p/",
                    "https://www.instagram.com/p/",
                ]
                valid_prefix = False
                for prefix in prefixes:
                    if directUrls.startswith(prefix):
                        directUrls = directUrls.replace(
                            prefix, "https://www.instagram.com/p/"
                        )
                        valid_prefix = True
                        break
                if not valid_prefix:
                    st.error("URL unggahan Instagram tidak valid.")
                else:
                    if "?" in directUrls:
                        directUrls = directUrls.split("?")[0]
                    post_id = directUrls.split("https://www.instagram.com/p/")[
                        1
                    ].rstrip("/")
                    if post_id == "":
                        st.error(
                            "URL unggahan tidak boleh kosong setelah 'https://www.instagram.com/p/'."
                        )

                    elif resultsLimit is None:
                        st.error("Batas maksimum unggahan tidak boleh kosong.")
                    elif resultsLimit < 1:
                        st.error("Batas unggahan minimal adalah 1.")
                    else:
                        if container.button("Crawl dan Klasifikasi"):
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

                            filename = (
                                f"instagram-data/ie_unggahan_komentar_{timestamp}.csv"
                            )
                            df.to_csv(filename, index=False)

                            # Load data
                            file_path = filename
                            if file_path is not None:
                                try:
                                    df = pd.read_csv(file_path, encoding="latin1")
                                except pd.errors.EmptyDataError:
                                    st.error("Unggahan tidak ditemukan.")
                                    st.stop()
                            else:
                                st.error("Unggahan tidak ditemukan.")
                                st.stop()

                            with st.spinner("Pre-processing..."):
                                df["Processed"] = df["text"].apply(cleaning)
                                df["Processed"] = df["Processed"].apply(normalize_slang)
                                df["Processed"] = df["Processed"].apply(
                                    remove_stopwords
                                )
                                df["Processed"] = df["Processed"].apply(stemming)

                                df.to_csv(file_path, index=False)

                            # Klasifikasi menggunakan model SVM

                            try:
                                model = joblib.load(model_path)
                            except FileNotFoundError:
                                st.error("Model tidak ditemukan.")
                                st.stop()

                            X = df[
                                "Processed"
                            ]  # Menggunakan kolom 'processed' untuk klasifikasi
                            with st.spinner("Classifying data..."):
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
                            df.index = range(1, len(df) + 1)

                            output_filename = (
                                f"{filename.replace('.csv', '')}_predicted"
                            )
                            df[["Text", "Username", "Label"]].to_csv(
                                f"{output_filename}.csv", index=False
                            )

                            st.success("Crawling dan klasifikasi selesai!")
                            st.dataframe(
                                df[["Text", "Username", "Label"]],
                                column_config={"URL": st.column_config.LinkColumn()},
                                use_container_width=True,
                            )

                            visualize_data(df)

    elif explorer_option == "X":
        st.header("X")
        st.write(
            "X Explorer memungkinkan Anda untuk crawling unggahan pribadi atau unggahan berdasarkan kata kunci dan mengklasifikasikannya."
        )

        container = st.container(border=True)

        x_auth_token = os.getenv("X_AUTH_TOKEN")

        x_option = container.selectbox(
            "Pilih jenis crawling:",
            ("Unggahan Pribadi", "Pencari Unggahan"),
        )
        
        directory = "tweets-data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        if x_option == "Unggahan Pribadi":
            # Tambahkan kode untuk menangani unggahan pribadi di sini
            col1, col2 = container.columns(2)
            with col1:
                search_keyword = st.text_input(
                    "Masukkan username X:", value="ardisatriawan"
                )
            with col2:
                limit = st.number_input(
                    "Masukkan batas maksimum unggahan:", min_value=20, value=200
                )

            col3, col4 = container.columns(2)
            with col3:
                start_date = st.date_input("Hanya unggahan sejak tanggal:", value=today)
            with col4:
                end_date = st.date_input(
                    "Hanya unggahan sampai tanggal:", value=tomorrow
                )

            if not search_keyword:
                st.error("Username X tidak boleh kosong.")
            elif " " in search_keyword:
                st.error("Username X tidak boleh mengandung spasi.")
            elif limit is None:
                st.error("Batas maksimum unggahan tidak boleh kosong.")
            elif limit < 20:
                st.error("Batas unggahan minimal adalah 20.")
            elif start_date is None or end_date is None:
                st.error("Tanggal tidak boleh kosong.")
            elif start_date > end_date:
                st.error("Silakan masukkan rentang tanggal yang valid.")
            else:
                if container.button("Crawl dan Klasifikasi"):
                    # Format search keyword
                    search_keyword_formatted = (
                        f"from:{search_keyword} since:{start_date} until:{end_date}"
                    )

                    filename = f"xe_unggahan_{search_keyword}_{timestamp}.csv"

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
                                    or "Total tweets saved: 0" in line
                                ):
                                    print(
                                        "No more tweets found or done scrolling, stopping..."
                                    )
                                    process.terminate()
                                    break
                        rc = process.poll()

                    # Load data
                    file_path = f"tweets-data/{filename}"
                    if file_path is not None:
                        try:
                            df = pd.read_csv(file_path, encoding="latin1")
                        except pd.errors.EmptyDataError:
                            st.error("Unggahan tidak ditemukan.")
                            st.stop()
                    else:
                        st.error("Unggahan tidak ditemukan.")
                        st.stop()

                    with st.spinner("Pre-processing..."):
                        df["Processed"] = df["full_text"].apply(cleaning)
                        df["Processed"] = df["Processed"].apply(normalize_slang)
                        df["Processed"] = df["Processed"].apply(remove_stopwords)
                        df["Processed"] = df["Processed"].apply(stemming)

                        df.to_csv(file_path, index=False)

                    # Klasifikasi menggunakan model SVM

                    try:
                        model = joblib.load(model_path)
                    except FileNotFoundError:
                        st.error("Model tidak ditemukan.")
                        st.stop()

                    X = df[
                        "Processed"
                    ]  # Menggunakan kolom 'processed' untuk klasifikasi
                    with st.spinner("Classifying data..."):
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
                    df.index = range(1, len(df) + 1)

                    output_filename = f"{filename.replace('.csv', '')}_predicted"
                    df[["Text", "URL", "Username", "Label"]].to_csv(
                        f"tweets-data/{output_filename}.csv", index=False
                    )

                    st.success("Crawling dan klasifikasi selesai!")
                    st.dataframe(
                        df[["Text", "URL", "Username", "Label"]],
                        column_config={"URL": st.column_config.LinkColumn()},
                        use_container_width=True,
                    )

                    visualize_data(df)

        elif x_option == "Pencari Unggahan":
            # Tambahkan kode untuk menangani pencarian unggahan di sini
            col1, col2 = container.columns(2)
            with col1:
                search_keyword = st.text_input(
                    "Masukkan kata kunci pencarian:", value="judi"
                )
            with col2:
                limit = st.number_input(
                    "Masukkan batas maksimum unggahan:", min_value=20, value=200
                )

            col3, col4 = container.columns(2)
            with col3:
                start_date = st.date_input("Hanya unggahan sejak tanggal:", value=today)
            with col4:
                end_date = st.date_input(
                    "Hanya unggahan sampai tanggal:", value=tomorrow
                )
            recommendations()
            if not search_keyword:
                st.error("Kata kunci pencarian tidak boleh kosong.")
            elif limit is None:
                st.error("Batas maksimum unggahan tidak boleh kosong.")
            elif limit < 20:
                st.error("Batas unggahan minimal adalah 20.")
            elif start_date is None:
                st.error("Tanggal mulai tidak boleh kosong.")
            elif end_date is None:
                st.error("Tanggal akhir tidak boleh kosong.")
            elif start_date > end_date:
                st.error("Tanggal mulai tidak boleh lebih besar dari tanggal akhir.")
            else:
                if container.button("Crawl dan Klasifikasi"):
                    # Format search keyword
                    search_keyword_formatted = (
                        f"{search_keyword} lang:id since:{start_date} until:{end_date}"
                    )

                    filename = (
                        f"xe_unggahan_{search_keyword.replace(' ', '')}_{timestamp}.csv"
                    )

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
                                    or "Total tweets saved: 0" in line
                                ):
                                    print(
                                        "No more tweets found or done scrolling, stopping..."
                                    )
                                    process.terminate()
                                    break
                        rc = process.poll()

                    # Load data
                    file_path = f"tweets-data/{filename}"
                    if file_path is not None:
                        try:
                            df = pd.read_csv(file_path, encoding="latin1")
                        except pd.errors.EmptyDataError:
                            st.error("Unggahan tidak ditemukan.")
                            st.stop()
                    else:
                        st.error("Unggahan tidak ditemukan.")
                        st.stop()

                    with st.spinner("Pre-processing..."):
                        df["Processed"] = df["full_text"].apply(cleaning)
                        df["Processed"] = df["Processed"].apply(normalize_slang)
                        df["Processed"] = df["Processed"].apply(remove_stopwords)
                        df["Processed"] = df["Processed"].apply(stemming)

                        df.to_csv(file_path, index=False)

                    # Klasifikasi menggunakan model SVM

                    try:
                        model = joblib.load(model_path)
                    except FileNotFoundError:
                        st.error("Model tidak ditemukan.")
                        st.stop()

                    X = df[
                        "Processed"
                    ]  # Menggunakan kolom 'processed' untuk klasifikasi
                    with st.spinner(
                        "Classifying data..."
                    ):  # Menggunakan kolom 'processed' untuk klasifikasi
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
                    df.index = range(1, len(df) + 1)

                    output_filename = f"{filename.replace('.csv', '')}_predicted"
                    df[["Text", "URL", "Username", "Label"]].to_csv(
                        f"tweets-data/{output_filename}.csv", index=False
                    )

                    st.success("Crawling dan klasifikasi selesai!")
                    st.dataframe(
                        df[["Text", "URL", "Username", "Label"]],
                        column_config={"URL": st.column_config.LinkColumn()},
                        use_container_width=True,
                    )

                    visualize_data(df)


elif page == "Archive":
    st.title("Archive")
    st.write("Unduh data yang telah di-crawl sebelumnya.")

    with st.container(border=True):

        def list_files_in_directory(directory):
            files_data = []
            for file in os.listdir(directory):
                if file.endswith("predicted.csv"):
                    parts = file.split("_")
                    date_time_str = f"{parts[-3]}_{parts[-2]}"
                    date_time = datetime.strptime(date_time_str, "%Y%m%d_%H%M%S")
                    formatted_date_time = date_time.strftime("%Y-%m-%d %H:%M:%S")
                    name = parts[2] if len(parts) > 2 else "Unknown"
                    file_path = os.path.join(directory, file)
                    files_data.append(
                        {
                            "File": file,
                            "Name": name,
                            "Date Time": formatted_date_time,
                            "Path": file_path,
                            "Timestamp": date_time,
                        }
                    )

            # Mengurutkan file berdasarkan timestamp, dari yang terbaru ke yang terlama
            files_data.sort(key=lambda x: x["Timestamp"], reverse=True)

            return files_data

        directories = [
            "plain-text-data",
            "facebook-data",
            "instagram-data",
            "tweets-data",
        ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
            with st.expander(f"{directory.replace('-', ' ').title()}"):
                files = list_files_in_directory(directory)
                if files:
                    for file in files:
                        with st.container(border=True):
                            col1, col2, col3, col4 = st.columns(
                                [3, 2, 1, 1], vertical_alignment="center"
                            )
                            col1.write(file["Name"])
                            col2.write(file["Date Time"])
                            with col3:
                                with open(file["Path"], "rb") as f:
                                    st.download_button(
                                        label="Unduh",
                                        data=f,
                                        file_name=file["File"],
                                        mime="text/csv",
                                    )
                            with col4:
                                if st.button("Hapus", key=f"delete_{file['File']}"):
                                    # Hapus file yang diprediksi
                                    os.remove(file["Path"])

                                    # Bangun kembali nama file asli
                                    original_file_name = file["File"].replace(
                                        "_predicted", ""
                                    )
                                    original_file_path = os.path.join(
                                        directory, original_file_name
                                    )

                                    # Periksa dan hapus file asli jika ada
                                    if os.path.exists(original_file_path):
                                        os.remove(original_file_path)

                                    st.rerun()
                else:
                    st.info("Tidak ada file yang tersedia.")

        with st.popover("Hapus Semua"):
            st.write(
                "Apakah Anda yakin ingin menghapus semua data yang telah di-crawl?"
            )
            all_files = []
            for directory in directories:
                all_files.extend(list_files_in_directory(directory))
            if all_files:
                if st.button("Hapus"):
                    for directory in directories:
                        for file in os.listdir(directory):
                            if file.endswith(".csv"):
                                os.remove(os.path.join(directory, file))
                    st.rerun()
            else:
                st.button("Tidak ada file", disabled=True)

    with st.container(border=True):
        st.subheader("About")
        st.write(
            "OGPD atau Online Gambling Promotion Detector merupakan sebuah proyek yang dibangun untuk memberikan wawasan kepada pihak berwenang mengenai konten promosi judi daring yang belakangan ini marak di Indonesia. Selain itu, diciptakan sebuah sistem yang mampu mengumpulkan dan mengklasifikasikan konten yang tersebar di berbagai media sosial seperti Facebook, Instagram, dan X. Konten yang terindikasi mengandung promosi judi daring akan diberi label positif, sedangkan konten yang tidak mengandung promosi judi daring diberi label negatif."
        )
        st.write(
            "Kemampuan sistem dalam mengklasifikasikan konten dibantu dengan algoritma Support Vector Machine (SVM) yang memiliki tingkat akurasi di atas 90%. Meskipun tingkat akurasinya cukup tinggi, penting untuk melakukan pengecekan ulang terkait konten yang diberi label."
        )
        st.write("For any inquiries, please contact me at: faarismudawork@gmail.com")
