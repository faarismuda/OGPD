import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from streamlit_navigation_bar import st_navbar

# Setup
nltk.download('stopwords')
stop_words = list(stopwords.words('indonesian'))  # Convert to list

@st.cache_data
def load_data():
    return pd.read_csv('labeled_final.csv')

@st.cache_data
def generate_wordcloud(all_text, circle_mask):
    return WordCloud(
        width=800, height=800, background_color='white', mask=circle_mask,
        contour_width=1, contour_color='steelblue', min_font_size=10,
        colormap='viridis', prefer_horizontal=1.0, scale=3,
        max_words=50000000, relative_scaling=0.5, normalize_plurals=False,
        stopwords=stop_words
    ).generate(all_text)

def get_top_ngrams(corpus, ngram_range=(1, 1), n=None):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# Load Data
data = load_data()

# Navigation
pages = ["Home", "Explorer", "About"]
styles = {
    "nav": {
        "background-color": "rgb(116, 185, 255)",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(49, 51, 63)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}
page = st_navbar(pages, styles=styles)
st.write(page)

# Display content based on selected page
if page == "Home":
    st.title('Home')
    st.write('Penjelasan singkat aplikasi.')

    # Pie Chart
    st.subheader('Distribusi Label')
    label_counts = data['label'].value_counts()
    st.bar_chart(label_counts)

    # Wordcloud
    st.subheader('Word Cloud')
    all_texts = data['text']
    all_text = ' '.join(all_texts.astype(str).tolist())
    circle_mask = np.array(Image.open('mask.png'))
    wordcloud = generate_wordcloud(all_text, circle_mask)
    st.image(wordcloud.to_array(), use_column_width=True)

    # Frequency of Words
    st.subheader('Frekuensi Kata')
    positive_texts = data[data['label'] == "Positif"]['text']
    negative_texts = data[data['label'] == "Negatif"]['text']
    top_positive_words = get_top_ngrams(positive_texts, ngram_range=(1, 1), n=20)
    top_negative_words = get_top_ngrams(negative_texts, ngram_range=(1, 1), n=20)
    positive_df = pd.DataFrame(top_positive_words, columns=['word', 'frequency'])
    negative_df = pd.DataFrame(top_negative_words, columns=['word', 'frequency'])
    st.bar_chart(positive_df.set_index('word'))
    st.bar_chart(negative_df.set_index('word'))

    # Bigram Analysis
    st.subheader('Analisis Bigram')
    top_positive_bigrams = get_top_ngrams(positive_texts, ngram_range=(2, 2), n=20)
    top_negative_bigrams = get_top_ngrams(negative_texts, ngram_range=(2, 2), n=20)
    positive_bigram_df = pd.DataFrame(top_positive_bigrams, columns=['bigram', 'frequency'])
    negative_bigram_df = pd.DataFrame(top_negative_bigrams, columns=['bigram', 'frequency'])
    st.bar_chart(positive_bigram_df.set_index('bigram'))
    st.bar_chart(negative_bigram_df.set_index('bigram'))

    # Histogram
    st.subheader('Distribusi Panjang Tweet')
    data['tweet_length'] = data['text'].apply(len)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data, x='tweet_length', hue='label', multiple='stack', palette=['red', 'blue'], bins=50, ax=ax)
    ax.set_xlabel('Panjang Tweet')
    ax.set_ylabel('Frekuensi')
    ax.legend(title='Label', labels=['Positif', 'Negatif'])
    st.pyplot(fig)

elif page == "Explorer":
    st.title('Explorer')
    social_media = st.selectbox("Select Platform", ["Facebook", "Instagram", "X"])
    st.write(f'Exploring {social_media} data.')

elif page == "About":
    st.title('About')
    st.write('This is an about page.')
