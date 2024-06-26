import streamlit as st
import pandas as pd

def recommendations():
# Data kata kunci
    kata_kunci_data = {
        "Kata Kunci 1": ["judi", "online", "main", "situs", "terpercaya", "slot", "game", "menang", "agen"],
        "Kata Kunci 2": ["bola", "bonus", "daftar", "terbaik", "deposit", "link", "toto", "gelap", "poker"],
        "Kata Kunci 3": ["besar", "prediksi", "bandar", "indonesia", "gacor", "minimal", "casino", "pulsa", "togel"]
    }

    # Data bigram
    bigram_data = {
        "Bigram 1": ["judi online", "situs judi", "main judi", "online terpercaya", "judi slot", "judi bola", "agen judi", "toto gelap", "bandar judi", "di indonesia"],
        "Bigram 2": ["slot online", "game judi", "dan terpercaya", "online terbaik", "poker online", "di situs", "main game", "online yang", "slot gacor", "judi poker"],
        "Bigram 3": ["terpercaya di", "minimal deposit", "hanya di", "casino online", "menang judi", "percaya dan", "deposit pulsa", "terbaik dan", "judi togel", "main di"]
    }

    # Membuat DataFrame
    kata_kunci_df = pd.DataFrame(kata_kunci_data)
    bigram_df = pd.DataFrame(bigram_data)

    # Expander untuk kata kunci
    with st.expander("Lihat rekomendasi kata kunci dan bigram"):
        st.subheader("Kata Kunci")
        st.write("Kata kunci berikut dapat Anda kombinasikan dengan kata kunci lainnya:")
        st.dataframe(kata_kunci_df, height=300, use_container_width=True, hide_index=True)
        st.subheader("Bigram")
        st.write("Bigram berikut dapat Anda gunakan untuk mencari kata kunci yang lebih spesifik:")
        st.dataframe(bigram_df, height=300, use_container_width=True, hide_index=True)