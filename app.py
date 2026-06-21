import re
import string
import pickle

import pandas as pd
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import streamlit as st

# ───────────────────────────────────────────────
# Download resource NLTK (sekali saja, otomatis di-skip jika sudah ada)
# ───────────────────────────────────────────────
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

st.header('Analisis Sentimen Berbasis Aspek Ulasan Restoran menggunakan Label Powerset dan XGBoost')

text_input = st.text_area("Masukkan Teks Ulasan")
submit = st.button("Submit", type="primary")


# ───────────────────────────────────────────────
# Fungsi-fungsi preprocessing (identik dengan Colab)
# ───────────────────────────────────────────────
def case_folding(text):
    return text.lower()


def remove_punctuation(text):
    data = re.sub('@[^\s]+', ' ', text)
    data = re.sub(r'http\S*', ' ', data)
    data = data.translate(str.maketrans(' ', ' ', string.punctuation))
    data = re.sub('[^a-zA-Z]', ' ', data)
    data = re.sub('\n', ' ', data)
    data = re.sub(r'\b[a-zA-z]\b', ' ', data)
    return data


def tokenize(text):
    return nltk.word_tokenize(text)


def remove_stopword(tokens):
    hasil = []
    for word in tokens:
        if word not in stopwords.words('indonesian'):
            hasil.append(word)
    return hasil


@st.cache_data
def load_kamus_normalisasi():
    kamus = pd.read_csv('model/colloquial-indonesian-lexicon.csv')
    kamus = kamus.drop(columns=['In-dictionary', 'context', 'category1', 'category2', 'category3'])
    return kamus


def normalization(tokens, kamus_normalisasi):
    hasil = []
    for kata in tokens:
        if kata in kamus_normalisasi['slang'].values:
            formal = kamus_normalisasi.loc[kamus_normalisasi['slang'] == kata, 'formal'].values[0]
            hasil.append(formal)
        else:
            hasil.append(kata)
    return hasil


@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()


def stemming(tokens, stemmer):
    hasil = []
    for word in tokens:
        hasil.append(stemmer.stem(word))
    return hasil


# ───────────────────────────────────────────────
# Load model, vectorizer, dan label encoder (sekali saja, di-cache)
# ───────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('model/model_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model/label_encoder_lp.pkl', 'rb') as f:
        le_lp = pickle.load(f)
    return model, vectorizer, le_lp


def decode_label_powerset(kode_str):
    """
    Mengubah string kode Label Powerset (mis. '0_0_0_1_0_1_0_0')
    menjadi nama label yang dapat dibaca (mis. 'makanan positif, pelayanan positif').

    Urutan digit mengikuti urutan kolom hasil MultiLabelBinarizer (mlb.classes_),
    SESUAIKAN list `nama_aspek` di bawah ini dengan urutan aktual mlb.classes_ Anda.
    """
    nama_aspek = [
        'harga negatif', 'harga positif',
        'makanan negatif', 'makanan positif',
        'pelayanan negatif', 'pelayanan positif',
        'tempat negatif', 'tempat positif',
    ]
    digit = kode_str.split('_')
    label_aktif = [nama_aspek[i] for i, d in enumerate(digit) if d == '1']
    if not label_aktif:
        return 'Tidak terdeteksi aspek/sentimen'
    return ', '.join(label_aktif)


# ───────────────────────────────────────────────
# Proses utama saat tombol submit ditekan
# ───────────────────────────────────────────────
if submit:
    if text_input:
        df_mentah = pd.DataFrame({'ulasan': [text_input]})

        with st.spinner('Memproses teks...'):
            kamus_normalisasi = load_kamus_normalisasi()
            stemmer = load_stemmer()

            df_mentah['Case Folding'] = df_mentah['ulasan'].apply(case_folding)
            df_mentah['Remove Punctuation'] = df_mentah['Case Folding'].apply(remove_punctuation)
            df_mentah['Tokenized'] = df_mentah['Remove Punctuation'].apply(tokenize)
            df_mentah['Stopword Removal'] = df_mentah['Tokenized'].apply(remove_stopword)
            df_mentah['Normalisasi'] = df_mentah['Stopword Removal'].apply(
                lambda x: normalization(x, kamus_normalisasi)
            )
            df_mentah['Stemming'] = df_mentah['Normalisasi'].apply(lambda x: stemming(x, stemmer))
            df_mentah['HasilProcessing'] = df_mentah['Stemming'].apply(lambda x: ' '.join(x))

            model, vectorizer, le_lp = load_artifacts()

            X_input = vectorizer.transform(df_mentah['HasilProcessing'])
            y_pred_enc = model.predict(X_input)
            y_pred_str = le_lp.inverse_transform(y_pred_enc)[0]

            hasil_label = decode_label_powerset(y_pred_str)

        st.success('Prediksi selesai!')
        st.write(f'Ulasan: **"{text_input}"**')
        st.write(f'Hasil Prediksi Aspek dan Sentimen: **{hasil_label}**')

        with st.expander('Lihat detail preprocessing'):
            st.write('Case Folding:', df_mentah['Case Folding'].iloc[0])
            st.write('Remove Punctuation:', df_mentah['Remove Punctuation'].iloc[0])
            st.write('Tokenized:', df_mentah['Tokenized'].iloc[0])
            st.write('Stopword Removal:', df_mentah['Stopword Removal'].iloc[0])
            st.write('Normalisasi:', df_mentah['Normalisasi'].iloc[0])
            st.write('Stemming:', df_mentah['Stemming'].iloc[0])
            st.write('Hasil Akhir:', df_mentah['HasilProcessing'].iloc[0])
    else:
        st.warning('Kamu belum memasukkan teks ulasan 😊')
