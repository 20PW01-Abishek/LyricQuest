import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def build_tf_idf(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df)
    vocabulary = vectorizer.get_feature_names_out()
    idf = vectorizer.idf_
    tfidf = pd.DataFrame(X.toarray(), columns=vocabulary)
    return vocabulary, idf, tfidf

df = pd.read_csv('./irpackage.csv')
df.dropna(inplace=True)
df.set_index('Unnamed: 0', inplace=True)

lyrics_vocabulary, lyrics_idf, tfidf_lyrics_df = build_tf_idf(df['lyrics'])
title_vocabulary, title_idf, tfidf_title_df = build_tf_idf(df['title'])

# Serialize the precomputed data
with open('utils/precomputed_data.pkl', 'wb') as f:
    pickle.dump((lyrics_vocabulary, lyrics_idf, tfidf_lyrics_df, title_vocabulary, title_idf, tfidf_title_df), f)