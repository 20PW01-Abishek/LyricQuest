import streamlit as st
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from services.youtube_crawler_service import youtube_crawler_service

def create_vote_buttons(i):
  col1, col2 = st.columns(2)

  with col1:
    if st.button(label="\U0001f44d", key=f"upvote_{i}", help="Up vote"):
      df.loc[i, 'vote'] += 1
      st.write("Up voted for item ", i)

  with col2:
    if st.button(label="\U0001f44e", key=f"downvote_{i}", help="Down vote"):
      df.loc[i, 'vote'] += -1
      st.write("Down voted for item ", i)

def build_tf_idf(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df)
    vocabulary = vectorizer.get_feature_names_out()
    idf = vectorizer.idf_
    tfidf = pd.DataFrame(X.toarray(), columns=vocabulary)
    return vocabulary, idf, tfidf

def tokenize(string):
    tokens = []
    ps = PorterStemmer()
    for word in string.split(' '):
        token = re.sub('[^a-zA-Z0-9]', '', word).lower()
        token = ps.stem(token)
        if token:
            tokens.append(token)
    lyrics = ' '.join(tokens)
    return lyrics

def create_query_df(query_dict, idf):
    df = pd.DataFrame([query_dict])
    df = df.div(df.sum(axis=1), axis=0)
    df = df.mul(idf, axis=1)
    df.fillna(0, inplace=True)
    return df

def modify_vector(df, i):
    return np.array(df.iloc[i]).reshape(1, -1)

df = pd.read_csv('./irpackage.csv')
df.dropna(inplace=True)
df['vote'] = 1
df.set_index('Unnamed: 0', inplace=True)

lyrics_vocabulary, lyrics_idf, tfidf_lyrics_df = build_tf_idf(df['lyrics'])
title_vocabulary, title_idf, tfidf_title_df = build_tf_idf(df['title'])

st.title("LyricQuest - A song search engine")

query = st.text_input('',placeholder='Enter phrases of lyrics')

alpha = st.slider("Select alpha value", 0.0, 1.0, 0.5)

num = 5

if st.button("Find the song"):
    query = tokenize(query)

    query_title_dict = dict.fromkeys(title_vocabulary, 0)
    query_lyrics_dict = dict.fromkeys(lyrics_vocabulary, 0)

    for term in query.split(' '):
        if term in query_title_dict:
            query_title_dict[term] += 1
        if term in query_lyrics_dict:
            query_lyrics_dict[term] += 1

    query_title_df = create_query_df(query_title_dict, title_idf)
    query_lyrics_df = create_query_df(query_lyrics_dict, lyrics_idf)

    query_lyrics_vector = modify_vector(query_lyrics_df, 0)
    query_title_vector = modify_vector(query_title_df, 0)

    similarity = {}

    for i in range(len(tfidf_lyrics_df)):
        lyric_vector = modify_vector(tfidf_lyrics_df, i)
        title_vector = modify_vector(tfidf_title_df, i)

        title_similarity = cosine_similarity(query_title_vector, title_vector)[0][0]
        lyric_similarity = cosine_similarity(query_lyrics_vector, lyric_vector)[0][0]

        similarity[i] = alpha * title_similarity + (1 - alpha) * lyric_similarity

    sorted_similarity = dict(sorted(similarity.items(), key=lambda item: -item[1]))

    for i in range(num):
        index, score = list(sorted_similarity.items())[i]
        if score == 0:
            if i == 0:
                st.write('There is no such lyrics found! please try again with some other lyric...')
            break
        cdf = df.iloc[index]
        st.write(cdf)
        st.markdown(youtube_crawler_service.GetYtVideo(cdf['title']+' '+cdf['movie']+' '+cdf['artist']), unsafe_allow_html=True)
        create_vote_buttons(i)