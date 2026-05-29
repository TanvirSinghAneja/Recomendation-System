import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

df = pd.read_csv('my_movie_dataset.csv')

def recomendation(x, df, n):
    def low(x):
        for a in df.columns:
            if a.lower() == x.lower():
                return a
        return None

    feat = 'Genre'
    feat = low(feat)
    
    tf = TfidfVectorizer(stop_words='english')
    vec = tf.fit_transform(df[feat].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)))
    score = cosine_similarity(vec)

    def rec(similar):
        similar = similar.lower()
        title = 'Title'
        title = low(title)
        ind_low = df[title].str.lower()
        
        if similar in ind_low.values:
            ind = np.where(similar == ind_low)[0][0]
            sim = sorted(list(enumerate(score[ind])), reverse=True, key=lambda q: q[1])[1:]

            st.write(f'### Recommendation for {x}')
            
            seen_names = set()
            count = 0
            for i in sim:
                name = df.iloc[i[0]][title]
                if name not in seen_names:
                    st.write(f"- {name}")
                    seen_names.add(name)
                    count += 1
                if count == int(n):
                    break
        else:
            st.write(f'*{x} is not in data*')
            
    rec(x)

st.title('Movie Recommendations')

movie_input = st.text_input('Enter Movie/ Show Name.....')
num_rec = st.text_input('How Many Recommendations ......', value="5")

if movie_input:
    try:
        n_val = int(num_rec)
        recomendation(movie_input, df, n_val)
    except ValueError:
        st.error("Please enter a valid number for recommendations.")
