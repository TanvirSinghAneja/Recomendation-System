import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv('my_movie_dataset.csv')
def recomendation(x,df,n):
  def low(x):
    '''
    Just to make selection case insensitive
    '''
    for a in df.columns:
      if a.lower()==x.lower():
        return a
    return None

  '''
  recommend similar items based on text content
  '''
  feat='Genre'
  # feat=input('Enter Feature Column = ')
  feat=low(feat)
  print('-'*50)
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.metrics.pairwise import cosine_similarity

  tf=TfidfVectorizer(stop_words='english')
  vec=tf.fit_transform(df[feat].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)))
  score=cosine_similarity(vec)

  def rec(similar):
    similar=similar.lower()
    title='Title'
    # title=input('Enter what to search = ')
    print('-'*50)
    title=low(title)
    ind_low=df[title].str.lower()
    if similar in ind_low.values:
      ind=np.where(similar==ind_low)[0][0]
      sim=sorted(list(enumerate(score[ind])),reverse=True,key=lambda q:q[1])[1:]

      print(f'Recommendation for {x}')
      print('-'*50)

      seen_names = set()
      count = 0
      for i in sim:
        name = df.iloc[i[0]][title]
        if name not in seen_names:
          print(name)
          seen_names.add(name)
          count += 1
        if count == n:
          break

      print('-'*50)
    else:
        print(f'{x} is not in data')
  rec(x)

# Usage

# description-product
# recomendation('naruto',df,5)
import streamlit as st

st.title('Movie Recomendations')

input=st.text_area('Enter Movie/ Show Name.....')
n=st.text_area('How Many Recomend ......')

recomendation(input,df,n)
