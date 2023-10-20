import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

tfidf = pickle.load(open('F:\\class\\Machine Learning\\project\\spam message detection\\vectorizer.pkl', 'rb'), encoding='utf-8')
model = pickle.load(open('F:\\class\\Machine Learning\\project\\spam message detection\\model.pkl', 'rb'), encoding='utf-8')


st.title('Email/SMS Spam Classifier By Hakim')

input_sms = st.text_area('Enter The Message Or Mail')
if st.button('Predict'):

    transform_sms = transform_text(input_sms)

    vettor_input = tfidf.transform([transform_sms])

    result = model.predict(vettor_input)[0]

    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')