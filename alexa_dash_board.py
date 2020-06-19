import streamlit as st
import pandas as pd
import re    # for regular expressions 
import nltk  # for text manipulation 
import string 
import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# for advanced visualizations
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff
from sklearn.model_selection import StratifiedKFold
from wordcloud import WordCloud

st.header("Alexa Dashboard")
st.sidebar.header("dashboard")
data = pd.read_csv('amazon_alexa.tsv',sep='\t')
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english')
words =cv.fit_transform(data.verified_reviews)
sum_words = words.sum(axis=0)
words_freq =[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq,key = lambda x:x[1],reverse=True)
word_cloud = WordCloud(width=800,height=400,random_state=21,max_font_size=110).generate_from_frequencies(dict(words_freq))

def most_frequently_occuring_word():
    frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
    plt.style.use('fivethirtyeight')
    color = plt.cm.ocean(np.linspace(0, 1, 20))
    frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 6), color = color)
    plt.title("Most Frequently Occuring Words - Top 20")
    st.pyplot()
def show_alexa_rating():
    ratings = data['rating'].value_counts()
    label_rating = ratings.index
    size_rating = ratings.values
    colors = ['pink', 'lightblue', 'aqua', 'gold', 'crimson']
    rating_piechart = go.Pie(labels = label_rating,
                         values = size_rating,
                         marker = dict(colors = colors),
                         name = 'Alexa', hole = 0.3)

    df = [rating_piechart]

    layout = go.Layout(
           title = 'Distribution of Ratings for Alexa')

    fig = go.Figure(data = df,
                 layout = layout)
    st.plotly_chart(fig)
def alexa_variation():
    data['variation'].value_counts().plot.bar(figsize=(15,6))
    plt.title("Distribution of Alexa's variation")
    plt.xlabel("variations")
    plt.ylabel("count")
    st.pyplot()
def alexa_sentiment():
    feedback = data['feedback'].value_counts()
    label_rating = feedback.index
    size_rating = feedback.values
    colors = ['red', 'blue']
    rating_piechart = go.Pie(labels = label_rating,
                         values = size_rating,
                         marker = dict(colors = colors),
                         name = 'Alexa', hole = 0.3)
    df = [rating_piechart]
    layout = go.Layout(
           title = 'Distribution of Sentiments for Alexa')
    fig = go.Figure(data = df,
                 layout = layout)
    st.plotly_chart(fig)
def variation_versus_rating():
    plt.rcParams['figure.figsize'] = (15, 9)
    sns.boxenplot(x='variation',y='rating',data =data,palette="spring")
    plt.title("Variation vs Rating")
    plt.xticks(rotation=90)
    st.pyplot()

def feedback_wise_mean_rating():
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.style.use('fivethirtyeight')
    sns.violinplot(data['feedback'], data['rating'], palette = 'cool')
    plt.title("feedback wise Mean Ratings")
    st.pyplot()
def show_word_cloud():
    plt.figure(figsize=(20,10))
    plt.imshow(word_cloud,interpolation="bilinear")
    plt.axis("off")
    st.pyplot()
if st.sidebar.checkbox("show alexas rating",False):
    show_alexa_rating()
if st.sidebar.checkbox("Show amazons variations",False):
    alexa_variation()
if st.sidebar.checkbox("Sentiment of alexa",False):
    alexa_sentiment()
if st.sidebar.checkbox("Variation vs Rating",False):
    variation_versus_rating()
if st.sidebar.checkbox("Feedback wise mean rating",False):
    feedback_wise_mean_rating()
if st.sidebar.checkbox("Show most frequently occuring word",False):
    most_frequently_occuring_word()
if st.sidebar.checkbox("Show word cloud",False):
    show_word_cloud()
    

