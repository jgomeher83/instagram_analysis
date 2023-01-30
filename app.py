import streamlit as st # web development
import numpy as np # np mean, np random
import pandas as pd # read csv, df manipulation
import time # to simulate a real time data, time loop
import plotly.express as px # interactive charts
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

st.set_page_config(
    page_title = 'Instagram Analysis',
    page_icon = '✅',
    layout = 'wide',
)
st.title("Instagram Analysis")

df = pd.read_csv('ig_data.csv', encoding = 'latin1')
describe = df.describe()

text0 = " ".join(i for i in df.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text0)
wordcloud.to_file('caption.png')
text2 = " ".join(i for i in df.Hashtags)
stopwords2 = set(STOPWORDS)
wordcloud2 = WordCloud(stopwords=stopwords2, background_color="white").generate(text2)
wordcloud2.to_file('Hashtags.png')

captions = df["Caption"].tolist()
uni_tfidf = text.TfidfVectorizer(input=captions, stop_words="english")
uni_matrix = uni_tfidf.fit_transform(captions)
uni_sim = cosine_similarity(uni_matrix)
def recommend_post(x):
  return ", ".join(df["Caption"].loc[x.argsort()[-5:-1]])

df["Recommended Post"] = [recommend_post(x) for x in uni_sim]

# Correlation
df_corr = df.corr().round(1)  
# Mask to matrix
mask = np.zeros_like(df_corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
# Viz
df_corr_viz = df_corr.mask(mask).dropna(how='all').dropna('columns', how='all')

home = df["From Home"].sum()
hashtags = df["From Hashtags"].sum()
explore = df["From Explore"].sum()
other = df["From Other"].sum()
labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

placeholder = st.empty()

# while True:
with placeholder.container():
    st.header('The main idea of this streamlit app is to make a description of an Instragram account data, looking the distribution of each feature, see the tendencies between the realtionships of the most important sources of impressions and finally using sickit learn, show the most uses words in the caption and hashtags of every post to create a recomendation system.')
    st.info('source code: https://github.com/jgomeher83/instagram_analysis') # Blue Colored Info message
    st.markdown("### Description of the data")
    st.info('Here you can see that the data contains information about 119 Instagram posts. The variables or features are segmented by source of each publication, saves, comments, shares, likes, profile views, follows, hashtags and captions') # Blue Colored Info message
    st.dataframe(df.head(5), width=1500)
    st.dataframe(describe, width=1500)
    st.subheader('Correlation chart and Sources from Impressions Pie Chart')
    
    correlation, pie = st.columns(2)
    with correlation:
        st.markdown("#### Correlation chart")
        fig = px.imshow(df_corr_viz, text_auto=True)
        st.write(fig)
    with pie:
        st.markdown("#### Pie chart")
        piee = px.pie(df, values=values, names=labels)
        st.write(piee)
    st.info('Here you can see the correlation between the data and the pie chart of sources that shows which generates the most attraction') # Orange 


    
    st.subheader('Histograms of discrete variables')
    fig_time, fig_time_ac = st.columns(2)
    with fig_time:
        st.markdown("#### From Home")
        fig_time_0 = px.histogram(df, x="From Home")
        st.write(fig_time_0)
    with fig_time_ac:
        st.markdown("#### From Hashtags")
        fig_time_acc = px.histogram(df, x="From Hashtags")
        st.write(fig_time_acc)
    st.info('Here you can see a comparison between the distributions of impressions of each post that cames from Home and Hashtags. The right chart source shows more traffic indicading that is the best way of generate impressions.') # Orange 
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("#### From Explore")
        fig = px.histogram(df, x="From Explore")
        st.write(fig)
    # st.dataframe(df_gate_total)
    with fig_col2:
        st.markdown("#### From Other")
        fig2 = px.histogram(df, x="From Other")
        st.write(fig2)
    st.info('Here you can see that the impressions generated from Explore and Others are very low in comparison with Home and Hashtags') # Orange 

    st.subheader('Comparison between total impression of each post and each source to see tendencies.')
    fig_col5, fig_col6 = st.columns(2)
    with fig_col5:
        st.markdown("#### Relationship Between Likes and Impressions")
        fig5 = px.scatter(df, x="Impressions",y="Likes", size="Likes", trendline="ols")
        st.write(fig5)
    # st.dataframe(df_gate_total)
    with fig_col6:
        st.markdown("#### Relationship Between Comments and Impressions")
        fig6 = px.scatter(df, x="Impressions",y="Comments", size="Comments", trendline="ols")
        st.write(fig6)
    st.info('Here you can see that the first chart, shows a positive tendency between the Likes and Impressions. The second chart shows that the number of comments does not seem to be important when it comes to generating more impressions according to the trend line.') # Orange 

    fig_col7, fig_col8 = st.columns(2)
    with fig_col7:
        st.markdown("#### Relationship Between Shares and Impressions")
        fig7 = px.scatter(df, x="Impressions",y="Shares", size="Shares", trendline="ols")
        st.write(fig7)
    # st.dataframe(df_gate_total)
    with fig_col8:
        st.markdown("#### Relationship Between Saves and Impressions")
        fig8 = px.scatter(df, x="Impressions",y="Saves", size="Saves", trendline="ols")
        st.write(fig8)
    
    st.info('Here you can see that this two charts shows a positive relationship according to the trend line, and at the same time, that when people save a post means that the post can generate more impressions because was interesting for them.') # Orange 
    

    st.subheader('Most used words on Captions and Hashtags')
    fig_col3, fig_col4 = st.columns(2)
    with fig_col3:
        st.markdown("#### Words from Captions")
        st.image('caption.png',use_column_width=True)
    # st.dataframe(df_gate_total)
    with fig_col4:
        st.markdown("#### Words from Hashtags")
        st.image('Hashtags.png', use_column_width=True)

    st.info('Here using worcloud we can see the most used words in Captions and Hashtags.') # Orange 


    st.subheader('Recomendation System using the captions of each post and the library sickit-learn')
    st.image('sklearn_1.png',use_column_width=True)

    st.info('Cosine Similarity in Machine Learning Cosine similarity is used to find similarities between the two documents. It does this by calculating the similarity score between the vectors, which is done by finding the angles between them. The range of similarities is between 0 and 1. If the value of the similarity score between two vectors is 1, it means that there is a greater similarity between the two vectors.On the other hand, if the value of the similarity score between two vectors is 0, it means that there is no similarity between the two vectors. When the similarity score is one, the angle between two vectors is 0 and when the similarity score is 0, the angle between two vectors is 90 degrees.') # Orange Colored warning message
    
    st.markdown("#### Examples of recomended captions")
    st.success(df["Recommended Post"][0]) # Green colored success message
    st.warning(df["Recommended Post"][1]) # Orange Colored warning message
    st.info(df["Recommended Post"][2]) # Blue Colored Info message
    st.error(df["Recommended Post"][3]) # Red Colored error message

    # st.markdown("### Detailed Data View")
    # st.dataframe(df_trajectories)
    time.sleep(1)
#placeholder.empty()






