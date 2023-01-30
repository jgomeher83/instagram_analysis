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

st.set_page_config(
    page_title = 'Instagram Analysis',
    page_icon = '✅',
    layout = 'wide',
)
st.title("Instagram Analysis")

df = pd.read_csv('ig_data.csv', encoding = 'latin1')

text0 = " ".join(i for i in df.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text0)
wordcloud.to_file('caption.png')
text2 = " ".join(i for i in df.Hashtags)
stopwords2 = set(STOPWORDS)
wordcloud2 = WordCloud(stopwords=stopwords2, background_color="white").generate(text2)
wordcloud2.to_file('Hashtags.png')

#filters 
#selected_movement = st.sidebar.multiselect("Gate", df.line_name.unique(),df.line_name.unique())
#selected_clase = st.sidebar.multiselect("class", df.nomClass.unique(),df.nomClass.unique())

captions = df["Caption"].tolist()
uni_tfidf = text.TfidfVectorizer(input=captions, stop_words="english")
uni_matrix = uni_tfidf.fit_transform(captions)
uni_sim = cosine_similarity(uni_matrix)

def recommend_post(x):
  return ", ".join(df["Caption"].loc[x.argsort()[-5:-1]])

df["Recommended Post"] = [recommend_post(x) for x in uni_sim]

placeholder = st.empty()

# while True:
with placeholder.container():
  
    #lines charts
    fig_time, fig_time_ac = st.columns(2)
    with fig_time:
        st.markdown("### From Home")
        fig_time_0 = px.histogram(df, x="From Home")
        st.write(fig_time_0)
    with fig_time_ac:
        st.markdown("### From Hashtags")
        fig_time_acc = px.histogram(df, x="From Hashtags")
        st.write(fig_time_acc)
    
    # st.markdown("### DataFrame")
    # st.dataframe(df, width=2000, height=None)

    #bar charts
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("### From Explore")
        fig = px.histogram(df, x="From Explore")
        st.write(fig)
    # st.dataframe(df_gate_total)
    with fig_col2:
        st.markdown("### From Other")
        fig2 = px.histogram(df, x="From Other")
        st.write(fig2)
    # st.dataframe(df_class_total)

    fig_col3, fig_col4 = st.columns(2)
    with fig_col3:
        st.markdown("### Words from caption")
        st.image('caption.png',use_column_width=True)
    # st.dataframe(df_gate_total)
    with fig_col4:
        st.markdown("### Words from Hashtags")
        st.image('Hashtags.png', use_column_width=True)

    fig_col5, fig_col6 = st.columns(2)
    with fig_col5:
        st.markdown("### Relationship Between Likes and Impressions")
        fig5 = px.scatter(df, x="Impressions",y="Likes", size="Likes", trendline="ols")
        st.write(fig5)
    # st.dataframe(df_gate_total)
    with fig_col6:
        st.markdown("### Words from Hashtags")
        fig6 = px.scatter(df, x="Impressions",y="Comments", size="Comments", trendline="ols")
        st.write(fig6)

    fig_col7, fig_col8 = st.columns(2)
    with fig_col7:
        st.markdown("### Relationship Between Likes and Impressions")
        fig7 = px.scatter(df, x="Impressions",y="Shares", size="Shares", trendline="ols")
        st.write(fig7)
    # st.dataframe(df_gate_total)
    with fig_col8:
        st.markdown("### Words from Hashtags")
        fig8 = px.scatter(df, x="Impressions",y="Saves", size="Saves", trendline="ols")
        st.write(fig8)

    st.markdown("### Examples of recomended captions")
    st.success(df["Recommended Post"][0]) # Green colored success message
    st.warning(df["Recommended Post"][1]) # Orange Colored warning message
    st.info(df["Recommended Post"][2]) # Blue Colored Info message
    st.error(df["Recommended Post"][3]) # Red Colored error message

    # st.markdown("### Detailed Data View")
    # st.dataframe(df_trajectories)
    time.sleep(1)
#placeholder.empty()






