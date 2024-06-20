import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    # Replace with your dataset path
    data = pd.read_csv(r"https://raw.githubusercontent.com/LUMINOXCS/THE_Sentaly./main/zomato_reviews.csv")
    return data

# Function to perform sentiment analysis
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

# Streamlit app
st.markdown("<h1 style='text-align: center;'>Zomato Analysed Preset</h1>", unsafe_allow_html=True)


# Load data
data = load_data()

# Display dataset
st.write("Dataset Preview:")
st.write(data.head())

# Column selection
columns = data.columns.tolist()
selected_column = st.selectbox('Select a column for sentiment analysis', columns)

if selected_column:
    with st.spinner('Please wait...'):
        st.write(f'Sentiment analysis for: {selected_column}')
        
        # Apply sentiment analysis
        data['sentiment'] = data[selected_column].apply(lambda x: analyze_sentiment(str(x))['compound'])
        
        # Categorize sentiment scores
        data['sentiment_category'] = pd.cut(data['sentiment'], bins=[-1, -0.05, 0.05, 1], labels=['Negative', 'Neutral', 'Positive'])
        
        # Sentiment distribution
        sentiment_counts = data['sentiment_category'].value_counts()

        # Create two columns for side-by-side layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Sentiment Scores Distribution')
            fig_bar, ax_bar = plt.subplots(figsize=(5, 3))
            sentiment_counts.plot(kind='bar', ax=ax_bar, color=['red', 'grey', 'green'])
            ax_bar.set_title('Sentiment Scores Distribution')
            ax_bar.set_xlabel('Sentiment Category')
            ax_bar.set_ylabel('Frequency')
            st.pyplot(fig_bar)
        
        with col2:
            st.subheader('Sentiment Scores Distribution')
            fig_pie, ax_pie = plt.subplots(figsize=(5, 3))
            sentiment_counts.plot(kind='pie', ax=ax_pie, autopct='%1.1f%%', startangle=90, colors=['red', 'grey', 'green'])
            ax_pie.set_ylabel('')
            ax_pie.set_title('Sentiment Scores Distribution')
            st.pyplot(fig_pie)