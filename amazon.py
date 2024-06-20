import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Download NLTK data
nltk.download('stopwords')

# Define datasets
amazon_alexa_data = pd.read_csv("https://raw.githubusercontent.com/LUMINOXCS/THE_Sentaly./main/amazon_alexa%20(1).csv", delimiter='\t', quoting=3)
laptop_data = pd.read_csv("https://raw.githubusercontent.com/LUMINOXCS/THE_Sentaly./main/laptop.csv")
mobile_data = pd.read_csv("https://raw.githubusercontent.com/LUMINOXCS/THE_Sentaly./main/mobile%20dataset.csv")

# Set up the sidebar with main categories
st.sidebar.title("Menu")
main_option = st.sidebar.selectbox("Select a Category", ["Alexa", "Electronics"])

# Main panel content based on the selected main category
if main_option == "Alexa":
    st.markdown("<h1 style='text-align: center; font-size: 60px'>AMAZON ANALYSED PRESET</h1>", unsafe_allow_html=True)
    st.title("Alexa")
    data = amazon_alexa_data

elif main_option == "Electronics":
    st.title("Electronics")
    electronics_option = st.sidebar.selectbox("Select Electronics Category", ["Laptop", "Mobile"])

    if electronics_option == "Laptop":
        st.subheader("Laptop")
        data = laptop_data

    elif electronics_option == "Mobile":
        st.subheader("Mobile")
        data = mobile_data

# Display column selection option
if 'data' in locals():
    # Option to display the dataset
    st.subheader("Dataset Preview")
    num_rows = st.slider("Select number of rows to display", min_value=5, max_value=100, value=10, step=5)
    st.write(data.head(num_rows))
    
    column_option = st.selectbox("Select a Column for Sentiment Analysis", data.columns)

    if column_option:
        # Show graphs based on selected column
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Distribution Count of {column_option}")
            fig1, ax1 = plt.subplots()
            data[column_option].value_counts().plot.bar(color='orange', ax=ax1)
            ax1.set_title(f'{column_option} Distribution Count')
            ax1.set_xlabel(column_option)
            ax1.set_ylabel('Count')
            st.pyplot(fig1)

        with col2:
            st.subheader(f"Percentage-wise Distribution of {column_option}")
            fig2 = plt.figure(figsize=(7, 7))
            colors = sns.color_palette("husl", len(data[column_option].unique()))
            wp = {'linewidth': 1, "edgecolor": "black"}
            tags = data[column_option].value_counts() / data.shape[0]
            explode = [0.1] * len(tags)
            tags.plot(kind='pie', autopct="%1.1f%%", shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode, ax=plt.gca())
            plt.title(f'Percentage-wise Distribution of {column_option}')
            st.pyplot(fig2)
else:
    st.write("Select a category from the sidebar to get started.")
