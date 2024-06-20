import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(r"https://raw.githubusercontent.com/LUMINOXCS/THE_Sentaly./main/tata%201%20mg.csv")

# Main function to run the app
def main():
    st.markdown("<h1 style='text-align: center;'>TATA 1mg Analysed Preset</h1>", unsafe_allow_html=True)

    # Load data
    data = load_data()
    
    # Sidebar for dataset preview
    st.sidebar.header("Dataset Preview")
    if st.sidebar.checkbox("Show dataset preview", True):
        num_rows = st.sidebar.slider("Select number of rows to display", min_value=5, max_value=100, value=10, step=5)
        st.write(data.head(num_rows))
    
    # Sidebar for column selection
    st.sidebar.header("Select Column for Visualization")
    selected_column = st.sidebar.selectbox("Column", options=data.columns)

    if selected_column:
        st.header(f"Visualization based on {selected_column}")
        
        col1, col2 = st.columns(2)
        
        # Bar chart
        with col1:
            st.subheader(f"Count of {selected_column.capitalize()}")
            bar_fig, bar_ax = plt.subplots()
            sns.countplot(x=selected_column, data=data, ax=bar_ax)
            bar_ax.set_xticklabels(bar_ax.get_xticklabels(), rotation=45)
            st.pyplot(bar_fig)
        
        # Pie chart
        with col2:
            st.subheader(f"Distribution of {selected_column.capitalize()}")
            pie_data = data[selected_column].value_counts()
            pie_fig, pie_ax = plt.subplots()
            pie_ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            pie_ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(pie_fig)

if __name__ == '__main__':
    main()
