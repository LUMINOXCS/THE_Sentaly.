import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from st_aggrid import AgGrid
import plotly.express as px
import plotly.graph_objs as go
import stemgraphic

def create_stem_and_leaf_plot(data):
    st.subheader('Stem-and-Leaf Plot')
    st.write("Creating the Stem-and-Leaf plot...")
    fig, ax = stemgraphic.stem_graphic(data)
    st.pyplot(fig)

st.set_page_config(page_title="Sentaly !", layout="wide")

# Styling the title
st.write("""
<style>
.centered {
  text-align: center;
  font-family: Changa One;
  font-size: 135px;
  color: white;
  font-weight: bolder;
}
</style>
<div class="centered">
  SENTALY.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='text-align: center; color: Gray;'>ANALYSER TOOLBAR</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='text-align: center; color: white;'>Select The Way You Want To Analyse:</h1>", unsafe_allow_html=True)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment score using VADER
def score_vader(text):
    if not isinstance(text, str):
        text = str(text)
    vs = analyzer.polarity_scores(text)
    return vs['compound']

def analyze(polarity):
    if polarity >= 0.5:
        return 'Positive'
    elif polarity <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

def sentiment_to_star_rating(score):
    if score >= 0.5:
        return 5
    elif score >= 0.2:
        return 4
    elif score >= -0.2:
        return 3
    elif score >= -0.5:
        return 2
    else:
        return 1

def star_rating_html(rating):
    stars = "★" * rating + "☆" * (5 - rating)
    return f'{stars}'

# Main function
def main(options):
    with st.expander('Analyze Text'):
        text = st.text_input('Text here: ')
        if text:
            vs = analyzer.polarity_scores(text)
            st.write('Positive: ', round(vs['pos'], 1))
            st.write('Neutral: ', round(vs['neu'], 0))  
            st.write('Negative: ', round(vs['neg'], -1))

    uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        all_data = []
        for uploaded_file in uploaded_files:
            try:
                data = pd.read_csv(uploaded_file, encoding='ISO-8859-1', on_bad_lines='skip')
                all_data.append(data)
                AgGrid(data, editable=True, height=400, fit_columns_on_grid_load=True)
                columns = data.columns.tolist()
                column_to_analyze = st.selectbox(f'Select the column to analyze for {uploaded_file.name}:', columns)
            except pd.errors.ParserError:
                st.error(f"Error reading {uploaded_file.name}. Please ensure it is correctly formatted.")
                return

        if st.button("Analyze Sentiments"):
            results = []
            for i, data in enumerate(all_data):
                if column_to_analyze in data.columns:
                    # Apply the analysis only to non-null entries
                    data["Sentiment Score"] = data[column_to_analyze].dropna().apply(score_vader)
                    data["Sentiment"] = data["Sentiment Score"].apply(analyze)
                    data["Star Rating"] = data["Sentiment Score"].apply(sentiment_to_star_rating)
                    data["Star Rating HTML"] = data["Star Rating"].apply(star_rating_html)

                    st.write(f"Analysis for {uploaded_files[i].name}")
                    st.write(data)

                    avg_sentiment_score = data["Sentiment Score"].mean()
                    avg_star_rating = data["Star Rating"].mean()

                    results.append({
                        'Dataset': uploaded_files[i].name,
                        'Average Sentiment Score': avg_sentiment_score,
                        'Average Star Rating': avg_star_rating
                    })

            results_df = pd.DataFrame(results)
            st.write(results_df)

            # Compute sentiment counts for combined data
            combined_data = pd.concat(all_data, ignore_index=True)
            sentiment_counts = combined_data['Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']

            # Visualizations based on the combined data
            if options == 'Home':
                pass  # Do nothing for Home
            
            elif options == 'Data Summary':
                st.subheader('Data Summary')
                st.write(combined_data.describe())
                st.write(combined_data.info())
                
            elif options == 'Pictograph':
                st.subheader('Pictograph')
                symbol_map = {'Positive':'star', 'Neutral':'circle','Negative':'square'}
                combined_data['Symbol'] = combined_data['Sentiment'].map(symbol_map)
                fig = px.scatter(combined_data, x=combined_data.index, y=combined_data['Sentiment Score'], symbol='Symbol',
                                 color='Sentiment',
                                 title='Sentiment Analysis Pictograph',
                                 labels={'x': 'Index', 'Sentiment Score': 'Sentiment Score'},
                                 size_max=20)  # Set a fixed size for all symbols/icons
                fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
                fig.update_layout(width=1500, height=900)
                st.plotly_chart(fig, use_container_width=True)

            elif options == 'Bar Graph':
                st.subheader('Bar Graph')
                bar_data = combined_data['Sentiment'].value_counts().reset_index()
                bar_data.columns = ['Sentiment', 'Count']
                color_map = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
                fig = px.bar(bar_data, x='Sentiment', y='Count', 
                             title='Sentiment Analysis Bar Graph',
                             labels={'Sentiment': 'Sentiment', 'Count': 'Count'},
                             color='Sentiment',
                             color_discrete_map=color_map)
                fig.update_layout(width=1000, height=700)
                st.plotly_chart(fig, use_container_width=True)

            elif options == 'Pie Chart':
                st.subheader('Pie Chart')
                fig = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                             title='Sentiment Analysis Pie Chart')
                fig.update_layout(width=1000, height=700)
                st.plotly_chart(fig, use_container_width=True)

            elif options == 'Donut Chart':
                st.subheader('Donut Chart')
                fig = px.pie(sentiment_counts, values='Count', names='Sentiment',
                             title='Sentiment Analysis Donut Chart', hole=0.5)
                fig.update_layout(width=1000, height=700)
                st.plotly_chart(fig, use_container_width=True)

            elif options == 'Scatter Plot':
                st.subheader('Scatter Plot')
                fig = px.scatter(combined_data, x=combined_data.index, y='Sentiment Score', color='Sentiment',
                                 title='Sentiment Analysis Scatter Plot',
                                 labels={'x': 'Index', 'Sentiment Score': 'Sentiment Score'})
                fig.update_layout(width=1100, height=900)
                st.plotly_chart(fig, use_container_width=True)

            elif options == 'Interactive Plot':
                st.subheader('Interactive Plot')
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data['Sentiment Score'], mode='lines+markers'))
                fig.update_layout(title='Sentiment Analysis Interactive Plot',
                                  xaxis_title='Index',
                                  yaxis_title='Sentiment Score')
                fig.update_layout(width=1600, height=900)  # Adjust width and height
                st.plotly_chart(fig)

            elif options == 'Box Plot':
                st.subheader('Box Plot')
                fig = px.box(combined_data, y='Sentiment Score', color='Sentiment',
                             title='Sentiment Analysis Box Plot',
                             labels={'Sentiment Score': 'Sentiment Score'})
                fig.update_layout(width=1500, height=900)
                st.plotly_chart(fig, use_container_width=True)

            elif options == 'Histogram':
                st.subheader('Histogram')
                fig = px.histogram(combined_data, x='Sentiment Score', color='Sentiment',
                                   title='Sentiment Analysis Histogram',
                                   labels={'Sentiment Score': 'Sentiment Score'})
                fig.update_layout(width=1000, height=900)
                st.plotly_chart(fig, use_container_width=True)

            elif options == 'Stem-and-Leaf Plot':
                 create_stem_and_leaf_plot(combined_data['Sentiment Score'])

            elif options == 'Frequency Polygon':
                st.subheader('Frequency Polygon')
                fig = px.histogram(combined_data, x='Sentiment Score', nbins=10, marginal='rug', histnorm='probability density',
                                   title='Frequency Polygon of Sentiment Scores')
                fig.update_traces(marker=dict(color='rgb(158,202,225)', line=dict(color='rgb(8,48,107)', width=1.5)))
                fig.update_layout(barmode='overlay')  # Set barmode to overlay for frequency polygon
                fig.update_layout(width=1000, height=900)
                st.plotly_chart(fig, use_container_width=True)

            elif options == 'Pareto Chart':
                st.subheader('Pareto Chart')
                sentiment_counts = combined_data['Sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                sentiment_counts = sentiment_counts.sort_values(by='Count', ascending=False)
                sentiment_counts['Cumulative Percentage'] = (sentiment_counts['Count'].cumsum() / sentiment_counts['Count'].sum()) * 100
                fig = go.Figure()
                fig.add_trace(go.Bar(x=sentiment_counts['Sentiment'], y=sentiment_counts['Count'], name='Frequency', marker_color='rgb(158,202,225)'))
                fig.add_trace(go.Scatter(x=sentiment_counts['Sentiment'], y=sentiment_counts['Cumulative Percentage'], name='Cumulative Percentage', 
                                         mode='lines+markers', yaxis='y2', marker_color='rgb(255,127,14)'))
                fig.update_layout(title='Pareto Chart of Sentiment Analysis',
                                  xaxis=dict(title='Sentiment'),
                                  yaxis=dict(title='Frequency'),
                                  yaxis2=dict(title='Cumulative Percentage', overlaying='y', side='right', showgrid=False),
                                  width=1500, height=900)
                st.plotly_chart(fig, use_container_width=True)
            
            elif options == 'Rating Chart':
                st.subheader('Rating Chart')

                # Calculate average star rating
                avg_rating = combined_data['Star Rating'].mean()
                avg_star_html = star_rating_html(round(avg_rating))

                st.markdown(f"<h3>Average Rating: {avg_star_html}</h3>", unsafe_allow_html=True)

                # Display individual star ratings as text
                st.markdown(combined_data[['Star Rating HTML']].to_html(escape=False), unsafe_allow_html=True)

# Sidebar navigation for different sections
sections = st.sidebar.selectbox('Select Analysis Section:', ['Home', 'Data Summary', 'Pictograph', 'Bar Graph', 'Pie Chart', 'Donut Chart', 'Scatter Plot', 'Interactive Plot', 'Box Plot', 'Histogram', 'Stem-and-Leaf Plot', 'Frequency Polygon', 'Pareto Chart', 'Rating Chart'])
main(sections)
