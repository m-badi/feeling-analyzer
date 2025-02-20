import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.express as px


def classify_sentiment_openai(review_text):
    client = OpenAI(api_key=openai_api_key)
    prompt = f'''
        Classify the following customer review. 
        State your answer
        as a single word, "positive", 
        "negative" or "neutral":

        {review_text}
        '''

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    ) 

    return completion.choices[0].message.content

# OpenAI API Key input
openai_api_key = st.sidebar.text_input(
    "Enter you own OpenAI API key", 
    type="password", 
    help="You can find your API key at https://platform.openai.com/account/api-keys"
)

st.title("🥗 Customer Review Sentiment Analyzer")
st.markdown("This app analyzes the sentiment of customer reviews to gain insights into their opinions.")

user_input = st.text_input("Enter customer review")
#csv file uploader
uploaded_file = st.file_uploader("Upload your csv file here", type=["CSV"], accept_multiple_files=False)


if uploaded_file!= None:
    reviews_df = pd.read_csv(uploaded_file)

    #check if text columns
    text_columns = reviews_df.select_dtypes(include="object").columns

    if len(text_columns)==0:
        st.error("No text columns found in the uploaded file")

    #show a dropdown menu to select which column to take
    review_colum = st.selectbox(
        "Select the column with the customer reviews",
        text_columns
    )

    #analyze the sentiment of the selected column
    reviews_df["sentiments"] = reviews_df[review_colum].apply(classify_sentiment_openai)
    reviews_df["sentiments"] = reviews_df["sentiments"].str.title()
    sentiment_counts = reviews_df["sentiments"].value_counts()

    st.write(reviews_df)    
    st.write(sentiment_counts)

    #create 3 columns to display the 3 metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        positive_count = sentiment_counts.get("Positive",0)
        st.metric("Positive", positive_count, f"{positive_count/len(reviews_df)*100:.2f}%")

    with col2:
        negative_count = sentiment_counts.get("Negative",0)
        st.metric("Negative", negative_count, f"{negative_count/len(reviews_df)*100:.2f}%")

    with col3:
        neutral_count = sentiment_counts.get("Neutral",0)
        st.metric("Neutral", neutral_count, f"{neutral_count/len(reviews_df)*100:.2f}%")

    fig = px.pie(
        values = sentiment_counts.values,
        names = sentiment_counts.index,
        title = 'Sentiment Distribution',
        color_discrete_sequence= ['#636EFA', '#EF553B', '#00CC96']
    )

    st.plotly_chart(fig)