import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from apify_client import ApifyClient
import base64

st.set_page_config(page_title="Instagram Comment Sentiment Analyzer", layout="centered")
st.title("📊 Instagram Comment Sentiment Analyzer")

post_url = st.text_input("📥 Paste Instagram Post/Reel Link")

if st.button("🚀 Run Sentiment Analysis") and post_url:
    with st.spinner("Extracting comments..."):
        APIFY_TOKEN = "apify_api_qluL57quFZRFyhRCptedkDlrOtUvaW09WzdU"
        client = ApifyClient(APIFY_TOKEN)

        run_input = {
            "directUrls": [post_url],
            "resultsType": "comments",
            "resultsLimit": 100,
            "addParentData": False
        }

        run = client.actor("shu8hvrXbJbY3Eb9W").call(run_input=run_input)

        comments = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            if "text" in item:
                comments.append(item["text"])

        df = pd.DataFrame(comments, columns=["Comment"])

        analyzer = SentimentIntensityAnalyzer()
        def analyze_sentiment(comment):
            score = analyzer.polarity_scores(str(comment))
            compound = score['compound']
            if compound >= 0.05:
                sentiment = "Positive"
            elif compound <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            return pd.Series([compound, sentiment])
        df[['Compound Score', 'Sentiment']] = df['Comment'].apply(analyze_sentiment)

        positive_df = df[df["Sentiment"] == "Positive"][["Comment"]].rename(columns={"Comment": "Positive Comments"})
        negative_df = df[df["Sentiment"] == "Negative"][["Comment"]].rename(columns={"Comment": "Negative Comments"})
        neutral_df = df[df["Sentiment"] == "Neutral"][["Comment"]].rename(columns={"Comment": "Neutral Comments"})

        positive_df.reset_index(drop=True, inplace=True)
        negative_df.reset_index(drop=True, inplace=True)
        neutral_df.reset_index(drop=True, inplace=True)

        segregated_df = pd.concat([positive_df, negative_df, neutral_df], axis=1)

        total_comments = len(df)
        positive_count = len(positive_df)
        negative_count = len(negative_df)
        neutral_count = len(neutral_df)

        st.subheader("📊 Sentiment Summary")
        st.write(f"**Total Comments:** {total_comments}")
        st.write(f"✅ Positive: {positive_count} | ❌ Negative: {negative_count} | 😐 Neutral: {neutral_count}")

        st.subheader("🧁 Sentiment Distribution")
        fig, ax = plt.subplots()
        ax.pie(
            [positive_count, negative_count, neutral_count],
            labels=['Positive', 'Negative', 'Neutral'],
            colors=['#2ecc71', '#e74c3c', '#f1c40f'],
            autopct='%1.1f%%',
            startangle=140
        )
        ax.axis('equal')
        st.pyplot(fig)

        def get_csv_download_link(df, filename, link_text):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
            return href

        st.markdown("### 📁 Download CSVs")
        st.markdown(get_csv_download_link(df[['Comment']], 'all_comments.csv', '📥 Download Extracted Comments'), unsafe_allow_html=True)
        st.markdown(get_csv_download_link(segregated_df, 'sentiment_segregated_comments.csv', '📥 Download Sentiment Analyzed Comments'), unsafe_allow_html=True)

        st.subheader("🔍 Preview of Comments by Sentiment")
        st.markdown("**😊 Positive Comments:**")
        st.dataframe(positive_df.head(5))
        st.markdown("**😐 Neutral Comments:**")
        st.dataframe(neutral_df.head(5))
        st.markdown("**😡 Negative Comments:**")
        st.dataframe(negative_df.head(5))
