import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

def classify_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def perform_sentiment_analysis(df):
    # Initialize the VADER sentiment intensity analyzer
    sid = SentimentIntensityAnalyzer()

    # Perform sentiment analysis on each review and add sentiment scores to the DataFrame
    df['compound'] = df['Review'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['neg'] = df['Review'].apply(lambda x: sid.polarity_scores(x)['neg'])
    df['neu'] = df['Review'].apply(lambda x: sid.polarity_scores(x)['neu'])
    df['pos'] = df['Review'].apply(lambda x: sid.polarity_scores(x)['pos'])

    # Classify sentiment based on the compound score
    df['sentiment'] = df['compound'].apply(classify_sentiment)

    return df

def plot_sentiment_distribution(df_result):
    # Plot a pie chart for sentiment distribution
    sentiment_counts = df_result['sentiment'].value_counts()
    total_reviews = len(df_result)

    # Plotting
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'red', 'blue'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f'Sentiment Distribution - Total Reviews: {total_reviews}')
    plt.show()

def main():
    # Load your dataset into a Pandas DataFrame
    # Update 'your_dataset.csv' with the actual file path
    df = pd.read_csv('engineering_colleges_reviews.csv')

    # Perform sentiment analysis
    df_result = perform_sentiment_analysis(df)

    # Display the result
    print(df_result)

    # Save the result to a new CSV file
    # Update 'sentiment_result.csv' with the desired output file name
    df_result.to_csv('sentiment_result.csv', index=False)

    # Plot sentiment distribution
    plot_sentiment_distribution(df_result)

if __name__ == "__main__":
    main()
