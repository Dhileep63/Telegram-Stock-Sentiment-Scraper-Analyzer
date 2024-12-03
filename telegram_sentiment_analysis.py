from textblob import TextBlob

# Function to perform sentiment analysis
# Function to perform sentiment analysis
def analyze_sentiment(text):
    if not text:
        return 0  # Neutral sentiment for empty or None text
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # -1 to 1 scale (negative to positive sentiment)
    return sentiment