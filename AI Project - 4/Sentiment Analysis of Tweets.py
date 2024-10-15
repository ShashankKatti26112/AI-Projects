import tweepy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Twitter API credentials
api_key = 'your_api_key'
api_secret = 'your_api_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Set up tweepy
auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Fetch tweets
tweets = api.user_timeline(screen_name='@TwitterUser', count=100, tweet_mode='extended')

# Analyze sentiment
sia = SentimentIntensityAnalyzer()
for tweet in tweets:
    sentiment = sia.polarity_scores(tweet.full_text)
    print(f"Tweet: {tweet.full_text}")
    print(f"Sentiment: {sentiment}")
