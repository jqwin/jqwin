import pandas as pd
from tqdm.notebook import tqdm
import snscrape.modules.twitter as sntwitter
import spacy
import pytz
from spacytextblob.spacytextblob import SpacyTextBlob
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')
tweets=[]


scraper = sntwitter.TwitterSearchScraper('wemby lang:en since:2023-06-22 until:2023-06-23')
for i, tweet in enumerate(scraper.get_items()):
    data =[tweet.date, tweet.id, tweet.rawContent, tweet.user.username, tweet.likeCount]
    tweets.append(data)
    if i > 10:
        break
        
tweet_df = pd.DataFrame(tweets, columns=['date', 'id', 'content', 'username', 'like_count'])
tweet_df['date'] = pd.to_datetime(tweet_df['date']).dt.tz_convert('MST')
tweet_df['time'] = tweet_df['date'].dt.tz_convert('MST').dt.tz_convert(pytz.timezone('America/Denver')).dt.time
tweet_df['date'] = tweet_df['date'].dt.date

sentiments=[]
for tweet_content in tweet_df['content']:
    sentiment = nlp(tweet_content)._.polarity
    sentiments.append(sentiment)
    
tweet_df['sentiment'] = sentiments

tweet_df = tweet_df.sort_values(by=['date', 'time'])

tweet_df.to_csv('python_nba_tweets_wemby_2.csv', index=False)