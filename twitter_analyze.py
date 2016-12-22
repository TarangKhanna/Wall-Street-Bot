# TODO-get more tweets 
# TODO-tweets from reliable accounts
# TODO-take into account retweets,likes,time,followers
# analyze google searches to predict stock market
# remove tweets from other languages?
from __future__ import division
import tweepy
import shutil
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import pandas as pd
# import matplotlib.pyplot as plt
import csv 
from textblob import TextBlob
import numpy as np
# from pylab import *
import os.path

access_token = "301847288-lWXEQAwNc7kvyIF4E6w3TCzj7FfWYyUs2FKXbkcR"
access_token_secret = "dXv1ktTNVsHVHsx7AUyVilLOx3tEWPc0Ffi8BvSh9VN10"
consumer_key = "MyrxJJIAAbIupjvNbqyUTzJOZ"
consumer_secret = "ZBZrMl7jEv1DGt76hCV60K7j8Z8uDx8K710cO1w6SBelNVSeqD"
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

class twitter_analyze:

	def __init__(self):
		pass
		
	# current feelings about stock
	# Todo plot according to location
	def analyze_feelings(self, stock):
		
		# tweets_file = 'data/%s_tweets.csv' %stock

		# if not os.path.isfile(tweets_file) : 
		tweets = self.analyze_stock(stock)
		
		# tweets = pd.read_csv('data/%s_tweets.csv' %stock)

		sentiment = []
		for index, row in tweets.iterrows():
			value = 0.0
			if isinstance(row['polarity'], float):
				value = round(row['polarity'], 3)
			else:
				x = float(row['polarity'])
				value = round(x, 3)
			if value < 0.0:
				sentiment.append('negative')
			elif value == 0.0:
				sentiment.append('neutral')
			else:
				sentiment.append('positive')

		tweets['sentiment'] = sentiment
		# tweets['sentiment'].value_counts().plot(kind='bar')
		# tweets['sentiment'].value_counts().plot(kind='pie')
		# plt.show()
		print tweets
		counts_list = []
		print tweets['sentiment'].value_counts()['positive']
		counts_list.append(tweets['sentiment'].value_counts()['positive'])
		counts_list.append(tweets['sentiment'].value_counts()['negative'])
		counts_list.append(tweets['sentiment'].value_counts()['neutral'])

		# file_feelings = ('data/%s_feelings.csv' % stock)
		# cur_path = os.getcwd()
		# abs_path_feelings = cur_path+'/'+file_feelings
		# with open(file_feelings, "w") as output:
		#     writer = csv.writer(output, lineterminator='\n')
		#     for val in counts_list:
		#         writer.writerow([val])  

		return counts_list

	def analyze_stock(self, stock):
		all_tweets = self.get_tweets(stock)
		tweets = pd.DataFrame()
		analysis_list = []
		polarity_list = []
		subjectivity_list = []
		tweet_text = []
		tweet_dates = []
		for tweet in all_tweets:
			tweet_text.append(tweet.text.encode("utf-8"))
			analysis = TextBlob(tweet.text)
			# prints-Sentiment(polarity=0.0, subjectivity=0.0), polarity is how positive or negative, subjectivity is if opinion or fact
			# analysis_list.append('polarity:' + str(analysis.se 1ntiment.polarity) + ' subjectivity:' + str(analysis.sentiment.subjectivity))
			polarity_list.append(str(analysis.sentiment.polarity))
			subjectivity_list.append(str(analysis.sentiment.subjectivity))
			tweet_dates.append(tweet.created_at)

		tweets['text'] = np.array(tweet_text)
		# tweets['analysis'] = np.array(analysis_list)
		tweets['polarity'] = np.array(polarity_list)
		tweets['subjectivity'] = np.array(subjectivity_list)
		tweets['date'] = np.array(tweet_dates)
		# tweets = tweets.sort_values(by=['subjectivity'], ascending=0)
		print tweets
		# tweets.to_csv('data/%s_tweets.csv' % stock)
		return tweets

	def get_tweets(self, stock):
		alltweets = []  
		public_tweets = api.search(stock)
		alltweets.extend(public_tweets)
		oldest = alltweets[-1].id - 1

		# Todo date constraint?

		#keep grabbing tweets until there are no tweets left to grab
		while len(public_tweets) > 0:
		    print "getting tweets before %s" % (oldest)
		    # filter by users too, todo
		    public_tweets = api.search(stock,count=200,max_id=oldest)
		    
		    #save most recent tweets
		    alltweets.extend(public_tweets)
		    
		    #update the id of the oldest tweet less one
		    oldest = alltweets[-1].id - 1
		    
		    print "...%s tweets downloaded so far" % (len(alltweets))

		    if len(alltweets) > 500:
		    	break

		#transform the tweepy tweets into a 2D array that will populate the csv 
		outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in public_tweets]
		print outtweets
		return alltweets

if __name__ == "__main__":
	analyze = twitter_analyze()
	# analyze.analyze_stock('$AAPL')
	print analyze.analyze_feelings('$TSLA')
	# analyze.analyze_feelings('$AAPL')
	# analyze.analyze_feelings('$GOOGL')

