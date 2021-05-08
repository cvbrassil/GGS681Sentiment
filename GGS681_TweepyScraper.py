# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:57:01 2021

@author: connor.brassil
"""
####Imports
import tweepy
import json
import os

os.chdir('directory')

####Credentials
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)


####Parameters

raw_output = "filename.txt"
keywords = ['vaccine','vaccination','vaccinate',]
caplimit = 100000


####Stream listener

class MyStreamListener(tweepy.StreamListener):
    """
    Twitter listener, collects streaming tweets and output to a file
    """
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open(raw_output, "a")

    def on_status(self, status):
        tweet = status._json
        self.file.write( json.dumps(tweet) + '\n' )
        self.num_tweets += 1
        
        # Stops streaming when it reaches the limit
        if self.num_tweets <= caplimit:
            if self.num_tweets % 1000 == 0: # just to see some progress...
                print('Numer of tweets captured so far: {}'.format(self.num_tweets))
            return True
        else:
            print('stream complete')
            return False
        self.file.close()

    def on_error(self, status):
        print(status)
        
####Stream
# Initialize Stream listener
l = MyStreamListener()

# Create you Stream object with authentication
stream = tweepy.Stream(auth, l)

# Filter Twitter Streams to capture data by the keywords:
stream.filter(track=keywords)
