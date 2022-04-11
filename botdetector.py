import botometer
import os
import glob
import pandas as pd
import sys
import json
from langdetect import detect

USE_PRO_API = False
USE_S2_ACCT = True

if USE_S2_ACCT: # Use S2 and pmb credentials for Sample API
    twitter_app_auth = {
        'consumer_key': '____________________________',
        'consumer_secret': '_____________________________',
        'access_token': '_____________________________',
        'access_token_secret': '_______________________'
    }
    # Sample API (17280/day, 180 per 15 minutes)
    rapidapi_key = "__________________________"
    bom = botometer.Botometer(wait_on_ratelimit=True,
                            rapidapi_key=rapidapi_key,
                            **twitter_app_auth)
elif USE_PRO_API:
    # App-only authentication
    twitter_app_auth = {
        'consumer_key': '_________________',
        'consumer_secret': '_____________________________',
    }
    # Pro API endpoint (17280/day, 450 per 15 minutes, costs $ for overage)
    botometer_api_url = 'https://botometer-pro.p.mashape.com'
    bom = botometer.Botometer(botometer_api_url=botometer_api_url,
                            mashape_key='e_____________________________',
                            **twitter_app_auth)
else: # Use SFM credentials on Sample API
    # Sample API (17280/day, 180 per 15 minutes)
    rapidapi_key = "____________________________"
    twitter_app_auth = {
        'consumer_key': '________________',
        'consumer_secret': '__________________________',
        'access_token': '_______________________________',
        'access_token_secret': '___________________________',
    }
    bom = botometer.Botometer(wait_on_ratelimit=True,
                            rapidapi_key=rapidapi_key,
                            **twitter_app_auth)

nv_fields = ["index", "ID", "Username", "Content", "CreatedTime", "TweetType", "ReTweetedBy", "NumberOfReTweets", "Hashtags", "Tagged", "Name", "Location", "Web", "Bio", "NumberOfTweets", "NumberOfFollowers", "NumberFollowing", "Coordinates"]
#nv_dtypes = ['str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'int', 'int', 'int', 'str']

source_path = sys.argv[1]

possible_bot_authors = set()
masked_bot_authors = set()
tweets_with_possible_bot_author = 0
protected_user_tweets = 0
total_tweets = 0
retweets = 0
original_tweets = 0

cache_dir = './bot_data'
output_dir = './debotted_tweets'

BOT_THRESHOLD = .5 # This also could be read in from cmd line

bot_probs = {}

tweets_seen = set()

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

if not os.path.isdir(cache_dir):
    os.mkdir(cache_dir)
else:
    for fn in os.listdir(cache_dir):
        if not os.path.isfile(os.path.join(cache_dir, fn)) or not fn.endswith('.json'):
            continue
        with open(os.path.join(cache_dir, fn), 'r') as f:
            bot_result = json.load(f)
        user_name = os.path.splitext(os.path.basename(fn))[0]
        if bot_result == 'null' or bot_result is None:
            masked_bot_authors.add(user_name)
        else:
            bot_probs[user_name] = bot_result

for file_path in glob.glob(source_path):
    if not os.path.isfile(file_path):
        print("Invalid input file path:", file_path)
        continue

    print("Checking",file_path)

    out_path = os.path.join(output_dir, os.path.basename(file_path).replace('.dat', '.nobots.tsv'))
    out_tweets = pd.DataFrame(columns = nv_fields)
    out_tweets.to_csv(out_path, sep="\t", mode='w', columns=nv_fields, header=True, index=False)

    with open(file_path, 'r') as tf:
        for t in tf:
            tvalues = t.strip().split("\t")

            try:
                tweet = dict(zip(nv_fields, tvalues))
            except:
                print("error parsing tweet, skipping:",t.strip())
                continue

            tweeter = tweet['Username'].strip()
            total_tweets += 1

            bot_result = None

            # If we've already seen a tweet before, and this is a retweet, don't include it
            print(tweet['TweetType'])
            tweet_content = tweet['Content'].strip()
            if tweet['TweetType'].strip() == "Retweet":
                if tweet_content in tweets_seen:
                    print("Skipping retweet we've seen before:",tweet_content)
                    retweets += 1
                    continue
            if tweeter in bot_probs:
                bot_result = bot_probs[tweeter]
            elif tweeter in masked_bot_authors:
                protected_user_tweets += 1
                continue
            else:
                try:
                    text_language = detect(tweet_content + " " + tweet['Bio'].strip())
                    if (text_language != 'en'):
                        print("Non-English tweet, skipping:",tweet_content)
                        tweets_seen.add(tweet_content)
                        continue
                except:
                    print("Error with language check, skipping tweet",tweet_content)
                    continue
            #elif 'lang' in tweet and tweet['lang'] != 'en':
            #    print("Non-English tweet, skipping")
            #    continue
            if bot_result == None:
                # Can also check by user ID number
                try:
                    bot_result = bom.check_account(tweeter)
                except:
                    # Common cause of errors is when users have protected tweets;
                    # botometer's calls to .followers() will fail. What to do?
                    # XXX Consider submitting just the user's tweets from the
                    # archive via the API -- maybe it will use these instead of
                    # trying to check .followers() ?
                    print("ERROR during bot check for",tweeter,"- skipping tweet and disregarding user")
                    masked_bot_authors.add(tweeter)
                    bot_result = None
                    with open(os.path.join(cache_dir, tweeter + '.json'), 'w') as f:
                        json.dump(bot_result,f)
                    continue
                with open(os.path.join(cache_dir, tweeter + '.json'), 'w') as f:
                    json.dump(bot_result,f)

            if bot_result['cap']['english'] > BOT_THRESHOLD and bot_result['cap']['universal'] > BOT_THRESHOLD:
                possible_bot_authors.add(tweeter)
                print("Possible bot:",tweeter,bot_result['cap']['english'],bot_result['cap']['universal'])
                tweets_with_possible_bot_author += 1
            else:
                tweets_seen.add(tweet_content)
                single_tweet_df = pd.DataFrame([tweet], columns=nv_fields)
                #if tweet['original_text'].find('RT @') == 0:
                if tweet['TweetType'].strip() == "Retweet":
                    retweets += 1
                else:
                    original_tweets += 1
                #with open(out_path, 'a') as out_file:
                single_tweet_df.to_csv(out_path, mode='a', sep="\t", columns=nv_fields, header=False, index=False)

print(len(possible_bot_authors),"possible bot authors:",possible_bot_authors)
print("number of tweets with possible bot authors:",tweets_with_possible_bot_author,"out of",total_tweets)
print("number of tweets with protected users (unknown bot status):",protected_user_tweets)
print("DEBOTTED TWEETS:",original_tweets,"original",retweets,"retweets")
