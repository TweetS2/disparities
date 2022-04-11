#!/bin/python

# topic_model_tweets.py
# Code to run topic modeling on a collection of tweet texts.
# Can use either LDA or NMF methods.
# Closely based upon this example:
# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

import pandas as pd

# This expects a tab-delimited file with full tweet contents and a header row
tweets = pd.read_csv("../Tweets/BlackTwitter.txt", sep="\t")
tweet_texts = [tweet['Tweet'] for t, tweet in tweets.iterrows()]

n_samples = 2000
n_features = 1000
n_components = 50 # This is the number of topics returned for the corpus
n_top_words = 15 # This is just how many top words will be shown for each topic

def print_top_words(model, feature_names, n_top_words):
  for topic_idx, topic in enumerate(model.components_):
    message = "Topic #%d: " % topic_idx
    message += " ".join([feature_names[i]
    					 for i in topic.argsort()[:-n_top_words - 1:-1]])
    print(message)
  print()

# Add Twitter-specific stopwords to the default list of English stopwords
# NOTE: The standard English stopword set may NOT be appropriate for this or
# other tasks -- should compare results with and without first to make sure.
twitter_stopwords = ['https', 'http']
tf_vectorizer = CountVectorizer(stop_words='english')
expanded_stop_words = set(tf_vectorizer.get_stop_words().copy())
expanded_stop_words.update(twitter_stopwords)

# Modify default tokenization string to retain @ and # prefixes (mentions and hashtags)
# Remove the token_pattern= parameter below if you don't want to do this.
twitter_token_pattern=r'\b\w\w+\b|(?<!\w)@\w+|(?<!\w)#\w+'
  
# LDA first

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
								max_features=n_features,
								token_pattern=twitter_token_pattern,
								stop_words=expanded_stop_words)

tf = tf_vectorizer.fit_transform(tweet_texts) # This might take a little while

print("LDA topics:")
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
								learning_method='online',
								learning_offset=50.,
								random_state=0)

lda.fit(tf) # This actually builds the topic model from the corpus -- may take a while

tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

# Now NMF (2 methods)

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   token_pattern=twitter_token_pattern,
                                   stop_words=expanded_stop_words)

tfidf = tfidf_vectorizer.fit_transform(tweet_texts)

# NMF with Frobenius norm method
print("NMF topics with Frobenius norm method:")
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

# NMF with Kullback-Leibler method
print("NMF topics with Kullback-Leibler method:")
nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)
