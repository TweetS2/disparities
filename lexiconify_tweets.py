import pickle
import numpy as np
import json
import pandas as pd
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#twitter_path = '_______________________.tsv'
#bt_path = '_________________________.tsv'
twitter_path = 'debotted_tweets/#Twitter.nobots.tsv'
bt_path = 'debotted_tweets/#BlackTwitter.nobots.tsv'

lexicon_corpus = '#BlackTwitter'
lexicon_out = 'bt_lexicon.tsv'
lexicon_rejects = 'bt_rejects.tsv'

lexicon_terms_seen = set()

MAX_TERMS = 4000

if os.path.isfile(lexicon_out):
  with open(lexicon_out, 'r') as lf:
    for line in lf:
      if line.strip().split("\t")[0] == 'term':
        continue
      else:
        lexicon_terms_seen.add(line.strip().split("\t")[0])

if os.path.isfile(lexicon_rejects):
  with open(lexicon_rejects, 'r') as lf:
    for line in lf:
      lexicon_terms_seen.add(line.strip().split("\t")[0])

control = pd.read_csv(twitter_path, sep="\t")
bt = pd.read_csv(bt_path, sep="\t")

use_cache = True

target_labels = ['#BlackTwitter', '#Twitter']
class_priors = [.182, .818]
norm_labels = [lab.lower() for lab in target_labels]

c_vectorizer = CountVectorizer(stop_words='english')
tokenize = c_vectorizer.build_tokenizer()

# Return a list of tweet texts (and possibly other features)
# after running various filters on them, e.g., removing usernames
# and hashtags (or moving these to separate feature arrays),
# resolving shortened URLs, using bio data (or not)
def preprocess_tweets(tweets_df):
  processed_tweets = []
  for t, tweet in tweets_df.iterrows():
    tweet_content = tweet['Content'].strip().split(' ')
    cleaned_content = []
    for term in tweet_content:
      if term.lower() in norm_labels:
        continue
      elif term.startswith('@'):
        #cleaned_content.append(term.replace('@', ''))
        continue
      elif term.startswith('#'):
        cleaned_content.append(term.replace('#', ''))
        #continue
      else:
        cleaned_content.append(term)
    tweet_content = tokenize(' '.join(cleaned_content))
    cleaned_content = []
    for term in tweet_content:
      try:
        int(term.strip())
        continue
      except ValueError:
        cleaned_content.append(term)
    processed_tweets.append(' '.join(cleaned_content))
  return processed_tweets


n_features = 2000000 # This requires a lot of RAM, but produces better results

control_tweets = preprocess_tweets(control)
bt_tweets = preprocess_tweets(bt)

all_texts = []
all_labels = []

classifier_data = []

for tweet in bt_tweets:
  all_texts.append(tweet)
  all_labels.append('#BlackTwitter')
  classifier_data.append([tweet, '#BlackTwitter'])
for tweet in control_tweets:
  all_texts.append(tweet)
  all_labels.append('#Twitter')
  classifier_data.append([tweet, '#Twitter'])

random.shuffle(classifier_data)

#extra_stopwords = []
#tf_vectorizer = TfidfVectorizer(stop_words='english')
#expanded_stop_words = set(tf_vectorizer.get_stop_words().copy())
#expanded_stop_words.update(extra_stopwords)

if use_cache:
  if os.path.isfile("tfidf_tweet_vectorizer.rb"):
    vectorizer = pickle.load(open("tfidf_tweet_vectorizer.rb", "rb"))
  else:
    vectorizer = TfidfVectorizer(norm=None, use_idf=True, max_features=n_features, min_df=2, ngram_range=(1, 1), max_df=.95, stop_words='english')
  
  if os.path.isfile("tfidf_tweet_dtm.rb"):
    dtm = pickle.load(open("tfidf_tweet_dtm.rb", "rb"))
  else:
    dtm = vectorizer.fit_transform(all_texts)
    pickle.dump(dtm, open("tfidf_tweet_dtm.rb", "wb"))
  
  if not os.path.isfile("tfidf_tweet_vectorizer.rb"):
    pickle.dump(vectorizer, open("tfidf_tweet_vectorizer.rb", "wb"))

else:
  vectorizer = TfidfVectorizer(norm=None, use_idf=True, max_features=n_features, min_df=2, ngram_range=(1, 1), max_df=.95, stop_words='english')
  # Try only looking at function/stop words, i.e., stylistcs
  #vectorizer = TfidfVectorizer(norm=None, use_idf=True, ngram_range=(1, 1), vocabulary=tf_vectorizer.get_stop_words())
  dtm = vectorizer.fit_transform(all_texts)

vocab = np.array(vectorizer.get_feature_names())

def data_iterator(input_data, step):
  max_limit = len(input_data)
  counter = 0
  while counter + step < max_limit:
    yield input_data[counter:counter+step]
    counter += step

# batcher step should be the size of the test set
# (and also the step size of the training data)
batcher = data_iterator(classifier_data, 1000)

test_data = next(batcher)

test_texts = []
test_labels = []

for text, label in test_data:
  test_texts.append(text)
  test_labels.append(label)
    
X_test = np.array(test_texts)
y_test = np.array(test_labels)

X_test_features = vectorizer.transform(X_test)

# ## Naive Bayes classifier with tf-idf text features

from sklearn.naive_bayes import MultinomialNB

if use_cache and os.path.isfile("tfidf_tweet_classifier.rb"):
  cls = pickle.load(open("tfidf_tweet_classifier.rb", "rb"))
else:
  cls = MultinomialNB(alpha=1.0, class_prior=class_priors, fit_prior=True)

  i = 0
  train_data = next(batcher)
  while train_data:
    train_texts = []
    train_labels = []

    for text, label in train_data:
      train_texts.append(text)
      train_labels.append(label)
      
    X_train = np.array(train_texts)
    y_train = np.array(train_labels)
      
    X_train_features = vectorizer.transform(X_train)

    i += 1

    cls.partial_fit(X_train_features, y_train, classes=target_labels)
      
    if i % 50 == 0:
      accuracy = cls.score(X_test_features, y_test)
      print("batch",i,"accuracy",accuracy)

    try:
      train_data = next(batcher)
    except:
      train_data = None

if use_cache and not os.path.isfile("tfidf_tweet_classifier.rb"):
  pickle.dump(cls, open("tfidf_tweet_classifier.rb", "wb"))

def most_informative_features(classifier, vectorizer=None, n=MAX_TERMS):
  class_labels = classifier.classes_
  if vectorizer is None:
      feature_names = classifier.steps[0].get_feature_names()
  else:
      feature_names = vectorizer.get_feature_names()
  topn_class1 = sorted(zip(classifier.feature_log_prob_[0], feature_names))[-n:]
  topn_class2 = sorted(zip(classifier.feature_log_prob_[1], feature_names))[-n:]
  features = {}
  features[class_labels[1]] = {}
  for prob, feat in reversed(topn_class2):
      features[class_labels[1]][feat] = prob
      #print(class_labels[1], prob, feat)
  print()
  features[class_labels[0]] = {}
  for prob, feat in reversed(topn_class1):
      features[class_labels[0]][feat] = prob
      #print(class_labels[0], prob, feat)
  return features

mif = most_informative_features(cls, vectorizer)

"""
print("Running classification test using #Twitter tweets")
tweets = pd.read_csv(twitter_path, sep="\t")

import statistics
predictions = {}

for t, tweet in tweets.iterrows():
  pred = cls.predict(vectorizer.transform([tweet['Content'].strip()]))[0]
  if pred not in predictions:
    predictions[pred] = 1
  else:
    predictions[pred] += 1

print(str(predictions))
print("{:.2%}".format(float(predictions['#BlackTwitter']) / float(sum(predictions.values()))))
"""

# ## Statistical tests for "distinctive" words

from sklearn.feature_selection import chi2

def get_chi2_keyness(dtm, labels, corpus, vocab, vectorizer_type="", n=MAX_TERMS):
  keyness, _ = chi2(dtm, labels)
  ranking = np.argsort(keyness)[::-1]
  ranked_terms = vocab[ranking][:n]
  return ranked_terms

keys_by_corpus = {}

for corpus in target_labels:
  uni_labels = []
  for lab in all_labels:
    if lab == corpus:
      uni_labels.append(corpus)
    else:
      uni_labels.append("The Rest")
  chi2_keys = get_chi2_keyness(dtm, uni_labels, corpus, vocab, "TF-IDF")
  keys_by_corpus[corpus] = chi2_keys.tolist()

if use_cache:
  with open('chi2_features.json', 'w') as jfile:
    json.dump(keys_by_corpus, jfile)

c_dtm = c_vectorizer.fit_transform(all_texts)
c_vocab = c_vectorizer.get_feature_names()

print("DTM stats:")
print(c_dtm.shape)
print(len(all_texts))


# Get the total number of terms in each document of the corpus as read by the vectorizer
total_terms = 0 #np.sum(c_dtm)

document_term_totals = []

all_text_count = len(all_texts)

for i in range(len(all_texts)):
    terms_in_doc = np.sum(c_dtm[i])
    document_term_totals.append(terms_in_doc)
    total_terms += terms_in_doc

print("Vocabulary stats:")
print(len(document_term_totals))
print(total_terms)


from scipy.stats import chi2_contingency, fisher_exact
import datetime
# For each top term for African-American
# Get the total number of occurrences of this term in #BlackTwitter texts
# Get the total number of occurrences of the term in non-BT texts

def termness_test(target_word, target_label, c_vocab, all_labels):
  w = c_vocab.index(target_word)
  w_in_target = 0
  not_w_in_target = 0
  w_not_in_target = 0
  not_w_not_in_target = 0
  doc_freq = c_dtm[:,w].count_nonzero()
  log_inv_freq = np.log(all_text_count / (1 + doc_freq))
  for i, label in enumerate(all_labels):
    if label == target_label:
      w_in_target += c_dtm[i,w]
      not_w_in_target += document_term_totals[i] - c_dtm[i,w]
    else:
      w_not_in_target += c_dtm[i,w]
      not_w_not_in_target += document_term_totals[i] - c_dtm[i,w]

  obs = np.array([[w_in_target,not_w_in_target],[w_not_in_target,not_w_not_in_target]])

  g, p, dof, expctd = chi2_contingency(obs, lambda_="log-likelihood")
  ratio_to_expected = float(w_in_target) / expctd[0,0]
  oddsratio, pvalue = fisher_exact(obs)
    
  return [ratio_to_expected, p, oddsratio, pvalue, w_in_target, w_not_in_target, doc_freq, log_inv_freq]

lexicon_file = open(lexicon_out, 'a')

rejects_file = open(lexicon_rejects, 'a')

if len(lexicon_terms_seen) == 0:
  lexicon_file.write("\t".join(['term','corpus_freq','all_freq','log_ratio','log_p','fisher_ratio','fisher_p','bayes_nll','doc_freq','log_inv_freq']) + "\n")

label_list = lexicon_corpus
target_corpus = lexicon_corpus
term_count = 0
for term in keys_by_corpus[label_list]:
  term_count += 1
  if term in lexicon_terms_seen:
    continue
  print(term,term_count,target_corpus,datetime.datetime.now(), flush=True)
  log_ratio, log_p, fisher_ratio, fisher_p, corpus_freq, others_freq, doc_freq, log_inv_freq = termness_test(term, target_corpus, c_vocab, all_labels)
  print("document frequency:",doc_freq,"inverse doc frequency:",log_inv_freq, flush=True)
  print("log likelihood:",log_ratio,log_p, flush=True)
  print("Fisher test:",fisher_ratio,fisher_p, flush=True)
  if log_ratio < 1 or fisher_ratio < 1 or log_p > .05 or fisher_p > .05:
    print("REJECTED FROM LEXICON:",term, flush=True)
    rejects_file.write(term + "\n")
    continue
  if term in mif['#BlackTwitter']:
    bayesian_nll = mif['#BlackTwitter'][term]
  else:
    bayesian_nll = "NA" 
  print("Bayesian negative log likelihood:",bayesian_nll, flush=True)
  print(term,"freq in",lexicon_corpus,"corpus:",corpus_freq,"freq in all_corpora",corpus_freq+others_freq, flush=True)
  row_data = [term, corpus_freq, corpus_freq+others_freq, log_ratio, log_p, fisher_ratio, fisher_p, bayesian_nll, doc_freq, log_inv_freq]
  row_strings = [str(item) for item in row_data]
  lexicon_file.write("\t".join(row_strings) + "\n")
  lexicon_file.flush()

lexicon_file.close()
rejects_file.close()
