import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import warnings
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import time
from sklearn.cross_validation import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

#Read csv
tweets = pd.read_csv("Tweets.csv")

#change the target column to 0, 1 and 2 for sentiment
pTweets = tweets.iloc[:,(10,1)]
pTweets.columns = ['data', 'target']
pTweets['target'] = pTweets['target'].str.strip().str.lower()
pTweets['target'] = pTweets['target'].map({'negative': 0 , 'positive': 1 , 'neutral': 2})

dataTweets = pTweets

#Removes special characters and numbers
dataTweets['data'] = dataTweets['data'].str.replace("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([0-9])"," ")

#Tokenizes into words
dataTweets['data'] = dataTweets['data'].apply(nltk.word_tokenize)

#Stem the tokens
stemmer = SnowballStemmer('english')
dataTweets['data'] = dataTweets['data'].apply(lambda x: [stemmer.stem(y) for y in x])

#Lemmatize the tokens
lemmatizer = nltk.WordNetLemmatizer()
dataTweets['data'] = dataTweets['data'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])

#Remove stopwords
stopwords = nltk.corpus.stopwords.words('english')

#Stem the stopwords
stemmed_stops = [stemmer.stem(t) for t in stopwords]

#Remove stopwords from stemmed/lemmatized tokens
dataTweets['data']=dataTweets['data'].apply(lambda x: [stemmer.stem(y) for y in x if y not in stemmed_stops])

#Remove words that are too short
dataTweets['data']=dataTweets['data'].apply(lambda x: [e for e in x if len(e) >= 3])

dataTweets['data']=dataTweets['data'].str.join(" ")

#X becomes our tweets and y becomes the target
X = dataTweets['data']
y = dataTweets['target']

arr_Accu=[]

#Simple Dummy Classifier for choosing best random state
for i in range(1,20):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, random_state=i)

    vect = CountVectorizer(stop_words='english',analyzer="word",min_df = 2, max_df = 0.8)
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    feat_dtm = vect.get_feature_names()

    clf = DummyClassifier()
    clf.fit(X_train_dtm, y_train)
    y_pred = clf.predict(X_test_dtm)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    #print(accuracy)
    arr_Accu.append(accuracy)

#Vectorize
vect = CountVectorizer(stop_words='english',analyzer="word",min_df = 2, max_df = 0.8)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)
feat_dtm = vect.get_feature_names()

clf_stats = pd.DataFrame()

#Logististic regression portion
clf = LogisticRegression()
start_time = time.time()
clf.fit(X_train_dtm, y_train)
runtime = time.time()-start_time
y_pred = clf.predict(X_test_dtm)
accuracy = metrics.accuracy_score(y_test, y_pred)
#print'Accuracy : ',accuracy

#Store classifier stats
clf_stats = clf_stats.append({'Classifier': 'Logistic Regression', 'Accuracy': accuracy, 'Runtime': runtime, 'Callable': 'clf = LogisticRegression()'}, ignore_index=True)

# Vectorize, fit, transform. Select model randomly
vect = CountVectorizer(stop_words='english', analyzer="word", min_df = 2, max_df = 0.8)
X_dtm = vect.fit_transform(X)
feat_dtm = vect.get_feature_names()

# Select the best performing classifier
Call_clf = str(clf_stats[['Callable','Accuracy']].sort(['Accuracy'], ascending=[False]).head(1).iloc[:,(0)])
temp = Call_clf.__repr__()
Call_clf = temp[temp.index('c'):(temp.index(')'))+1]
exec(Call_clf)
clf.fit(X_dtm.toarray(), y) 

def fmtInputTweet(txt):
    
    #Remove special characters and numbers
    txt = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([0-9])"," ",txt)
    
    #Tokenizes into words
    tokens = [word for word in nltk.word_tokenize(txt)]

    #Only keep the tokens that begin with a letter
    clean_tokens = [token for token in tokens if re.search(r'^[a-zA-Z]+', token)]

    #Stem the tokens
    stemmer = SnowballStemmer('english')
    stemmed_tokens = [stemmer.stem(t) for t in clean_tokens]

    #Lemmatize
    lemmatizer = nltk.WordNetLemmatizer()
    lem_tokens = [lemmatizer.lemmatize(t) for t in stemmed_tokens]
    
    #Remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')

    #Stem the stopwords
    stemmed_stops = [stemmer.stem(t) for t in stopwords]

    lem_tokens_no_stop = [stemmer.stem(t) for t in lem_tokens if t not in stemmed_stops]

    #Remove words that are too short
    clean_lem_tok = [e for e in lem_tokens_no_stop if len(e) >= 3]
    
    #Detokenize new tweet for vector processing
    new_formatted_tweet=" ".join(clean_lem_tok)
    
    return new_formatted_tweet

def classifyNewTweet(new_twt):  

    fmtTweet = fmtInputTweet(new_twt)
    print fmtTweet
    fmtTweetDtm = vect.transform([fmtTweet])[0]
    pred = clf.predict(fmtTweetDtm.toarray())

    def mood(x):
        return {
            0: 'negative',
            1: 'positive',
            2: 'neutral'
        }[x]

    print'Tweet sentiment:',mood(pred[0])

twt = '@AmericanAirlines I hate this plane. Its so sad. Im going to cry'
print twt
classifyNewTweet(twt)