import sys
import os
import django
import snscrape.modules.twitter as sntwitter
import tweepy as tw
import pandas as pd
from datetime import datetime, timedelta
import time
from alive_progress import alive_bar
import texthero as hero
from texthero import preprocessing

from transformers import pipeline
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
import spacy
# en = spacy.load('en')
from spacy import displacy
from collections import Counter
import en_core_web_sm
en = en_core_web_sm.load()
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
# datetime object containing current date and time
from transformers import pipeline
summarizer = pipeline("summarization")
import os.path
import schedule
import string
import sys
import pycountry
from pycountry_convert import  country_name_to_country_alpha2
import re
import numpy as np

# datetime object containing current date and time
now = datetime.now()
print("now =", now)

if(os.path.exists('static/csv/SinceDate.txt')):
  print("File exist")
  with open('static/csv/SinceDate.txt', 'r') as f:
      date = f.read()
      since=date[5:]
      if (since==" "):
          since = pd.to_datetime(now).date()
          until = pd.to_datetime(since).date() + timedelta(days=1)
          print(until)
          print("Today",since)
      else:
          print("FileDate",since)
          until = pd.to_datetime(now).date()
          print(until)
else:
  print("File not exist")
  since = pd.to_datetime(now).date()
  until = pd.to_datetime(since).date() + timedelta(days=1)
  print(until)

keyword_list = ["#DoctorProtest","Doctor Protest"," #DoctorStrikes","Doctor Strikes"," #HealthworkerProtest","#HealthworkerStrikes" , "#NurseProtest"," #NurseStrikes"," #NurseMarch","#MidwifeProtest"]
since = since
until = until
# since = '2022-01-06'
# until = '2022-01-06'

#Function to Extract Historical Tweets  using Snscrape
def get_Oldtweets(keyword_list,since,until):
    tweets_list = []
    count=0
    start_time = time.time()
    with alive_bar(len(keyword_list)) as bar:
        for keyword in keyword_list:
          try:
            for i, tweet in enumerate(
                      sntwitter.TwitterSearchScraper(f'{keyword} since:{since} until:{until} lang:en').get_items()):
                  count += 1
                  print(f'counter is at :{count} and tag is :{keyword} Date is : {tweet.date}')
                  tweets_list.append(
                      [tweet.date, tweet.id, tweet.content, tweet.user.location, tweet.media, tweet.url, keyword,
                      tweet.retweetCount, tweet.likeCount])
                  bar()
          except:
            print(f'{keyword} not found')
            continue        


    print(" Tweets Collected Successfully",count)
    print("--- %s seconds ---" % (time.time() - start_time))
    tweets_df = pd.DataFrame(tweets_list,
                             columns=['Datetime', 'Tweet Id', 'Text', 'location', 'Media', 'TweetUrl', 'Hashtags',
                                      'Retweets Counts', 'Like Counts'])
    if(tweets_df.empty):
        print("Sorr!,No Data Available for this date") 
        sys.exit()

    else:
        print(tweets_df)

        tweets_df.drop_duplicates(subset=['Tweet Id'], inplace=True)
        print("Tweets:", len(tweets_df))
        # Obtain timestamp in a readable format
        to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')

        # tweets_df.to_csv("HealthProtest" + to_csv_timestamp + ".csv", index=None)
        tweets_df.to_csv("HealthProtest.csv", index=None)
        # since = tweets_df['Datetime'].iloc[0]

        # since=datetime.strptime(str(since), '%Y-%m-%d %H:%M:%S+00:00').replace(tzinfo=None)
        since = pd.to_datetime(now).date()
        print(type(since))
        print("Date:" + since.strftime('%Y-%m-%d'))
        file = open('static/csv/SinceDate.txt', 'w')
        file.write("Date:"+ since.strftime('%Y-%m-%d'))
        file.close()
    return tweets_df

# tweets_df = get_Oldtweets(keyword_list,since,until)   

##################################################################################

#Function to Extract Tweets Data using tweetpy

def get_twitter_api():
    # personal details
    consumer_key = 'rZ8wywfYPZyWUe8W7GfYlscDb'
    consumer_secret = 'vqPVhWWfpa7mXurWwrHyTQwE7UyrkNJgnABTwJBhrOw6ixc71m'
    access_token = '147610353-SFr440pllFKt0URuV6FDxtgXC3uXGADgAeMaOGJ1'
    access_token_secret = 'prUs9uthDWj4G9ZiLgcOCcDri9rmJldpX4uyP32pTCoGA'

    # authentication of consumer key and secret
    auth = tw.OAuthHandler(consumer_key, consumer_secret)

    # authentication of access token and secret
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    return api

def get_tweets(keyword_list,since,until):

    api = get_twitter_api()
    tweets_list = []
    count=0
    start_time = time.time()

    for keyword in keyword_list:
        try:
            for tweet in tw.Cursor(api.search, q=keyword + " -filter:retweets",lang="en",since=since).items(500):
                count += 1
                print(f'counter is at :{count} and tag is :{keyword} Date is : {tweet.created_at}')
                url_list=[]

                if 'media' in tweet.entities:
                    for image in  tweet.entities['media']:
                        print(f"xxxxxxxx: {image['media_url']}")
                        x=image['media_url']
                        url_list.append(x)

                url = f"https://twitter.com/user/status/{tweet.id}"

                print(f'url is {url} and media link is : {url_list}')

                tweets_list.append(
                    [tweet.created_at, tweet.id, tweet.text, tweet.user.location,url_list, url, keyword, 
                    tweet.retweet_count, tweet.favorite_count])
    
        except:
            print(f'{keyword} not found')
            continue        


    print(" Tweets Collected Successfully",count)
    print("--- %s seconds ---" % (time.time() - start_time))
    tweets_df = pd.DataFrame(tweets_list,
                             columns=['Datetime', 'Tweet Id', 'Text', 'location',  'Media','TweetUrl', 'Hashtags',
                                      'Retweets Counts', 'Like Counts'])

    if(tweets_df.empty):
        print("Sorr!,No Data Available for this date") 
        sys.exit()

    else:
        print(tweets_df)

        tweets_df.drop_duplicates(subset=['Tweet Id'], inplace=True)
        print("Tweets:", len(tweets_df))
        # Obtain timestamp in a readable format
        to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')

        # tweets_df.to_csv("HealthProtest" + to_csv_timestamp + ".csv", index=None)
        tweets_df.to_csv("HealthProtest.csv", index=None)
        # since = tweets_df['Datetime'].iloc[0]

        # since=datetime.strptime(str(since), '%Y-%m-%d %H:%M:%S+00:00').replace(tzinfo=None)
        since = pd.to_datetime(now).date()
        print(type(since))
        print("Date:" + since.strftime('%Y-%m-%d'))
        file = open('static/csv/SinceDate.txt', 'w')
        file.write("Date:"+ since.strftime('%Y-%m-%d'))
        file.close()

    return tweets_df

tweets_df = get_tweets(keyword_list,since,until) 

def remove_URL(Text):
    """Remove URLs from a tweets"""
    return re.sub(r"http\S+", "", Text) 
    
def remove_user(Text):  
    """Remove @Users from a tweets"""  
    return re.sub("(@[A-Za-z0-9]+)|(@[^0-9A-Za-z \t]+)|(@\w+:\/\/\S+)", " ", Text)
    
#Function to pre-process tweets data
def preprocess_tweets(tweets_df):
    if(tweets_df.empty):
        tweets_df = pd.read_csv("HealthProtest.csv")
    else:
        print("Data before removing Duplicates:", len(tweets_df))
        tweets_df.drop_duplicates(subset='Text', inplace=True)
        print("Data after removing Duplicates:", len(tweets_df))
        missing_values_count = tweets_df.isnull().sum()
        missing_values_count
        # tweets_df = tweets_df.dropna()
        tweets_df = tweets_df.dropna(subset=['location'])
        print("Data after removing Missing Values from Location:", len(tweets_df))
        tweets_df["Media"].fillna("No Media Available", inplace = True)
        tweets_df["Hashtags"].fillna("No Hashtags Available", inplace = True)
        tweets_df = tweets_df.reset_index(drop=True)
        # print("Clean Dataset",tweets_df)

        custom_pipeline = [preprocessing.lowercase,
                        preprocessing.remove_whitespace,
                        preprocessing.remove_punctuation,
                        preprocessing.remove_diacritics,
                        # preprocessing.remove_stopwords,
                        preprocessing.remove_brackets,
                        preprocessing.remove_curly_brackets,
                        preprocessing.remove_angle_brackets,
                        preprocessing.remove_round_brackets,
                        preprocessing.remove_square_brackets,
                        preprocessing.remove_html_tags,
                        preprocessing.remove_urls,
                        # preprocessing.stem
                        # preprocessing.tokenize
                        ]

        custom_pipeline1 = [preprocessing.lowercase,
                        preprocessing.remove_whitespace,
                        preprocessing.remove_punctuation,
                        preprocessing.remove_diacritics,
                        preprocessing.remove_stopwords,
                        preprocessing.remove_brackets,
                        preprocessing.remove_curly_brackets,
                        preprocessing.remove_angle_brackets,
                        preprocessing.remove_round_brackets,
                        preprocessing.remove_square_brackets,
                        preprocessing.remove_html_tags,
                        preprocessing.remove_urls,
                        preprocessing.stem
                        # preprocessing.tokenize
                        ]

        tweets_df["clean_text"] = tweets_df["Text"].apply(lambda text: remove_URL(text))    
        tweets_df["clean_text"] = tweets_df["clean_text"].apply(lambda text: remove_user(text))                      
        tweets_df['clean_text'] = hero.clean(tweets_df['clean_text'], custom_pipeline)
        tweets_df['clean_text1'] = hero.clean(tweets_df['clean_text'], custom_pipeline1)
        print("Clean Dataset",tweets_df)
        # tweets_df['Text'] = hero.clean(tweets_df['Text'])

        tweets_df.to_csv("static/csv/pre-processed_HealthProtest.csv", index=None)
        
    return tweets_df

tweets_df = preprocess_tweets(tweets_df)

# tweets_df=pd.read_csv('Health_protest_500.csv',index_col=0)
# tweets_df=preprocess_tweets(tweets_df)

print('===============================')
print('Getting Organization.. .. ')

def get_organization(text):
  if type(text)==str and len(text)>10:
    sents = en(text)
    value = [str(ee) for ee in sents.ents if ee.label_ == 'ORG']
    value=list(set(value))
    if len(value)>0:
      return value
    else :
      return 'null'
  else:
    return 'null'
############  Function call get_organization  ##############    
tweets_df["Organization"] = tweets_df["clean_text"].apply(lambda text: get_organization(text))

print('===============================')
print('Getting Type of Protest.. .. ')

def get_organization(text):
  if type(text)==str and len(text)>10:
    sents = en(text)
    value = [str(ee) for ee in sents.ents if ee.label_ == 'EVENT']
    value=list(set(value))
    if len(value)>0:
      return value
    else :
      return 'null'
  else:
    return 'null'
############  Function call get_organization  ##############    
tweets_df["Protest Type"] = tweets_df["clean_text"].apply(lambda text: get_organization(text))

print('===============================')
print('Getting nationalities, religious, or political groups involved.. .. ')

def get_AnyGroupInvolved(text):
  if type(text)==str and len(text)>10:
    sents = en(text)
    value = [str(ee) for ee in sents.ents if ee.label_ == 'NORP']
    value=list(set(value))
    if len(value)>0:
      return value
    else :
      return 'null'
  else:
    return 'null'
############  Function call get_AnyGroupInvolved  ##############    
tweets_df["NORP"] = tweets_df["clean_text"].apply(lambda text: get_AnyGroupInvolved(text))

print('===============================')
print('Getting Protest Size.. .. ')
def get_ProtestSize(text):
  if type(text)==str and len(text)>10:
    sents = en(text)
    value = [str(ee) for ee in sents.ents if ee.label_ == 'QUANTITY']
    value=list(set(value))
    if len(value)>0:
      return value
    else :
      return 'null'
  else:
    return 'null'
############  Function call get_ProtestSize  ##############    
tweets_df["Protest-Size"] = tweets_df["clean_text"].apply(lambda text: get_ProtestSize(text))


print('===============================')
print('Getting GPE_Location from Tweet Text.. .. ')
def get_LocationInText(text):
  if type(text)==str:
    sents = en(text)
    value = [str(ee) for ee in sents.ents if ee.label_ == 'GPE' or ee.label_ == 'LOC']
    value=list(set(value))
    if len(value)>0:
      return value
    else :
      return np.NaN
  else:
    return np.NaN
############  Function call get_organization  ##############    
tweets_df["GPE"] = tweets_df["clean_text"].apply(lambda text: get_LocationInText(text))


print('===============================')
print('Getting Location from Tweet Text.. .. ')
def get_newLocationInText(text):
  if type(text)==str:
    sents = en(text)
    value = [str(ee) for ee in sents.ents if ee.label_ == 'LOC']
    value=list(set(value))
    if len(value)>0:
      return value
    else :
      return np.NaN
  else:
    return np.NaN
############  Function call get_organization  ##############    
tweets_df["LOC"] = tweets_df["clean_text"].apply(lambda text: get_newLocationInText(text))

print('===============================')
print('Getting Date from Tweet Text.. .. ')
def get_newDATEInText(text):
  if type(text)==str:
    sents = en(text)
    value = [str(ee) for ee in sents.ents if ee.label_ == 'DATE']
    value=list(set(value))
    if len(value)>0:
      return value
    else :
      return 'null'
  else:
    return 'null'
############  Function call get_organization  ##############    
tweets_df["NEW_DATE"] = tweets_df["clean_text"].apply(lambda text: get_newDATEInText(text))

print('===============================')
print('Getting person name.. .. ')

def get_Person_nameSpacy(text):
  if type(text)==str and len(text)>10:
    sents = en(text)
    value = [str(ee) for ee in sents.ents if ee.label_ == 'PERSON']
    value=list(set(value))
    if len(value)>0:
      return value
    elif 'doctor' in text:
      return 'doctor'
    elif 'nurse' in text:
      return 'nurse'
    elif 'Healthworker' in text:
      return 'Healthworker' 
  else:
    return 'null'    

tweets_df["name"] = tweets_df["clean_text"].apply(lambda text: get_Person_nameSpacy(text))

tweets_df = tweets_df.dropna(subset=['GPE'])
print("Data after removing Missing Values from GPE:", len(tweets_df))
# tweets_df.to_csv("D:/protest_tracker_10_01_2022/static/csv/HealthProtest_afterGPE.csv", index=None)

print('===============================')
print('Finding Location Using Geopy Libray.. .. ')
def findGeocode(country):
       
    # try and catch is used to overcome
    # the exception thrown by geolocator
    # using geocodertimedout  
    try:
          
        # Specify the user_agent as your app name

        geolocator = Nominatim(user_agent="http")
          
        x= geolocator.geocode(country,timeout=None)
        return str(x)

    except GeocoderTimedOut as e:
         time.sleep(1)
         return findGeocode(country)    

tweets_df["geo_country"] = tweets_df["GPE"].apply(lambda text: findGeocode(text))

def add_text(txt):
  txt=string.capwords(txt)
  a=' Country'
  txt=txt+a
  return txt

tweets_df["Country"] = tweets_df["geo_country"].apply(lambda text: add_text(text))


print('===============================')
print('Finding Country pycountry.. .. ')

def get_country(text):
  
  cc=[]
  if type(text)==str :

    for country in pycountry.countries:
        if country.name in text:
          cc.append(country.name)
          cc=list(set(cc))

    if len(cc)>0:
      u=cc[0]
      return u
    else:
      return np.NaN
  else:
    return np.NaN

tweets_df["Country"] = tweets_df["Country"].apply(lambda text: get_country(text))
tweets_df = tweets_df.dropna(subset=['Country'])
print("Data after removing Missing Values from Country:", len(tweets_df))

# print('===============================')
# print('Extrating Country Codes.. .. ')
def get_countryCode(text):
    try:
        cn_a2_code =  country_name_to_country_alpha2(text)
    except:
        cn_a2_code = np.NaN 
        
    return (cn_a2_code) 

tweets_df["Country_Code"] = tweets_df["Country"].apply(lambda text: get_countryCode(text))    
tweets_df = tweets_df.dropna(subset=['Country_Code'])
print("Data after removing Missing Values from Country_Code:", len(tweets_df))

# print('===============================')
# print('Sentiment-analysis started.. .. ')

model = pipeline("sentiment-analysis")
def perdict(text):
  text=model(text)
  x=text[0]
  return x.get('label')

tweets_df["sentiment"] = tweets_df["clean_text1"].apply(lambda text: perdict(text))

def extract_topic(text,n_top_words):
  q=(text,)
  value={}
  lda = LatentDirichletAllocation(n_components=5, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
  tf_vectorizer = CountVectorizer(
                                max_features=50,
                                stop_words='english')
  tf = tf_vectorizer.fit_transform(q)
  tf_feature_names = tf_vectorizer.get_feature_names_out()
  lda.fit(tf)

  for topic_idx, topic in enumerate(lda.components_):
        count=topic_idx
        x=" ".join([tf_feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
        value.update({count:x})
        
  return value 

# print('===============================')
# print('Extracting Topics.. .. ')

def get_topics(text):
  if type(text)==str and len(text)>10:
    
    value =extract_topic(text,5)
    if len(value)>0:
      return value
    else :
      return 'null'
  else:
    return 'null'

tweets_df["topics"] = tweets_df["clean_text1"].apply(lambda text: get_topics(text))

tweets_df.to_csv("static/csv/HealthProtest_Data.csv", index=None)
print("DONE!!!!!")   

print('===============================')
print('Extract Summary.. .. ')
def get_summry(place):
  
  data=tweets_df.query("Country==@place")
  data.columns = data.columns.str.replace(' ', '')
  
  text=''
  for i in data.clean_text.values:
    text=text+i
  
  summary=summarizer(text, max_length=300, min_length=50, do_sample=False,)
  
  summary=summary[0]
  summary=summary.get('summary_text')

  date=list(data.Datetime.values)
  text=summary
  tweet_ids=list(data.TweetId.values)
  location=list(data.location.values)
  media=list(data.Media.values)
  url=list(data.TweetUrl.values)
  hastags=list(data.Hashtags.values)
  retweet_count=sum(list(data.RetweetsCounts.values))
  likeCount=sum(list(data.LikeCounts.values))
  Organization=list(data.Organization.values)
  name=list(data.name.values)
  sentiment=list(data.sentiment.values)
  topics=list(data.topics.values)
  geo_country=list(data.geo_country.values)
  Country=list(data.Country.values)


  d={'Datetime':[date],'TweetId':[tweet_ids],'location':[location],'Media':[media],
     'url':[url],'hastag':[hastags],'retweet_count':retweet_count,'likeCount':likeCount,
     'Organization':[Organization],'name':[name],'sentiment':[sentiment],
     'topics':[topics],'summary':[text],'geo_country':[geo_country],'Country':[Country]}

  dataframe=pd.DataFrame(d)
  return dataframe


x=tweets_df.Country.unique()
print(f'X is{x}')
# if x == '' :
# print("No Data Avaialable!!!")

# else:  
frame=[]
for i in x:
  c=get_summry(i)
  frame.append(c)
final_df=pd.concat(frame)
final_df.to_csv('static/csv/Summary_data.csv')



