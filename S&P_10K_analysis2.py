
#import libraries

import pandas as pd
import numpy as np
import xlrd
import pickle
import random

from edgar import Company
from edgar import TXTML

from bs4 import BeautifulSoup

import lxml.html
import lxml.html.soupparser
import requests, re
print('LXML parser, BeautifulSoup, and SEC edgar Libraries imported.')
print()

import nltk
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from html import unescape

import time
from tqdm import tqdm
import urllib.request
import urllib.error

#  #initially set to zero until the build dataset function is run
# global counter
# counter = 0
# i = 0
#
# def sleep(timeout, retry=3):
#     counter = i #retain the counter i in case the sleep function is called
#     def the_real_decorator(function):
#         def wrapper(*args, **kwargs):
#             retries = 0
#             while retries < retry:
#                 try:
#                     value = function(*args, **kwargs)
#                     if value is None:
#                         return
#                 except:
#                     print(f'Sleeping for {timeout} seconds')
#                     time.sleep(timeout)
#                     retries += 1
#         return wrapper
#     return the_real_decorator

def scrape_wiki(url):
    scrape_url = requests.get(url).text
    soup = BeautifulSoup(scrape_url, 'xml')

    return(soup)
# create a dataframe from a word matrix
def wmdf(wm, feat_names):
    # create an index for each row
    doc_names =  [2019,2018,2017]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,
                      columns=feat_names).transpose()
    return(df)

# create a spaCy tokenizer
spacy.load('en')
nlp = spacy.lang.en.English()
nlp.max_length = 1500000  #a handful of companies have so much text it exceeds the character limit on the library

# remove html entities from docs and
# set everything to lowercase
def my_preprocessor(doc):
    return(unescape(doc).lower())

# tokenize the doc and lemmatize its tokens
def my_tokenizer(doc):
    tokens = nlp(doc, disable = ['ner', 'parser']) #disable RAM-hungry intensive parts of the pipeline you don't need for lemmatization
    return([token.lemma_ for token in tokens])


'''we use this function to perform the get filings. we need to run this function and iterarte
over our list of tickers. Each ticker will get parsed and collected into a dataframe.'''
def pull_10K(company_name, company_id):
    company = Company(company_name, company_id)
    tree = company.get_all_filings(filing_type = "10-K")
    pre_time = time.time()
    offset = random.randint(1,25) #seconds
    if pre_time + offset > time.time():
        docs = Company.get_documents(tree, no_of_documents=3)
        pre_time = time.time()
    text_l=[]
    for i in range(len(docs)):
        try:
            text=TXTML.parse_full_10K(docs[i])
            text_l.append(text)
        except IndexError:
            pass
    return text_l

'''this function will parse each 10k and use regex to find the Risk Factors section'''
def pull_risk_section(text):
    text = re.sub('\n', ' ', text)
    text = re.sub('\xa0', ' ', text)
    matches = list(re.finditer(re.compile('Item [0-9][A-Z]\.', re.IGNORECASE), text))
    #using this bock of code to isolate any null values
    if matches==[]: #avoid returning empty sets for 10k ammendments that may exclude 1A sections
        return text
    try:
        start = max([i for i in range(len(matches)) if matches[i][0].casefold() == ('Item 1A.').casefold()])
    except:
        return text #avoid returning empty sets for 10k ammendments that may exclude 1A sections
    #print(text)
    end = start+1
    start = matches[start].span()[1]
    if end > len(matches)-1:
        end = start
    else:
        end = matches[end].span()[0]

    text = text[start:end]
    #print(start, end)

    return text

'''We are building a dataset that will compile the relative changes in negative word
frequency to each filing, year after year.'''

def build_dataset(ticker_dict):
    #Iniitialize temporary dtatframe
    data_comp = pd.DataFrame(columns=['Company','YoY Chg','3yr-CAGR'])
    spx_length= len(ticker_dict)
  
    for i in range(470, spx_length):
        try:
            if i % 10 == 0: #save our data every 10 entries due to SEC throttling
                with open('/home/gregjroberts1/Projects/Data/comp_builder_update.p', 'wb') as f:
                    pickle.dump(data_comp, f)

            #print(data_comp)
            pairs = {k: ticker_dict[k] for k in list(ticker_dict)[i:i+1]}
            for key, value in pairs.items():
                pair_values = (key, value)
                print(i, pair_values[0], pair_values[1])
            #Feed the edgar filing functions the security values
            documents = pull_10K(pair_values[0],pair_values[1])
            #Error handing for newly listed with not enough 10ks to process...
            if len(documents) < 3:
                YoY=CAGR3 = float("NaN")
                data_comp = data_comp.append({'Company': pair_values[0],
                                  'YoY Chg':YoY, '3yr-CAGR': CAGR3}, ignore_index=True)
                continue

            #Pull the risk sections for all these 10k filings
            risk_sections = [pull_risk_section(document) for document in documents]

            #tokenize and stem each each token from the document
            custom_vec = CountVectorizer(preprocessor=my_preprocessor, tokenizer=my_tokenizer, stop_words='english')
            cwm = custom_vec.fit_transform(risk_sections)
            tokens = custom_vec.get_feature_names()
            #print(len(tokens))
            counts = wmdf(cwm, tokens)
            counts = counts.stack().reset_index() #recast the dataframe to allow for the 3 columns listed below

            # risk_sections = [stemmer.stem(risk_section) for risk_section in risk_sections]

            #tokenizing the document
            #counts = vectorizer.fit_transform(risk_sections)
            # counts = pd.DataFrame(counts.toarray(),columns=vectorizer.get_feature_names()).transpose()
            #
            # counts.columns = [2019,2018,2017,2016,2015]
            # counts = counts.stack().reset_index()

            #Creating a temp dataframe to count frequencies
            counts.columns = ["Word", "Time Period", "Count"]
            counts["Company"] = pair_values[0]

            #Create a matrix of word types and the words that match these types
            #The word list has multiple sheets with tone descriptions for different words
            #Something to note is that a word can be in multiple lists!

            word_list = []
            for sentiment_class in ["Negative", "Positive", "Uncertainty", "Litigious",
                           "StrongModal", "WeakModal", "Constraining"]:
                sentiment_list = pd.read_excel("/home/gregjroberts1/Projects/Data/LM_Word_List.xlsx", sheet_name=sentiment_class,header=None)
                sentiment_list.columns = ["Word"]
                sentiment_list["Word"] = sentiment_list["Word"].str.lower()
                sentiment_list[sentiment_class] = 1
                sentiment_list = sentiment_list.set_index("Word")[sentiment_class]
                word_list.append(sentiment_list)
            word_list = pd.concat(word_list, axis=1, sort=True).fillna(0)

            #Let's reindex by negative words, as well as drop na and calc the percent frequency
            counts = counts.set_index(["Company", "Time Period", "Word"])["Count"].unstack().transpose().fillna(0)
            tf_percent = (counts / counts.sum())
            negative_words = word_list[word_list["Negative"] == 1].index
            negative_frequency = tf_percent.reindex(negative_words).dropna()

            if (negative_frequency.sum(0)[2]== 0) or (negative_frequency.sum(0)[1]== 0):
                YoY=0
            else:
                YoY = ((negative_frequency.sum(0)[2])/(negative_frequency.sum(0)[1])-1)

            if negative_frequency.sum(0)[0]== 0:
                CAGR3=0
            else:
                CAGR3=((negative_frequency.sum(0)[2])/(negative_frequency.sum(0)[0]))**(1/3.0)-1

            data_comp = data_comp.append({'Company': pair_values[0],
                                  'YoY Chg':YoY, '3yr-CAGR': CAGR3}, ignore_index=True)
        except ValueError:
            pass
    return data_comp

'''Build our dataframe from wikipedia list of companies and respective CIK code table'''
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp_table = scrape_wiki(url).find('table')
data_df = pd.read_html(str(sp_table))[0]
data_df['Security'].replace({'Alphabet Inc. (Class A)': 'Alphabet Inc.'}, inplace=True)

#removing duplicative ticker and newly added with Amcor lacking sufficient filings
#or names like Citigroup and PSEG had filings that were too large to process, exceeding
#lxml threshholds
remove_tickers = data_df[data_df['Security'].isin(['Alphabet Inc. (Class C)', 'Amcor plc',
    'Citigroup Inc.', 'Consolidated Edison',
    'Corteva','Dow Inc.','Discovery, Inc. (Class C)','Entergy Corp.',
    'Exxon Mobil Corp.','Huntington Bancshares','Public Service Enterprise Group (PSEG)'
    'Realty Income Corporation'])].index
data_df.drop(remove_tickers, inplace=True)
#standardize our CIK and make them all 10 digits in length
#in order to make it easier to access the documents from the SEC Edgar site
data_df['CIK']=data_df['CIK'].apply(lambda x: '{0:0>10}'.format(x))
data = data_df[['Symbol','Security','GICS Sector', 'GICS Sub Industry', 'CIK']]
with open('/home/gregjroberts1/Projects/Data/spx_wiki_table.p', 'wb') as f:
    pickle.dump(data, f)
ticker_dict = data.set_index('Security').to_dict()['CIK']

get_dataframe = build_dataset(ticker_dict)
get_dataframe

with open('/home/gregjroberts1/Projects/Data/comp_builder_update.p', 'wb') as f:
    pickle.dump(get_dataframe, f)
#save down our final dataset
with open('/home/gregjroberts1/Projects/Data/spx_word_analysis.p', 'wb') as f:
    pickle.dump(get_dataframe, f)
