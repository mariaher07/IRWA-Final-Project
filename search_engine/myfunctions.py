
from __future__ import print_function, division
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy
import scipy
import scipy.spatial
import pandas
import gensim
import re

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import preprocess_string



import string
import regex

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer

#PREPROCESSING
def stemming (line):
  from nltk.stem import WordNetLemmatizer 
  
  lemmatizer = WordNetLemmatizer() 
   
  stemmer = SnowballStemmer("english" , ignore_stopwords=True)
  
  split_text = line.split()
  final =[]
  for word in split_text:
    lemma_word = lemmatizer.lemmatize(word)
  #  final.append(lemma_word)
    final.append(stemmer.stem(lemma_word))

  final = ' '.join(final)
  return final


abbr_dict={
    "aren't": "are not",
    "won't": "will not",
    "isn't": "is not",
    "don't": "do not",
    "doesn't": "does not",

    "y'all" : "you all",

    "what's":"what is",
    "what're":"what are",
    "who's":"who is",
    "who're":"who are",
    "where's":"where is",
    "where're":"where are",
    "when's":"when is",
    "when're":"when are",
    "how's":"how is",
    "how're":"how are",

    "i'm":"i am",
    "we're":"we are",
    "you're":"you are",
    "they're":"they are",
    "it's":"it is",
    "he's":"he is",
    "she's":"she is",
    "that's":"that is",
    "there's":"there is",
    "there're":"there are",

    "i've":"i have",
    "we've":"we have",
    "you've":"you have",
    "they've":"they have",
    "who've":"who have",
    "would've":"would have",
    "not've":"not have",

    "i'll":"i will",
    "we'll":"we will",
    "you'll":"you will",
    "he'll":"he will",
    "she'll":"she will",
    "it'll":"it will",
    "they'll":"they will",

    "isn't":"is not",
    "wasn't":"was not",
    "aren't":"are not",
    "weren't":"were not",
    "can't":"can not",
    "couldn't":"could not",
    "don't":"do not",
    "didn't":"did not",
    "shouldn't":"should not",
    "wouldn't":"would not",
    "doesn't":"does not",
    "haven't":"have not",
    "hasn't":"has not",
    "hadn't":"had not",
    "won't":"will not",
    '\s+':' ', # replace multi space with one single space
}

def expand_abbreviations(line, abbr_dict):
  split_text = line.split()
  final =[]
  for word in split_text:
    if word in list(abbr_dict.keys()):
      word = abbr_dict[word]
    final.append(word)
  final = ' '.join(final)
  return final



def preprocess_tweet(text):

    #expand abbreviations
    text= expand_abbreviations(text, abbr_dict)

    # remove usernames
    nopunc = regex.sub('@[^\s]+', '', text)

    #remove emojis
    nopunc= nopunc.encode('ascii', 'ignore').decode('ascii')

    # Check characters to see if they are in punctuation
    nopunc = [char for char in nopunc if char not in string.punctuation+"..."]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # convert text to lower-case
    nopunc = nopunc.lower()
    #print(nopunc)

    #stemming words
    nopunc = stemming(nopunc)
    #print(nopunc)

    # remove URLs
    nopunc = regex.sub('((www.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', nopunc)
    nopunc = regex.sub(r'http\S+', '', nopunc)

    #nopunc = regex.sub('@[^\s]+\s', '', nopunc)

    # remove the # in #hashtag
    nopunc = regex.sub(r'#([^\s]+)', r'\1', nopunc)

    # remove repeated characters
    nopunc = nltk.word_tokenize(nopunc)

    # remove stopwords from final word list
    return [word for word in nopunc if word not in stopwords.words('english')]



import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import Counter
from config import *
import os

import time 

import string
import regex


#imports para el inverted index
from collections import defaultdict
from array import array
import math
import collections
from numpy import linalg as la


#TF-IDF
def create_index_tfidf(lines, numDocuments):
    """
    Implement the inverted index and compute tf, df and idf
    
    Argument:
    lines -- collection of Wikipedia articles
    numDocuments -- total number of documents
    
    Returns:
    index - the inverted index (implemented through a python dictionary) containing terms as keys and the corresponding 
    list of document these keys appears in (and the positions) as values.
    tf - normalized term frequency for each term in each document
    df - number of documents each term appear in
    idf - inverse document frequency of each term
    """
        
    index=defaultdict(list)
    tf=defaultdict(list) #term frequencies of terms in documents (documents in the same order as in the main index)
    df=defaultdict(int)         #document frequencies of terms in the corpus
    idf=defaultdict(float)

    indices=lines.index
    for i in indices:      
        terms = preprocess_tweet(lines['full_text'][i])
        tweet_id = int(i)        
        
        ## ===============================================================        
        ## create the index for the **current page** and store it in termdictPage
        ## termdictPage ==> { ‘term1’: [currentdoc, [list of positions]], ...,‘termn’: [currentdoc, [list of positions]]}
        
        ## Example: if the curr_doc has id 1 and his text is 
        ## "web retrieval information retrieval":
        
        ## termdictPage ==> { ‘web’: [1, [0]], ‘retrieval’: [1, [1,4]], ‘information’: [1, [2]]}
        
        ## the term ‘web’ appears in document 1 in positions 0, 
        ## the term ‘retrieval’ appears in document 1 in positions 1 and 4
        ## ===============================================================

        termdictPage={}

        for position, term in enumerate(terms): ## terms contains page_title + page_text
            try:
                # if the term is already in the dict append the position to the corrisponding list
                termdictPage[term][1].append(position) 
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                termdictPage[term]=[tweet_id, array('I',[position])] #'I' indicates unsigned int (int in python)
        
        #normalize term frequencies
        # Compute the denominator to normalize term frequencies (formula 2 above)
        # norm is the same for all terms of a document.
        norm=0
        for term, posting in termdictPage.items(): 
            # posting is a list containing doc_id and the list of positions for current term in current document: 
            # posting ==> [currentdoc, [list of positions]] 
            # you can use it to inferr the frequency of current term.
            norm+=len(posting[1])**2
        norm=math.sqrt(norm)


        #calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in termdictPage.items():     
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append(np.round(len(posting[1])/norm,4))  ## SEE formula (1) above
            #increment the document frequency of current term (number of documents containing the current term)
            df[term] +=1   # increment df for current term
        
        #merge the current page index with the main index
        for termpage, postingpage in termdictPage.items():
            index[termpage].append(postingpage)
            
        # Compute idf following the formula (3) above. HINT: use np.log
    for term in df:
      idf[term] = np.round(np.log(float(numDocuments/df[term])),4)
            
    return index, tf, df, idf








def rankDocuments(terms, docs, index, idf, tf):
    """
    Perform the ranking of the results of a search based on the tf-idf weights
    
    Argument:
    terms -- list of query terms
    docs -- list of documents, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies
    
    Returns:
    Print the list of ranked documents
    """
        
    # I'm interested only on the element of the docVector corresponding to the query terms 
    # The remaing elements would became 0 when multiplied to the queryVector
    docVectors=defaultdict(lambda: [0]*len(terms)) # I call docVectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    queryVector=[0]*len(terms)    

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms) # get the frequency of each term in the query. 
    # Example: collections.Counter(["hello","hello","world"]) --> Counter({'hello': 2, 'world': 1})
    # HINT: use when computing tf for queryVector
    
    query_norm = la.norm(list(query_terms_count.values()))
    
    for termIndex, term in enumerate(terms): #termIndex is the index of the term in the query
        if term not in index:
            continue
                    
        ## Compute tf*idf(normalize tf as done with documents)
        queryVector[termIndex]=query_terms_count[term]/query_norm * idf[term]

        # Generate docVectors for matching docs
        for docIndex, (doc, postings) in enumerate(index[term]):
            # Example of [docIndex, (doc, postings)]
            # 0 (26, array('I', [1, 4, 12, 15, 22, 28, 32, 43, 51, 68, 333, 337]))
            # 1 (33, array('I', [26, 33, 57, 71, 87, 104, 109]))
            # term is in doc 26 in positions 1,4, .....
            # term is in doc 33 in positions 26,33, .....
            
            #tf[term][0] will contain the tf of the term "term" in the doc 26            
            if doc in docs:
                docVectors[doc][termIndex]=tf[term][docIndex] * idf[term]  # TODO: check if multiply for idf

    # calculate the score of each doc
    # compute the cosine similarity between queyVector and each docVector:
    # HINT: you can use the dot product because in case of normalized vectors it corresponds to the cosine siilarity
    # see np.dot
    
    docScores=[ [np.dot(curDocVec, queryVector), doc] for doc, curDocVec in docVectors.items() ]
    docScores.sort(reverse=True)
    resultDocs=[x[1] for x in docScores]
    #print document titles instead if document id's
    #resultDocs=[ titleIndex[x] for x in resultDocs ]
    if len(resultDocs) == 0:
        print("No results found, try again")
        query = input()
        docs = search_tf_idf(query, index)    
    #print ('\n'.join(resultDocs), '\n')
    return resultDocs




def search_tf_idf(query, index,df):
    '''
    output is the list of documents that contain any of the query terms. 
    So, we will get the list of documents for each query term, and take the union of them.
    '''

    query=preprocess_tweet(query)
    docs=set()
    for term in query:
        try:
            # store in termDocs the ids of the docs that contain "term"                        
            termDocs=[posting[0] for posting in index[term]]
            
            # docs = docs Union termDocs
            docs |= set(termDocs)
        except:
            #term is not in index
            pass


    ranked_docs = rankDocuments(query, docs, index, idf, tf) 

    docs=list(ranked_docs)
    output = []


    for i in docs:
      hashtag = []
      if df.loc[i]["entities"]["hashtags"]:
        for k in range(len(df.loc[i]["entities"]["hashtags"])):
          hashtag.append(df.loc[i]["entities"]["hashtags"][k]["text"])
      else:
        hashtag.append("No Hashtags")

      text = regex.sub('@[^\s]+', '', df.loc[i]["full_text"])
      text = regex.sub(r'http\S+', '', text)

    
      output.append([i,text, df.loc[i]["user"]["screen_name"], df.loc[i]["created_at"], hashtag, df.loc[i]["favorite_count"], df.loc[i]["retweet_count"],regex.findall('http\S+', df.loc[i]["full_text"])])

    
    
    return output




def search_tf_idf_YOURSCORE(query, index,df):
    '''
    output is the list of documents that contain any of the query terms. 
    So, we will get the list of documents for each query term, and take the union of them.
    '''

    query=preprocess_tweet(query)
    docs=set()
    for term in query:
        try:
            # store in termDocs the ids of the docs that contain "term"                        
            termDocs=[posting[0] for posting in index[term]]
            
            # docs = docs Union termDocs
            docs |= set(termDocs)
        except:
            #term is not in index
            pass


    ranked_docs = rankDocuments(query, docs, index, idf, tf) 
 
    rankingDataFrame = pd.DataFrame(columns = df.columns.values)

    print(ranked_docs)

    for i in ranked_docs:
      rankingDataFrame = rankingDataFrame.append(df.loc[i])
    
    rankingDataFrame.head()
    rankingDataFrame = rankingDataFrame.sort_values(by='favorite_count', ascending=False)

    orderedDocs = rankingDataFrame.index
    print(orderedDocs)

    #docs=list(ranked_docs)
    output = []


    for i in orderedDocs:
      hashtag = []
      if df.loc[i]["entities"]["hashtags"]:
        for k in range(len(df.loc[i]["entities"]["hashtags"])):
          hashtag.append(df.loc[i]["entities"]["hashtags"][k]["text"])
      else:
        hashtag.append("No Hashtags")

      text = regex.sub('@[^\s]+', '', df.loc[i]["full_text"])
      text = regex.sub(r'http\S+', '', text)

    
      output.append([i,text, df.loc[i]["user"]["screen_name"], df.loc[i]["created_at"], hashtag, df.loc[i]["favorite_count"], df.loc[i]["retweet_count"],regex.findall('http\S+', df.loc[i]["full_text"])])

    
    
    return output




#WORD2VEC
class TwitterArchiveCorpus():
    def __init__(self, df):
        self.dataframe = df
        self.lookup = dict()
        self.dictionary = gensim.corpora.Dictionary(self.iter_texts())
        
    def iter_texts(self):
        current = 0
        for index, row in self.dataframe.iterrows():
          self.lookup[current] = row # BAD BAD BAD
          current += 1
          yield preprocess_tweet(row["full_text"])
            
    def __iter__(self):
        for document in self.iter_texts():
            yield self.dictionary.doc2bow(document)
            
    def __len__(self):
        return self.dictionary.num_docs
            
    def get_original(self, key):
        return self.lookup[key]["full_text"]

    def get_original_username(self, key):
        return self.lookup[key]["user"]["screen_name"]

    def get_original_date(self, key):
        return self.lookup[key]["created_at"]
    
    def get_original_hashtags(self, key):
        hashtag = []
        if self.lookup[key]["entities"]["hashtags"]:
          for k in range(len(self.lookup[key]["entities"]["hashtags"])):
            hashtag.append(self.lookup[key]["entities"]["hashtags"][k]["text"])
        else:
          hashtag.append("No Hashtags") 
        
        return hashtag

    def get_original_likes(self, key):
        return self.lookup[key]["favorite_count"]
        
    def get_original_retweets(self, key):
        return self.lookup[key]["retweet_count"] 

    def get_original_url(self, key):
        return regex.findall('http\S+', self.lookup[key]["full_text"]) 


def get_sentences(docs, verbose=10000):
    #loop over all docs (paragraphs in our case)
    for i, doc in enumerate(docs):
        
        # use nltk.sent_tokenize to split paragraphs into sentences
        for s in nltk.sent_tokenize(doc):
            # preprocess each sentence using gensim (return string not list)
            yield' '.join(preprocess_string(s))
            
        # print progress if needed
        if verbose > 0 and (i + 1) % verbose == 0:
            print(f"Progress: {i + 1}")



def generate_vector_sums():
    for doc in tweets.iter_texts():  # remember me?
        yield gensim.matutils.unitvec(  # DRAGON: hack to make Similarity happy with fullvecs
            sum(
                (w2v[word] for word in doc if word in w2v),
                numpy.zeros(w2v.vector_size)
            )
        )

def search_w2v(query, top, trigger=True):
    query_tokens = preprocess_tweet(query)
    query_vec = gensim.matutils.unitvec(sum((w2v[word] for word in query_tokens if word in w2v), numpy.zeros(w2v.vector_size)))
    print("")

    if(trigger):
        print("\nFormat: Tweet	| Username | Date | Hashtags | Likes | Retweets | Url\n")
    mylist = []
    for doc, percent in w2v_index[query_vec][:top]:
        if(trigger):

          print("Page id"+ str(doc) + ", "+ tweets.get_original(doc) + ", "+ str(tweets.get_original_username(doc)), ", "+ str(tweets.get_original_date(doc))+ ", "+ 
                str(tweets.get_original_hashtags(doc))+ ", "+ str(tweets.get_original_likes(doc))+ ", " + str(tweets.get_original_retweets(doc))+ ", "
                + str(tweets.get_original_url(doc)))
         # print("%.3f" % percent, "=>", tweets.get_original(doc), "\n")
        mytuple = (tweets.get_original(doc),percent)
        mylist.append(mytuple)
    return mylist


