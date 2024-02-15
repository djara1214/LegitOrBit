import spacy
from string import punctuation as pun
import pandas as pd
import math
import trafilatura as traf
import json
import ast

import warnings
#warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')#supress gensim warning 'aliasing chunkize to chunkize_serial' its annoying...
warnings.filterwarnings(action='ignore')

from gensim.summarization import summarize
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import requests
import validators
import os
from dotenv import load_dotenv
from db import connect_db, close_connection, update_corpus, update_text
nlp=spacy.load("en_core_web_sm")

###########################################GLOBALS AND API KEYS#############################################
global count #how many terms I want to search for on google
count=4

def configure():
    load_dotenv()
load_dotenv()

#----------API code for ClaimBusters----------
api_base_url = "https://idir.uta.edu/claimbuster/api/v2/"
single_endpoint = "score/text/"
multiple_endpoint = "score/text/sentences/"
request_headers = {"x-api-key": os.getenv('claimBusters_api_key')}
#----------API code for Gooogle is in def submit()----------



def text_to_frame():
    """
    Load a table from the database into a Pandas DataFrame for text processing.

    This function establishes a connection to the database, reads the 'Text' table,
    and loads its contents into a DataFrame. It then closes the connection and returns
    the DataFrame.

    Returns:
    DataFrame: A DataFrame containing the contents of the 'Text' table from the database.
    """
    conn = connect_db()
    frame = pd.read_sql("SELECT * FROM Text;", conn)
    close_connection(conn)
    return frame


def corpus_to_frame():
    """
    Load a table from the database into a Pandas DataFrame for text processing.

    This function establishes a connection to the database, reads the 'Corpus' table.

    Returns:
    DataFrame: A DataFrame containing the contents of the 'Corpus' table from the database.
    """
    conn = connect_db()
    frame = pd.read_sql("SELECT * FROM Corpus;", conn)
    close_connection(conn)
    return frame

df = text_to_frame()
df_save = corpus_to_frame()


def get_article(url):
    """
    Get the full article from a URL, or return the original input if it's not a URL.

    This function takes a user query as input, which can be either a URL (incomplete or in full form) 
    or text. If the input is a URL, it retrieves the full document from the URL using the Trafilatura 
    library. If the input is not a URL, it simply returns the original input.

    Parameters:
    url (str): The user query, which can be a URL or text.

    Returns:
    str: The full article content if a URL was provided and a valid document was extracted, 
         otherwise returns the original input.
    """
    checked=check_if_site(url)
    if (validators.url(checked)):
        downloaded = traf.fetch_url(url)
        doc= traf.extract(downloaded)
        return doc
    else:
        return url

####################ALL FUNCTIONS FOR SIMILAR DOCUMENTS AND INPUT CLEANING####################

def clean_process_txt(txt):
    """
    Cleans the document by extracting lemmas of proper nouns, verbs, adjectives, nouns, and numbers.

    Uses Spacy to remove stop words. In the elif block, it includes all parts of speech 
    that are ignored in stop text and punctuation in the first if statement.

    Parameters:
    txt (str): The raw article text or raw user query to be cleaned.

    Returns:
    list: A list of lemmas, where repeated terms are allowed.    
    """
    result=[]
    pos_tag = ['PROPN', 'VERB', 'ADJ', 'NOUN','NUM']    #pos_tag= part of speech tag 
    wrds=nlp(txt.lower())               # make it all lowercase
    for char in wrds:
        if (char.text in nlp.Defaults.stop_words or char.text in pun):
            continue                    #skip all stopping words and punctuation
        elif(char.pos_ in pos_tag):
            result.append(char.lemma_)  #create list of tokens (lemmas only )
    return result

def tf(cln_txt):
    """
    Finds the count of occurrences of each lemma in the document.

    Parameters:
    lemmas (list): A list of lemmas.

    Returns:
    dict: A dictionary where keys are lemmas and values are the number of times each lemma occurs in the document.
    """
    result={}
    for wrds in cln_txt:
        if wrds not in result:  #if word is new give value of 1
            result[wrds]=1
        elif wrds in result:
            result[wrds]+=1     #o/w add 1
    return result 

def add_corpus_mod(lemmas):
    """
    Updates the corpus dataframe. If a word is already in the corpus, it adds 1 to its frequency.
    If the word is not present in the corpus, it adds the word to the corpus with a frequency of 1.

    Parameters:
    lemmas (dict): A dictionary of lemmas. This function will only consider the keys of the dictionary.

    Returns:
    None: This function does not return anything. It updates the corpus dataframe in place.
    """
    global df_save
    listof_keys=list(lemmas.keys())
    listof_keys_set=list(set(listof_keys))
    for i in listof_keys_set:
       #If the corpus does not have the word
        if(not(i in df_save['corpus'].unique())):
                #add word and frequency to corpus
                dfTemp=pd.DataFrame({'corpus':[i],'frequency':[1]})
                df_save=pd.concat([df_save,dfTemp],ignore_index=True,axis=0)
        else:#df has the word
            #curr_freq= frequency value at word location
            curr_freq=df_save[df_save['corpus']==i]['frequency'].values[0]
                # old frequency + new freqency
            curr_freq+=1
                #update the new frequency to olds location
            df_save.loc[df_save.corpus ==i ,['frequency']]= curr_freq


def calc_tf(diction):
    """
    Calculates the term frequency for each lemma in the document.

    Term frequency is calculated as (number of times the lemma appears in the document) / (total number of words).

    Parameters:
    diction (dict): A dictionary with lemmas as keys and their frequencies in the document as values.

    Returns:
    dict: An updated dictionary where the values represent the term frequency of each lemma.
    """
    denominator= len(diction)
    for word,keyz in diction.items():
        diction[word]= keyz/denominator
    return diction


def calc_idf(diction):
    """
    Calculates the inverse document frequency (IDF) for each lemma.

    IDF is calculated as (total number of documents) / (frequency of the lemma in the corpus).

    Parameters:
    diction (dict): A dictionary with lemmas as keys representing the term frequency in the document.

    Returns:
    dict: A dictionary where keys are lemmas and values are their IDF.
    """
    tempdict={}
    for i in df_save.index:
        tempdict[df_save['corpus'][i]]= df_save['frequency'][i]
    numerator=len(df.axes[0])
    for word in diction.keys():
        freq=1
        for i in tempdict.keys():
            if word==i:
                freq= tempdict[word]
        diction[word]=1+ math.log(float(numerator)/freq)
    return diction



def tfidf(dictiontf,dictionidf):
    """
    Combines term frequency (TF) and inverse document frequency (IDF) to calculate TF-IDF for each lemma.

    Parameters:
    dictiontf (dict): A dictionary where keys are lemmas and values are their term frequencies.
    dictionidf (dict): A dictionary where keys are lemmas and values are their inverse document frequencies.

        This code was adapted from:
        Ashwin N
        This includes code for calculating term frequency, Inverse Document Frequency, and the TFIDF itself.
        https://medium.com/@ashwinnaidu1991/creating-a-tf-idf-model-from-scratch-in-python-71047f16494e
        Accessed: Oct. 2022

    Returns:
    dict: A dictionary where keys are lemmas and values are their TF-IDF scores.
    """
    tempdict={}
    for word, keyz in dictiontf.items():
        tempdict[word]= keyz * dictionidf[word]
    return tempdict

def update_tfidf_ALL():
    """
    Updates all TF-IDF values after a new document is added to the corpus.

    This function recalculates TF-IDF values for all lemmas in the corpus 
    since a new document may introduce new words. It operates on global dataframes.

    Parameters:
    None: This function does not take any input.

    Returns:
    None: This function does not return anything. It updates the TF-IDF values in global dataframes.
    """
    global df
    global df_save
    tempdict_corp={}
    num_of_docs=(df.shape[0])-1
    for i in range(num_of_docs):
        tf_dict={}
        tf_dict=df['cleanText'][i]#term freq.
        tf1=calc_tf(eval(tf_dict))
        tf1=str(tf1)
        curr_idf=calc_idf(eval(tf_dict))
        update_tdidf=tfidf(eval(tf1),curr_idf)
        df.at[i,'tfidf']=update_tdidf


def cosine_similarity_fun(vec1, vec2):
  """
    Creates vectors of TF-IDF values.

    Documents with similar content will have TF-IDF vectors that are close to each other,
    indicating similarity. Conversely, if the vectors point away from each other, the documents
    are not similar.

    Parameters:
    vec1 (list): The first vector.
    vec2 (list): The second vector.

    Returns:
    float: The cosine similarity between the two vectors.
  """
#************************************************************************************************************
    #This code is from ChatGPT, accessed Dec 8,2022
    #Query was 'write a python script to calculate cosine similarity where tfidf is held in dictionary values'
#************************************************************************************************************
  # Calculate dot product
  dot_product = 0
  for key, value in vec1.items():
    if key in vec2:
      dot_product += value * vec2[key]

  # Calculate L2 norm for each vector
  vec1_norm = 0
  for value in vec1.values():
    vec1_norm += value ** 2
  vec1_norm = math.sqrt(vec1_norm)

  vec2_norm = 0
  for value in vec2.values():
    vec2_norm += value ** 2
  vec2_norm = math.sqrt(vec2_norm)

  # Return cosine similarity
  return dot_product / (vec1_norm * vec2_norm)


def get_similar_doc(last_tfidf,df):
    """
    Checks saved document TF-IDF values for matches.

    This function iterates through all documents in the saved_documents_df DataFrame and compares each document's TF-IDF values 
    with the TF-IDF values of the new entry. If a match is found with a similarity of at least 95%, it returns a tuple with True 
    as the first element and the index of the matching document as the second element. Otherwise, it returns a tuple with False 
    as the first element.

    Parameters:
    last_tfidf (dict): The TF-IDF values of the new entry.
    df (DataFrame): The DataFrame where documents are saved.

    Returns:
    tuple: If a match is found, returns (True, index of match found); otherwise, returns (False,).
    """
    for i in df.index:
        curr=str(df['tfidf'][i])
        curr_dict=eval(curr)
        cos=cosine_similarity_fun(last_tfidf,curr_dict)
        if cos >= 0.95:
            return True,i
    return False


def new_entry(doc,url):
    """
    Calls all functions for TF-IDF calculation for the new entry.

    This function computes the TF-IDF values for the new entry, checks if it is similar to any previously saved documents,
    and updates the TF-IDF values of all documents if necessary. If a similar document is found, it returns the saved URL 
    information associated with the previous user query. If no similar document is found, it updates the saved documents 
    with the new entry and returns -1 to indicate the occurrence.

    Parameters:
    doc (str): The full article or user query.
    url (str): The URL submitted for the new entry.

    Returns:
    str or int: If a match is found, returns the saved URL of the previous user query. 
               If no match is found, returns -1 to indicate the occurrence.
    """
    matched_doc=''
    global df
    cleaned_doc=clean_process_txt(doc)
    #dictionary value of frequencies
    doc_tf_Dict=tf(cleaned_doc)
    doc_for_idf=str(doc_tf_Dict)
    #update corpus
    add_corpus_mod(doc_tf_Dict)

    tf0=calc_tf(doc_tf_Dict)
    idf0=calc_idf(eval(doc_for_idf))
    fin_tfidf=tfidf( tf0,idf0)

    res=get_similar_doc(fin_tfidf,df)
    if not(res) :
        dftemp=pd.DataFrame({'originalText':[doc],'url':[url],'cleanText': [tf(cleaned_doc)],'tfidf':[fin_tfidf]})#if i dont recall function i get weird numbers...idk why
        df=pd.concat([df,dftemp],ignore_index=True,axis=0)
        update_tfidf_ALL() 
        return -1
    else:
        return df['url'][res[1]] 
    
###########################     ALL CODE FOR API CALLS      ####################################
#________________________________CLAIM BUSTERS________________________________

def submit_to_claimbust(url):
    """
    Send a query to Claim Busters and receive the results.

    This function sends a query to the Claim Busters service and receives the results in JSON format.

    Parameters:
    url (str): The query string to be sent to Claim Busters. This can be a single line or multiline string.

    Returns:
    dict: The JSON results received from Claim Busters.
    """
    site_tra=site_text=userQuery=''
    #If its a site, extract article
    if validators.url(url):
        site_tra=traf.fetch_url(url)
        site_text=traf.extract(site_tra)
        userQuery=summarize(site_text)
    else:
        userQuery=url


    #if it only 1 line- e.g can be user text submssion
    if '\n' in userQuery:
        api_endpoint = api_base_url + multiple_endpoint + userQuery
        response = requests.get(url=api_endpoint, headers=request_headers)
        res_body = response.json()
        return res_body
    #if there are multiple lines in submission to ClaimB    e.g. cleaned article summarized by gensim, gensim separtes large text into multiple lines 
    else:
        api_endpoint = api_base_url + single_endpoint + userQuery
        response = requests.get(url=api_endpoint, headers=request_headers)
        res_body = response.json()
        return res_body

#________________________________GOOGLE________________________________


def check_if_site(url):
    """
    Check if the user input is a URL.

    For Google Fact Checker, if the input is not in a perfect URL format (e.g., www.google.com), 
    it will be converted to the format https://www.google.com to limit issues. 
    This function is also used to check matches from Google since they return full format. 
    If the function adds 'https://' and the input is not a valid site, it's considered a text submission, 
    so the original input is returned.

    Parameters:
    user_input (str): The user input string, which can be text or a website with or without 'https://' prefix.

    Returns:
    str: If the input is not a website, returns the original text. 
         If it is a website, returns the full format (e.g., https://www.example.com).
    """
    http='https://'
    updated_url=''
    if len(url) > 2048: #max url length can be 2048 characters, if exceeds its long user submission NOT URL
        return url
    else:
        if url[0:8]== http:
            return url
        else:
            updated_url=http+url
            if not(validators.url(updated_url)):
                return url
            else:
                return updated_url

def submit(userQuery):
    """
    Search for similar documents using a multiple keyword search in Google.

    This function takes a query string as input, which should be in the format of comma-separated words
    with a space after each comma (e.g., 'i, am, input, to, google' )
    This format simulates a multiple keyword search in Google.
    It returns a dictionary of similar documents with metadata retrieved from the Google API.

            This code is from:
            AlecM33
            https://github.com/AlecM33/fact-check-bot/blob/master/bot.py
            Accessed: Nov. 2022
            Note: I hated the google doumentation, but his code was spot on with the call to the API

    Parameters:
    userQuery (str): The query string containing comma-separated keywords.

    Returns:
    dict: A dictionary containing similar documents with metadata retrieved from the Google API.
    """
    try:
        # attemt call to Google's fact check API 
        factCheckService = build("factchecktools", "v1alpha1", developerKey=os.getenv('google_api_key'))
       
        request = factCheckService.claims().search(query=userQuery)
        response = request.execute()
        # print(type(response))
        return response

        # TODO more specifically handle problems with Google's API
    except HttpError as err:
        print (err)

def submit_to_google(user_input):
    """
    Format metadata received from the Google API.

    This function takes a full text as input, which can be retrieved from a URL pull or user text.
    It then submits the text to a function (    submit()    ) to fetch similar documents with metadata 
    from the Google API.
    The output is a JSON formatted dictionary of similar documents with metadata.

    Parameters:
    user_input (str): The full text that will be submitted to fetch similar documents.

    Returns:
    dict: A dictionary containing similar documents with metadata retrieved from the Google API
    or text responce stating that no similar articles or reading material has been found.
    """
    textsub=''
    urlsub=''

    userQuery = check_if_site(user_input)       #Check and modify Query if its a site
    if validators.url(userQuery):
        google_responce=submit(userQuery)       #if its a full URL, send to google
        if len(google_responce)!=0:
            return google_responce              #if it returns "something" return to user
    # else:                                     #if no responce, treat as text submission
    site_tra=traf.fetch_url(user_input)
    extarcted_article=traf.extract(site_tra)
    if extarcted_article== None:
        textsub=userQuery
    #make list of most frequent words   e.g. 'i, am, input, to, google' 
    cleaned=clean_process_txt(extarcted_article) #output list of lemmas
    dict_clean=tf(cleaned)                      #output : dict {lemma:count}
                                                #define variables
    submit_dict={}
    submit_list=[]
    submit_str=''
    submit_dict= dict(sorted(dict_clean.items(), key=lambda item:item[1],reverse=True))# sort the dictionary greast to least
    submit_list=list(submit_dict.keys())[0: count]      # make list of n wanted terms 
    submit_str=', '.join(submit_list)                   # format to string for query entry
    google_responce=submit(submit_str)                  #Nothing was found
    if len(google_responce) !=0:
        pretty_google_responce=json.dumps(google_responce,indent=4)
        return pretty_google_responce
    return "We are sorry, but no similar articles or reading material has been found"


def check_google_is_exact(result,url):
    """
    Check if the user's URL query is present in Google's database.

    This function parses through the JSON returned by Google and checks if the first URL returned 
    matches the user-entered URL for claim checking.

    Parameters:
    result (dict): The JSON returned by Google containing search results.
    url (str): The user's URL query.

    Returns:
    bool: True if a match is found (the first URL matches the user's URL entry), False otherwise.
    """
    try:
        claim=result['claims'][0]
        review= claim["claimReview"][0]
        url=check_if_site(url)
        if ("url" in review):
            url_res = review["url"]
        if url==url_res:
            return True
    except: 
        return False
    return False

def submit_both_factAPI(query):
    """
    Call both fact-checking APIs together.

    This function takes a user query as input and calls both the Google and ClaimBusters fact-checking APIs.
    It returns a tuple containing the results from both APIs.

    Parameters:
    query (str): The user query to be fact-checked.

    Returns:
    tuple: A tuple containing the results from both APIs. 
           The first element is the result from Google fact-checking API,
           and the second element is the result from ClaimBusters API.

    Notes:
    The weird else statement is included as a reminder to   OURSELVES   that similar documents will be returned regardless of the outcome.
    """
    claimB_result=''
    google_result=submit_to_google(query)
    url_to_check= check_if_site(query)
    # length=len(google_result)
    if len(google_result) !=0:
        if not check_google_is_exact(google_result,query):
            claimB_result=submit_to_claimbust(url_to_check)
        else:
            claimB_result=submit_to_claimbust(url_to_check)
    else:
        claimB_result=submit_to_claimbust(url_to_check)

    return google_result,claimB_result


def claimB_score(from_claimB):
    """
    Average the score from ClaimBusters.

    This function takes the raw response from the ClaimBusters API as input. Each line in the multi-line API call
    gives an individual score. The function calculates the average of all scores.

    Parameters:
    from_claimB (str): The raw response from the ClaimBusters API.

    Returns:
    float: The average of all scores from ClaimBusters.
           If the raw response is blank, returns a negative number to indicate that no score was available.
    """
    try:
        results=from_claimB ['results']
        # all_data=results[1]
        scoresum=0.0
        amount=len(results)
        for i in range(amount):
            data=results[i]
            scoresum+=data['score']
            score=scoresum/amount
        return round(score,2)
    except:
        return -1           #return neg number if no score is found...aka blank responce


#Makes nested dictionary with information from  google API {0:{publisher,rating,url,summary},1:{......}...} length dictated by GLOBAL varaible  google_return_amt
#input: raw google api responce
#output: if responce was found from google: Nested dictionary with responce info, if no responce, empty dictionary
    # ***********
    #   This code was adapted from:
    #   AlecM33
    #   https://github.com/AlecM33/fact-check-bot/blob/master/bot.py
    #   Accessed: Nov. 2022
    # Note: I had some trouble unpacking the json, so his code had the correct calls to make grab the data I needed
    # *************
global google_return_amt
google_return_amt=2
def get_google_results(result):
    returndict={}
    try:
        if "claims" in result.keys() and len(result["claims"]) > 0:
            for i in range(google_return_amt):
                tempdict={}
                claim=result['claims'][i]
                current = claim["claimReview"][0]
                if ("publisher" in current):
                    publisher = current["publisher"]["name"]
                if ("textualRating" in current):
                    rating = current["textualRating"]
                if ("url" in current):
                    url = current["url"]
                    try:
                        site_tra=traf.fetch_url(url)
                        site_text=traf.extract(site_tra)
                        summary=summarize(site_text,word_count=50)
                    except:
                        summary=None
                tempdict={'publisher':publisher,'rating':rating,'url':url,'summary':summary}
                returndict[i]=tempdict
            return returndict
        else:
            return returndict #return empty dictionary if nothing found
    except:
        return returndict

#main call for the endpoints to use         hence the name, its magic https://media.giphy.com/media/LR5UmQvLDDRqp9BI9x/giphy.gif
def the_magic(url):

    doc= get_article(url)
    possible_similar= new_entry(doc,url)
    if possible_similar != -1:                              #if match was found in cosine sim, rewrite url, and use saved info
        url=possible_similar
    responce_both=submit_both_factAPI(url)
    google_output=responce_both[0]                          #get google results from tuple
    google_result_dict=get_google_results(google_output)
    #SCORE FOR CLAIM BUSTERS AVG OF ALL
    claimB_output=responce_both[1]                          #get ClaimB result from tuple
    claimB_avgScore=claimB_score(claimB_output)
    #####################################################################################       returns to user         #######################################################################################
    claimB_TO_USER=''
    google_TO_USER=''
    if claimB_avgScore > 0:
        claimB_TO_USER='According to Claim Busters, your query had a score of {0}'.format(claimB_avgScore)
    else:
        claimB_TO_USER='We are sorry but Claim Busters can not calculate a score based on your input. Please try again'
    #Makes nested dictionary with information from  google API {0:{publisher,rating,url,summary},1:{......}...} length dictated by GLOBAL varaible  google_return_amt
    if len(google_result_dict)>0:
        if len(google_result_dict[0]['summary']) >1:
            google_TO_USER="According to {0}, the following article has a rating of {1} , The Following is a summary: {2}  ".format(google_result_dict[0]['publisher'],google_result_dict[0]['rating'],google_result_dict[0]['summary'])
        else:
            google_TO_USER="According to {0}, the following article has a rating of {1} , and we are sorry but there is no summary available ".format(google_result_dict[0]['publisher'],google_result_dict[0]['rating'])
    else:
        google_TO_USER='We are sorry there are no similar articles realted to your query'
    return claimB_TO_USER, google_TO_USER, claimB_avgScore




## updates the DB using current data frames -- all code that works with data frames should be re-worked to interact directly with DB...extension*
update_corpus(df_save)
update_text(df)

#these are all URLS used for testing inside file
# url='https://www.nydailynews.com/news/politics/new-york-elections-government/ny-election-2022-nyc-early-voting-numbers-hochul-zeldin-20221031-pitbdklq2jaxlbqm6pv2br3y4q-story.html'
#  url='https://www.npr.org/2022/12/08/1141546218/supreme-court-leaks-reverend-rob-schenk-dobbs-hobby-lobby'
# url='gigafact.org/fact-briefs/does-gender-affirming-health-care-have-positive-outcomes-for-transgender-youths'
# url='https://www.foxnews.com/politics/abortions-since-roe-v-wade'
# url='https://apnews.com/article/russia-ukraine-kyiv-europe-8ade16c890a92e353f11cae01a01e498'
# url='https://www.politico.com/news/2022/12/08/raphael-warnock-georgia-runoff-00072999'
# url='gigafact.org/fact-briefs/does-gender-affirming-health-care-have-positive-outcomes-for-transgender-youths'
# url='https://www.usatoday.com/story/news/factcheck/2022/06/23/fact-check-handguns-legal-chicago/7625983001/'
#old above-new tested below all below dont have google
#url='https://www.npr.org/2022/12/14/1142605124/federal-student-aid-reverses-course-on-some-relief-approvals'
# url='https://www.politico.com/news/2022/12/14/house-vote-stopgap-funding-00073879'
# url='https://www.kcra.com/article/sandy-hook-shooting-10-years-later/42240837'
# url='https://www.sfgate.com/politics/article/san-francisco-treating-mental-illness-17641213.php'
# url='https://abcnews.go.com/Politics/house-approves-funding-extension-avert-government-shutdown-buying/story?id=95324941'
#url='https://abcnews.go.com/Politics/faucis-exit-amid-surge-covid-politics-note/story?id=95164845'
# url='https://www3.forbes.com/leadership/the-u-s-states-people-are-fleeing-and-the-ones-they-are-moving-to-version-5-ifs-vue-mn-wnb/'