#import sqlite3
#from google.cloud import storage
import pandas as pd
import numpy as np
from datetime import datetime
import mysql.connector
from mysql.connector.constants import ClientFlag
import json
import os
from dotenv import load_dotenv
load_dotenv()

#open a connection to cloudSQL instance and return that engine as an object
def connect_db():
    conn = mysql.connector.connect(user='NULL', password=os.getenv('db_pass'), host=os.getenv('db_host'), database='NULL', client_flags=[ClientFlag.SSL], ssl_ca='ssl/server-ca.pem', ssl_cert='ssl/client-cert.pem', ssl_key='ssl/client-key.pem')
    return conn

def close_connection(conn):
    conn.close()

#hack to utilize DB instead of excel sheets, text processing ideally would be refactored to interact directly with DB
# upon verifying a submission is new, we recreate the table, iterate through our dataframe and insert rows into the table
def update_corpus(df_save):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE Corpus")
    cursor.execute("CREATE TABLE Corpus (corpus VARCHAR(40) PRIMARY KEY, frequency INT);")    
    for row in range(0, df_save.shape[0]):
        cursor.execute("INSERT INTO Corpus (corpus, frequency) VALUES(%s, %s)", (df_save.loc[row]['corpus'], int(df_save.loc[row]['frequency'])))
    conn.commit
    conn.close()

#same idea as above
def update_text(df):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE Text")
    cursor.execute("CREATE TABLE Text (originalText TEXT, url VARCHAR(1000) PRIMARY KEY, cleanText TEXT, tfidf TEXT);")
    for row in range(0, df.shape[0]):
        cursor.execute("INSERT INTO Text (originalText, url, cleanText, tfidf) VALUES(%s, %s, %s, %s)", (df.loc[row]['originalText'], df.loc[row]['url'], str(df.loc[row]['cleanText']), str(df.loc[row]['tfidf'])))
    conn.commit()
    conn.close()

# outline how we would use information from cloudinary to commit image submissions to our DB
#currently unused but would be utilized once image checking API can be accessed
def commit_submission(img_url: str, hash, c_score, response, links: list):
    conn = mysql.connector.connect(user='root', password='NULL', host='NULL', database='NULL')
    cursor = conn.cursor()
    try:
        query = ("""INSERT INTO images (imgUrl, hash, confidenceScore, response, links)
                    VALUES({},{},{},{},{}) """).format(img_url, hash, c_score, response, links)
        cursor.execute(query)
        conn.commit()
    except:
        print("query failed...")
    finally:
        conn.close()

#The two below are function signatures for manual duplicate checking and return responses from DB instance. Because we cannot process images at this time, these are left empty
def check_submission(img_url, phash):
    #Upon upload a phash is produced, this function uses said phash and check against existing entires in the database for duplicates
    return

def return_response(img_url, response):
    #retrieve submission response from db, so that it can be packaged and returned to user
    return






