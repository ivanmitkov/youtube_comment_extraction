# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:38:45 2021

@author: ivan.mitkov
"""
from googleapiclient.discovery import build 
import pandas as pd 
import texthero as hero
from texthero import preprocessing
from textblob import TextBlob
  
api_key = 'AIzaSyDtMBpPLiPx-Ed2EwqfiLLsV6zP7aUxy3U'
  
def video_comments(video_id): 
    # empty list for storing reply 
    replies = [] 
  
    # creating youtube resource object 
    youtube = build('youtube', 'v3', developerKey=api_key) 
  
    # retrieve youtube video results 
    video_response=youtube.commentThreads().list( 
        part='snippet, replies', 
        videoId=video_id ,
        maxResults = 100
    ).execute() 

    next_page_token = True
    
    # dstacked f for the comments and responses
    df_stacked = pd.DataFrame()

    # iterate video response 
    while next_page_token: 
        
        # extracting required info 
        # from each result object  
        
        for item in video_response['items']: 
            
            
            # Extracting comments

            df = pd.DataFrame(item['snippet']['topLevelComment']['snippet'])
            df['etag'] = item['etag']
            df['id'] = item['id']
            df['kind'] = item['kind']
            df['canReply'] = item['snippet']['canReply']
            df['isPublic'] = item['snippet']['isPublic']
            df['totalReplyCount'] = item['snippet']['totalReplyCount']
            df['videoId'] = item['snippet']['videoId']
            
            df_stacked = df_stacked.append(df)
            
            # counting number of reply of comment 
            replycount = item['snippet']['totalReplyCount'] 
  
            # if reply is there 
            if replycount>0: 
                
                # iterate through all reply 
                for reply in item['replies']['comments']: 
                    
                    # Extract reply 
              
                    # Store reply
                    df = pd.DataFrame(reply['snippet'])
                    df['etag'] = reply['etag']
                    df['id'] = reply['id']
                    df['kind'] = reply['kind']
                    df_stacked = df_stacked.append(df)


  
            # empty reply list 
            replies = [] 
  
        # Again repeat 
        if 'nextPageToken' in video_response: 
            video_response = youtube.commentThreads().list( 
                    part = 'snippet,replies', 
                    videoId = video_id,
                    maxResults = 100 ,
                    pageToken = video_response['nextPageToken']
                ).execute() 
            next_page_token =  True
        else: 
            next_page_token = False
            break
    df_stacked.reset_index(drop = True) 
    return df_stacked
  
# Enter video id: https://youtu.be/ZQ6klONCq4s
video_id = "ZQ6klONCq4s"

# Call function 
result = video_comments(video_id)

# vectorize the comments and tfidf
custom_pipeline = [preprocessing.fillna,
                   preprocessing.lowercase,
                   preprocessing.remove_whitespace,
                   preprocessing.remove_diacritics,
                   preprocessing.remove_brackets,
                   preprocessing.remove_stopwords,
                   preprocessing.remove_punctuation
                  ]
result['clean_text'] = hero.clean(result['textOriginal'], custom_pipeline)
result['tfidf'] = (hero.tfidf(result['clean_text']))
cleaned_tfidf = hero.tfidf(result['clean_text'])
totimp = []
for i in cleaned_tfidf:
    totimp.append(sum(i))
result['importance'] = totimp

# sentiment
def senti_polarity(x):
    return TextBlob(x).sentiment[0]  

def senti_subjectivity(x):
    return TextBlob(x).sentiment[1]  

result['polarity'] = result['clean_text'].apply(senti_polarity)
result['subjectivity'] = result['clean_text'].apply(senti_subjectivity)
result.reset_index(drop = True, inplace = True)

# top words
top_words = pd.DataFrame(hero.top_words(result['clean_text'])).reset_index()
top_words.columns = ['word', 'occurancies']

# save for dashy
writer = pd.ExcelWriter('output.xlsx')
result.to_excel(writer,'Sheet1')
top_words.to_excel(writer,'Sheet2')
writer.save()