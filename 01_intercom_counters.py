#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 13:10:36 2021

@author: hbenshoshan
!pip install python-intercom
https://pythonrepo.com/repo/intercom-python-intercom-python-third-party-apis-wrappers
"""
 
import requests
from requests.auth import HTTPBasicAuth
import requests, json, pandas as pd, datetime
#from google.cloud import bigquery
from google.oauth2 import service_account
import pandas_gbq


token = 'xxxxx'
ID = 'yyyyyy'
today = datetime.date.today().strftime('%Y-%m-%d')
InsertTimeRecordUTC=today

r = requests.get('https://api.intercom.io/counts', auth = HTTPBasicAuth(token, ID)).text
json_data = json.loads(r.strip()) 

user_counter = json_data['user']['count'] 
company_counter = json_data['company']['count'] 
lead_counter = json_data['lead']['count'] 
segment_counter = json_data['segment']['count'] 
tag_counter = json_data['tag']['count']  
InsertTimeRecordUTC=today

print("Number of users :",user_counter)
print("Number of companies :",company_counter)
print("Number of leads :",lead_counter)
print("Number of segments :",segment_counter)
print("Number of tags :",tag_counter) 
print("InsertTimeRecordUTC :",InsertTimeRecordUTC)


#--------------------------------gbq
credentials = service_account.Credentials.from_service_account_file("/home/ubuntu/bites-ai2/integrations/bites-new-api-ga4-report-f8d733a305f1.json") 
projectid = "bites-new-api-ga4-report"
dataset_id = "integrations"
#--------------------------------End-gbq

#---dataframe---
dfcounters = pd.DataFrame(columns=['InsertTimeRecordUTC','user_counter','company_counter','lead_counter','segment_counter','tag_counter','created_at'])
#--------------------------------Conversation Messages
dfcounters = dfcounters.append    ({'InsertTimeRecordUTC'   : today,
                                    'user_counter'          : user_counter,
                                    'company_counter'       : company_counter,
                                    'lead_counter'          : lead_counter,
                                    'segment_counter'       : segment_counter,
                                    'tag_counter'           : tag_counter, 
                                    'created_at'            : datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    },ignore_index=True)
 

bitesResult = dfcounters

pandas_gbq.to_gbq(bitesResult, dataset_id +'.intercom_counters',projectid, if_exists='append',credentials=credentials)
