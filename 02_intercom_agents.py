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


token = 'dsdssds'
ID = 'sddsffd'
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

# ------Configurations------
token= 'sdsfsf='
headers = {'Accept': 'application/json',
           'Authorization': 'Bearer dsfdsfdffdsf='}

# ---dataframe---#




dfadmins = pd.DataFrame(columns=['InsertTimeRecordUTC', 'id', 'email', 'full_name', 'away_mode_enabled',
                                 'auto_reassign_new_conv_to_default_inbox', 'open_count', 'closed_count'])


# ------Fetches------
def fetch_count(fetch_type, user):
    r = requests.get("https://api.intercom.io/counts?type=conversation&count=admin", headers=headers).text
    json_data = json.loads(r.strip())
    for i in json_data['conversation']['admin']:
        if i['id'] == user:
            return i[fetch_type]


# --------------------------------Get all admin objects
today = datetime.date.today().strftime('%Y-%m-%d')
print(today)
r = requests.get("https://api.intercom.io/admins/", headers=headers).text
json_data = json.loads(r.strip())
for i in json_data['admins']:
    dfadmins = dfadmins.append({'InsertTimeRecordUTC': today,
                                'id': i['id'],
                                'email': i['email'],
                                'full_name': i['name'],
                                'away_mode_enabled': str(i['away_mode_enabled']),
                                'auto_reassign_new_conv_to_default_inbox': str(i['away_mode_reassign']),
                                'open_count': fetch_count('open', i['id']),
                                'closed_count': fetch_count('closed', i['id']),
                                'has_inbox_seat':str(i['has_inbox_seat']),
                                'type': i['type'],
                                'team_ids': i['team_ids'] 
                                }, ignore_index=True)

bitesResult = dfadmins

pandas_gbq.to_gbq(bitesResult, dataset_id +'.intercom_agents',projectid, if_exists='replace',credentials=credentials)
