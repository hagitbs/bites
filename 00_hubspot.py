#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:16:47 2021

@author: hbenshoshan
!pip install hubspot3
"""
from __future__ import with_statement

import pandas as pd
from hubspot3 import Hubspot3
#from hubspot3.companies import CompaniesClient
#from hubspot3.contacts  import ContactsClient
#import requests, json, pandas as pd, datetime
#from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import storage 
import pandas_gbq
import os

 
#--------------------------------gbq
credentials = service_account.Credentials.from_service_account_file("/home/ubuntu/bites-ai2/integrations/bites-new-api-ga4-report-f8d733a305f1.json")
#credentials = service_account.Credentials.from_service_account_file("/Users/hbenshoshan/Downloads/bites-new-api-ga4-report-c038f87b9c89.json")
projectid = "bites-new-api-ga4-report"
dataset_id = "integrations"
#--------------------------------End-gbq
bucket_name = "bites_hubspot"

API_KEY='ab1a9ce2-e69b-471c-8acc-47915929bff4'

client = Hubspot3(api_key=API_KEY)



params = {
    "includePropertyVersions": "true"
}


# all of the clients are accessible as attributes of the main Hubspot3 Client
 

all_companies = client.companies.get_all()
companies_df=pd.DataFrame(all_companies)
bitesResult = companies_df
#pandas_gbq.to_gbq(bitesResult, dataset_id +'.hubspot_companies',projectid, if_exists='replace',credentials=credentials)
bitesResult.to_csv('hubspot_companies.csv', encoding='utf-8')

gs_client = storage.Client.from_service_account_json(json_credentials_path='/home/ubuntu/bites-ai2/integrations/bites-new-api-ga4-report-f8d733a305f1.json') 

#gs_client = storage.Client.from_service_account_json(json_credentials_path='/Users/hbenshoshan/Downloads/bites-new-api-ga4-report-c038f87b9c89.json')
bucket = gs_client.get_bucket(bucket_name) 
object_name_in_gcs_bucket = bucket.blob('hubspot_companies.csv') 
object_name_in_gcs_bucket.upload_from_filename('hubspot_companies.csv')

 
 


client = Hubspot3(api_key=API_KEY) 
#Start of day: 	1633305600	Monday, October 4, 2021 12:00:00 AM

#a= client.companies.get_recently_created(since=1633305600)



all_clients=client.contacts.get_all()
clients_df=pd.DataFrame(all_clients) 
bitesResult = clients_df
#pandas_gbq.to_gbq(bitesResult, dataset_id +'.hubspot_clients',projectid, if_exists='replace',credentials=credentials)
bitesResult.to_csv('hubspot_clients.csv', encoding='utf-8')
object_name_in_gcs_bucket = bucket.blob('hubspot_clients.csv') 
object_name_in_gcs_bucket.upload_from_filename('hubspot_clients.csv')
 
#a= client.companies.get_recently_created(since='20210-10-01')
 
all_deals= client.deals.get_all()
deals_df=pd.DataFrame(all_deals) 

all_engagements= client.engagements.get_all()
engagements_df=pd.DataFrame(all_engagements) 

all_contact_lists= client.contact_lists.get_contact_lists()
contact_lists_df=pd.DataFrame(all_contact_lists) 

all_email_events= client.email_events.get_all_campaigns_ids()
email_events_df=pd.DataFrame(all_email_events) 


all_forms= client.forms.get_all()
forms_df=pd.DataFrame(all_forms) 

all_crm_pipelines= client.crm_pipelines.get_all()
crm_pipelines_df=pd.DataFrame(all_crm_pipelines)  

 
bitesResult = deals_df
#pandas_gbq.to_gbq(bitesResult, dataset_id +'.hubspot_deals',projectid, if_exists='replace',credentials=credentials)
bitesResult.to_csv('hubspot_deals.csv', encoding='utf-8')
object_name_in_gcs_bucket = bucket.blob('hubspot_deals.csv') 
object_name_in_gcs_bucket.upload_from_filename('hubspot_deals.csv')
 

bitesResult = engagements_df
#pandas_gbq.to_gbq(bitesResult, dataset_id +'.hubspot_engagements',projectid, if_exists='replace',credentials=credentials)
bitesResult.to_csv('hubspot_engagements.csv', encoding='utf-8')
object_name_in_gcs_bucket = bucket.blob('hubspot_engagements.csv') 
object_name_in_gcs_bucket.upload_from_filename('hubspot_engagements.csv')
 
bitesResult = contact_lists_df
#pandas_gbq.to_gbq(bitesResult, dataset_id +'.hubspot_contact_lists',projectid, if_exists='replace',credentials=credentials)
bitesResult.to_csv('hubspot_contact_lists.csv', encoding='utf-8')
object_name_in_gcs_bucket = bucket.blob('hubspot_contact_lists.csv') 
object_name_in_gcs_bucket.upload_from_filename('hubspot_contact_lists.csv')
 
bitesResult = email_events_df
#pandas_gbq.to_gbq(bitesResult, dataset_id +'.hubspot_email_events',projectid, if_exists='replace',credentials=credentials)
bitesResult.to_csv('hubspot_email_events.csv', encoding='utf-8')
object_name_in_gcs_bucket = bucket.blob('hubspot_email_events.csv') 
object_name_in_gcs_bucket.upload_from_filename('hubspot_email_events.csv')

bitesResult = forms_df
#pandas_gbq.to_gbq(bitesResult, dataset_id +'.hubspot_forms',projectid, if_exists='replace',credentials=credentials)
bitesResult.to_csv('hubspot_forms.csv', encoding='utf-8')
object_name_in_gcs_bucket = bucket.blob('hubspot_forms.csv') 
object_name_in_gcs_bucket.upload_from_filename('hubspot_forms.csv')

bitesResult = crm_pipelines_df
#pandas_gbq.to_gbq(bitesResult, dataset_id +'.hubspot_crm_pipelines',projectid, if_exists='replace',credentials=credentials)
bitesResult.to_csv('hubspot_crm_pipelines.csv', encoding='utf-8')
object_name_in_gcs_bucket = bucket.blob('hubspot_crm_pipelines.csv') 
object_name_in_gcs_bucket.upload_from_filename('hubspot_crm_pipelines.csv')

a= client.contacts.get_recently_created(90000)
cdf = pd.DataFrame(a)
bitesResult = cdf
#pandas_gbq.to_gbq(bitesResult, dataset_id +'.hubspot_forms',projectid, if_exists='replace',credentials=credentials)
bitesResult.to_csv('hubspot_ext.csv', encoding='utf-8')
object_name_in_gcs_bucket = bucket.blob('hubspot_ext.csv')
object_name_in_gcs_bucket.upload_from_filename('hubspot_ext.csv')

all_contacts=a
# Handle errors while calling os.remove()

try:
    os.remove('freemium_ext.csv')  
except:
    pass

with open('freemium_ext.csv','w') as result_file:
   
    result_file.write("vid,trial_plan_type,email,company"+"\n")
    for index in range  ( len(all_contacts)):
        rownum=all_contacts[index]['vid']
        contact_data = client.contacts.get_by_id(all_contacts[index]['vid'], params=params)
                 
        try:
            company=contact_data['properties']['company']['value']
        except KeyError:
                continue    
        try:
            email=contact_data['properties']['email']['value']
        except KeyError:
                continue
        try:
            trial_plan_type=contact_data['properties']['trial_plan_type']['value']
        except KeyError:
                continue
        try:
            result_file.write(str(rownum)+","+trial_plan_type+","+email+","+company+"\n")
        except KeyError:
                continue
                   
result_file.close()

bucket = gs_client.get_bucket(bucket_name)
object_name_in_gcs_bucket = bucket.blob('freemium_ext.csv')
object_name_in_gcs_bucket.upload_from_filename('freemium_ext.csv')


## Companies extra
# Handle errors while calling os.remove()

try:
    os.remove('companies_extra_ext.csv')
except:
    pass

with open('companies_extra_ext.csv','w') as result_file:
    result_file.write("id,name,website,bite_share_id,bites_username,came_from_free_trial,organization_id,organization_name"+"\n") 
 
    for index in range  ( len(all_companies)):
            rownum=all_companies[index]['id']
            comp_data = client.companies.get(all_companies[index]['id'])
                        
            try:
                company=comp_data['properties']['name']['value']
            except KeyError:
                    company='' 
            try:
                website=comp_data['properties']['website']['value']
            except KeyError:
                    website=''     
            try:
                bite_share_id=comp_data['properties']['bite_share_id']['value']
            except KeyError:
                    bite_share_id='' 
            try:
                bites_username=comp_data['properties']['bites_username']['value']
            except KeyError:
                    bites_username=''     
            try:
                came_from_free_trial=comp_data['properties']['came_from_free_trial']['value']
            except KeyError:
                    came_from_free_trial=''    
            try:
                organization_id=comp_data['properties']['organization_id']['value']
            except KeyError:
                    organization_id=''
            try:
                organization_name=comp_data['properties']['organization_name']['value'] 
            except KeyError:
                    organization_name=''
            try:
                result_file.write(str(rownum)+","+organization_name+","+organization_id+"," +  '"'+company+'"' + ","+ '"'+bite_share_id+'"' +  "," + '"'+bites_username+'"'  +"," +   '"'+came_from_free_trial+'"' +  "," +  '"'+website+'"' +"\n") 
            
            except KeyError:
                  continue
                    
result_file.close() 

bucket = gs_client.get_bucket(bucket_name) 
object_name_in_gcs_bucket = bucket.blob('companies_extra_ext.csv') 
object_name_in_gcs_bucket.upload_from_filename('companies_extra_ext.csv')
