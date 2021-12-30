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
all_owners = client.owners.get_owners()
owners_df=pd.DataFrame(all_owners)
bitesResult = owners_df
bitesResult.to_csv('hubspot_owners.csv', encoding='utf-8')

bucket = gs_client.get_bucket(bucket_name) 
object_name_in_gcs_bucket = bucket.blob('hubspot_owners.csv') 
object_name_in_gcs_bucket.upload_from_filename('hubspot_owners.csv')
##
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
   
    result_file.write("vid,company,email,trial_plan_type,company_id,company_name,portal_id,bites_username,company_came_from_free_trial,company_createdate,company_first_contact_createdate,
organization_id,organization_name,u_trial_status,u_trial_start_date,u_trial_plan_type,u_trial_days,u_qualification_steps,u_pql___sql,u_phone,u_original_owner,u_original_marketing_score,u_
organization_id,u_lastname,u_hubspot_owner_id,u_hubspot_owner_assigneddate,u_free_trial_contact,u_firstname,u_email,u_currentlyinworkflow,u_createdate,u_create_trial,u_contact_score,u_com
pany,u_associatedcompanyid,u_access_code,hs_analytics_source_data_1,hs_analytics_source_data_2,hs_analytics_source,ga_pseudo_user_id"+"\n")
    for index in range  ( len(all_contacts)):
        rownum=all_contacts[index]['vid']
        contact_data = client.contacts.get_by_id(all_contacts[index]['vid'], params=params)
       
            
        try:
            company=contact_data['properties']['company']['value']
        except KeyError:
                company=''    
        try:
            email=contact_data['properties']['email']['value']
        except KeyError:
                email=''
        try:
            trial_plan_type=contact_data['properties']['trial_plan_type']['value']
        except KeyError:
                trial_plan_type=''
        try:
            company_id=(contact_data['associated-company']['company-id'])
        except KeyError:
                company_id=''
        try:
            company_name=(contact_data['associated-company']['properties']['name']['value']) 
        except KeyError:
                company_name=''                      
        try:
            portal_id=(contact_data['associated-company']['portal-id'])
        except KeyError:
                portal_id=''
        
        try:
            bites_username=(contact_data['associated-company']['properties']['bites_username']['value'])
        except KeyError:
                bites_username=''
        try:
            company_came_from_free_trial=(contact_data['associated-company']['properties']['came_from_free_trial']['value'])
        except KeyError:
                company_came_from_free_trial=''
        try:
            company_createdate=(date.fromtimestamp( int( contact_data['associated-company']['properties']['createdate']['value']) /1000))    
        
        except KeyError:
                company_createdate=''
        try:
            company_first_contact_createdate=(date.fromtimestamp( int( contact_data['associated-company']['properties']['first_contact_createdate']['value']) /1000))      
        except KeyError:
                company_first_contact_createdate=''
        try:
            organization_id=(contact_data['associated-company']['properties']['organization_id']['value'])
        except KeyError:
                organization_id=''
        try:
            organization_name=(contact_data['associated-company']['properties']['organization_name']['value']) 
        except KeyError:
                organization_name=''    
        
        
        try: 
            u_access_code=(contact_data['properties']['access_code']['value']) 
        except KeyError:
            u_access_code=''
        try:
            u_associatedcompanyid=(contact_data['properties']['associatedcompanyid']['value']) 
        except KeyError:
             u_associatedcompanyid=''
        try:
            u_company=(contact_data['properties']['company']['value']) 
        except KeyError:
            u_company=''  
        try:
            u_contact_score=(contact_data['properties']['contact_score']['value']) 
        except KeyError:
            u_contact_score=''
        try:
            u_create_trial=(contact_data['properties']['create_trial']['value']) 
        except KeyError:
            u_create_trial='' 
        try:
            u_createdate=date.fromtimestamp( int( contact_data['properties']['createdate']['value']) /1000 )    
        except KeyError:
            u_createdate=''
        try:
            u_currentlyinworkflow=(contact_data['properties']['currentlyinworkflow']['value']) 
        except KeyError:
            u_currentlyinworkflow='' 
        try:
            u_email=(contact_data['properties']['email']['value']) 
        except KeyError:
            u_email=''  
        try:
            u_firstname=(contact_data['properties']['firstname']['value']) 
        except KeyError:
            u_firstname=''
        try:
            u_lastname=(contact_data['properties']['lastname']['value']) 
        except KeyError:
            u_lastname==''             
        try:
            u_free_trial_contact=(contact_data['properties']['free_trial_contact']['value']) 
        except KeyError:
            u_free_trial_contact='' 
        try:
            u_organization_id=(contact_data['properties']['organization_id']['value']) 
        except KeyError:
            u_organization_id=''  
        try:
            u_original_marketing_score=(contact_data['properties']['original_marketing_score']['value']) 
        except KeyError:
            u_original_marketing_score=''
        try:
            u_original_owner=(contact_data['properties']['original_owner']['value']) 
        except KeyError:
            u_original_owner=''
        try:
            u_phone=(contact_data['properties']['phone']['value']) 
        except KeyError:
            u_phone='' 
        try:
            u_pql___sql=(contact_data['properties']['pql___sql']['value']) 
        except KeyError:
            u_pql___sql=''  
        try:
            u_qualification_steps=(contact_data['properties']['qualification_steps']['value']) 
        except KeyError:
            u_qualification_steps=''  
        try:
            u_trial_days=(contact_data['properties']['trial_days']['value']) 
        except KeyError:
            u_trial_days=''  
        try:
            u_trial_plan_type=(contact_data['properties']['trial_plan_type']['value']) 
        except KeyError:
            u_trial_plan_type=''  
        try:
            u_trial_start_date=(date.fromtimestamp( int( contact_data['properties']['trial_start_date']['value']) /1000))    
        except KeyError:
            u_trial_start_date=''  
        try:
            u_trial_status=(contact_data['properties']['trial_status']['value'])     
        except KeyError:
            u_trial_status=''  
        try:
            u_hubspot_owner_assigneddate=date.fromtimestamp( int(contact_data['properties']['hubspot_owner_assigneddate']['value'])/1000)      
        except KeyError:
            u_hubspot_owner_assigneddate='' 
        try:
            u_hubspot_owner_id=(contact_data['properties']['hubspot_owner_id']['value'])  
        except KeyError:
            u_hubspot_owner_id=''
        try:
            hs_analytics_source_data_1=(contact_data['properties']['hs_analytics_source_data_1']['value'])
        except KeyError:
            hs_analytics_source_data_1=''
        try:
            hs_analytics_source_data_2=(contact_data['properties']['hs_analytics_source_data_2']['value'])
        except KeyError:
            hs_analytics_source_data_2=''
        try:
            hs_analytics_source=(contact_data['properties']['hs_analytics_source']['value'])
        except KeyError:
            hs_analytics_source=''
        try:
            ga_pseudo_user_id=(contact_data['properties']['ga_pseudo_user_id']['value'])
        except KeyError:
            ga_pseudo_user_id=''

        #onboarding_conversation__freemium_    
                    
        
        try:
            result_file.write(str(rownum)+
            ","+"\""+company+"\""+
            "," +email+
            ","+trial_plan_type+
            ","+ str(company_id) + 
            ", " +"\""+company_name +"\""+
            ","+str(portal_id)+
            ","+bites_username+
            ","+company_came_from_free_trial+
            ","+str(company_createdate)+
            ","+str(company_first_contact_createdate) +
            ","+str(organization_id) +
            ","+"\""+organization_name+"\""+
            ","+	u_trial_status	+
            ","+str(	u_trial_start_date)	+
            ","+	u_trial_plan_type+
            ","+str(u_trial_days)+
            ","+	u_qualification_steps+
            ","+	u_pql___sql	+
            ","+str(u_phone)+
            ","+	u_original_owner+
            ","+	u_original_marketing_score+
            ","+	u_organization_id+
            ","+	u_lastname+ 
            ","+	u_hubspot_owner_id+
            ","+str(u_hubspot_owner_assigneddate)+
            ","+	u_free_trial_contact+
            ","+	u_firstname+
            ","+	u_email+
            ","+	u_currentlyinworkflow+
            ","+str(	u_createdate)+
            ","+	u_create_trial+
            ","+	u_contact_score+
            ","+"\""+	u_company+"\""+
            ","+	u_associatedcompanyid+	
            ","+        u_access_code+
            ","+        hs_analytics_source_data_1+
            ","+        hs_analytics_source_data_2+
            ","+        hs_analytics_source+
            ","+        ga_pseudo_user_id+
            ","+                    "\n")
        except KeyError:
                continue

result_file.close()
"""
        try:
            result_file.write(str(rownum)+","+trial_plan_type+","+email+","+company+"\n")
        except KeyError:
                continue
                   
result_file.close()
"""
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
    result_file.write("id,organization_name,organization_id,company,bite_share_id,bites_username,came_from_free_trial,website"+"\n") 
 
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
                result_file.write(str(rownum)+ "," + "\""+organization_name+"\"" + ","+organization_id+"," +   "\""+company+"\""+  ","+ '"'+bite_share_id+'"' +  "," + '"'+bites_username+'
"'  +"," +   "\""+came_from_free_trial+"\""+  "," + "\""+website+"\"" +"\n")

                #result_file.write(str(rownum)+","+organization_name+","+organization_id+"," +  '"'+company+'"' + ","+ '"'+bite_share_id+'"' +  "," + '"'+bites_username+'"'  +"," +   '"'+
came_from_free_trial+'"' +  "," +  '"'+website+'"' +"\n") 
            
            except KeyError:
                  continue
                    
result_file.close() 

bucket = gs_client.get_bucket(bucket_name) 
object_name_in_gcs_bucket = bucket.blob('companies_extra_ext.csv') 
object_name_in_gcs_bucket.upload_from_filename('companies_extra_ext.csv')
