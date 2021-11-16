from sqlalchemy import create_engine
import pandas as pd
from decouple import config
import vimeo
from google.oauth2 import service_account
import pandas_gbq as gbq
import time

# Define Environment Variables
VIMEO_CLIENT_ID = config('VIMEO_CLIENT_ID')
VIMEO_TOKEN = config('VIMEO_TOKEN')
VIMEO_CLIENT_SECRET = config('VIMEO_CLIENT_SECRET')
VIMEO_USERNAME = config('VIMEO_USERNAME')
VIMEO_PASSWORD = config('VIMEO_PASSWORD')
TYPE = config('TYPE')
PROJECT_ID = config('PROJECT_ID')
CLIENT_MAIL = config('CLIENT_MAIL')
CLIENT_ID = config('CLIENT_ID')
PRIVATE_KEY_ID = config('PRIVATE_KEY_ID')
PRIVATE_KEY = config('PRIVATE_KEY')
TOKEN_URI = config('TOKEN_URI')
AUTH_URI = config('AUTH_URI')
CLIENT_X = config('CLIENT_X')
AUTH_PROVIDER = config('AUTH_PROVIDER')
DATABASE_URL = config('DATABASE_URL')

DATASET_ID = 'bites'
print('#### LOGS #### Env VARS set complete')
cred_dic = {"type": TYPE,
            "project_id": PROJECT_ID,
            "private_key_id": PRIVATE_KEY_ID,
            "private_key": PRIVATE_KEY.encode().decode('unicode_escape'),
            "client_email": CLIENT_MAIL,
            "client_id": CLIENT_ID,
            "auth_uri": AUTH_URI,
            "token_uri": TOKEN_URI,
            "auth_provider_x509_cert_url": AUTH_PROVIDER,
            "client_x509_cert_url": CLIENT_X
            }

# Credentials
# Google creds
credentials = service_account.Credentials.from_service_account_info(cred_dic)

# Vimeo creds

client = vimeo.VimeoClient(
    token=VIMEO_TOKEN,
    key=VIMEO_CLIENT_ID,
    secret=VIMEO_CLIENT_SECRET
)

con_uri = DATABASE_URL
engine = create_engine(con_uri, pool_recycle=3600).connect()
df = pd.read_sql("select b.id bite_id ,oo.lead_name, bsf.media_url,bsf.file_type, b.description description from"
                 " bites_bitesectionfile bsf join"
                 ' bites_bitesection bsec on bsec.id=bsf.bite_section_id  join'
                 ' bites_bite b on b.id=bsec.bite_id  join'
                 ' bites_biteshare bs on bs.bite_id =b.id join'
                 ' organizations_organization oo on oo.id = b.organization_id;'
                 , engine)

# Filter only for vimeo videos - hubspot
df = df[df['file_type'] == 'video'].drop(columns=['file_type']).reset_index(drop=True)
df = df[df['lead_name'] == 'hubspot']
df = df.mask(df.eq('None')).dropna()

df = df[df['description'] != '_onboarding']
# Vimeo Duration filtering

df['media_url'] = pd.to_numeric(df['media_url'])

df_grouped_bite = df.groupby('bite_id', as_index=False)['media_url'].max()

try:
    df_exist = gbq.read_gbq(credentials=credentials,
                            query="""SELECT bite_id, max(media_url) media_url, max(bite_duration) bite_duration FROM
                                   bites.bites_duration1 group by bite_id
                                   """,
                            project_id='bites-new-api-ga4-report')



    # relavant_videos_df = df_grouped_bite[df_grouped_bite['bite_id'] not in df_exist['bite_id']]

    relavant_videos_df = df_grouped_bite[~df_grouped_bite['bite_id'].isin(list(df_exist.iloc[:, 0]))]

    df_grouped_bite = pd.merge(df_grouped_bite, df_exist, on=['bite_id','media_url'], how='left')


except Exception as e1:
    print(e1.message if hasattr(e1, 'message') else e1)
    print('No Data in DB')
    relavant_videos_df = df_grouped_bite
    df_grouped_bite['bite_duration'] = None

for v_ind, vimeo_id in enumerate(relavant_videos_df['media_url']):
    try:
        vimeo_id = (str(int(vimeo_id)))
        print('row 101:', vimeo_id)
        # Check if URL is working
        video_vimeo_url_check = rf"https://api.vimeo.com/videos/{vimeo_id}"
        if not client.get(video_vimeo_url_check).json()['is_playable']:
            print('#### VIDEO IS NOT Playable - Not Valid')
            time.sleep(6)
            continue
        else:
            df_grouped_bite['bite_duration'][v_ind] = client.get(video_vimeo_url_check).json()['duration']
            print('row 110: ', df_grouped_bite['bite_duration'][v_ind])
            tmp = df_grouped_bite['bite_duration'][v_ind]
            time.sleep(6)
    except Exception as e1:
        print(e1.message if hasattr(e1, 'message') else e1)
        print(rf'Video {vimeo_id} duration extraction failed')
        time.sleep(6)

    tmp_check = (vimeo_id, df_grouped_bite['bite_duration'][v_ind])

gbq.to_gbq(df_grouped_bite, 'bites.bites_duration1',
           project_id='bites-new-api-ga4-report',
           if_exists='replace',
           chunksize=10000,
           progress_bar=True,
           credentials=credentials,
           table_schema=[{'name': 'bite_id', 'type': 'INTEGER'},
                         {'name': 'video_vimeo_id', 'type': 'INTEGER'},
                         {'name': 'video_duration', 'type': 'INTEGER'},
                         ])
