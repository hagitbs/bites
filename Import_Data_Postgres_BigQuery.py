#more Import_Data_Postgres_BigQuery.py
import pandas as pd
from sqlalchemy import create_engine
from google.oauth2 import service_account
from decouple import config
from pandas.io import gbq


# Define Environment Variables
TYPE = config('TYPE')
DATABASE_URL = config('DATABASE_URL')
PROJECT_ID = config('PROJECT_ID')
CLIENT_MAIL = config('CLIENT_MAIL')
CLIENT_ID = config('CLIENT_ID')
PRIVATE_KEY_ID = config('PRIVATE_KEY_ID')
PRIVATE_KEY = config('PRIVATE_KEY')
TOKEN_URI = config('TOKEN_URI')
AUTH_URI = config('AUTH_URI')
CLIENT_X = config('CLIENT_X')
AUTH_PROVIDER = config('AUTH_PROVIDER')



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
credentials = service_account.Credentials.from_service_account_info(cred_dic)

# BigQuery credentials
DATASET_ID = 'bites'
# Path for the dump directory
DIRECTORY = 'dump'


def main():
    # Let's begin!
    print("Start Import")

    con_uri = DATABASE_URL

    try:
        engine = create_engine(con_uri, pool_recycle=3600).connect()
    except Exception as e:
        print("Error {}".format(e))

    tables_query = "SELECT table_name FROM information_schema.tables WHERE TABLE_SCHEMA = 'public' and TABLE_NAME in" \
                   "('users_user','bites_playlist','bites_biteshare'," \
                   "'bites_biteshareuser','bites_comment','bites_choice'," \
                   "'bites_bite','bites_bitesection'," \
                   "'bites_question','bites_content'," \
                   "'bites_userchoice','bites_summary','organizations_organization','users_userorganization','bites_groupshare','bites_bitesectionfile');"

    list_tables = pd.read_sql(tables_query, con_uri)

    # This print is only for information
    print(list_tables)


    # Iterate over table list, get data and upload to BigQuery via pandas_gbq
    for index, row in list_tables.iterrows():
        table_id = '{}.{}'.format(DATASET_ID, row['table_name'])

        print("Loading Table {}".format(table_id))
        df = pd.read_sql_table(row['table_name'], engine)
        from pandas_gbq import schema
        schema.generate_bq_schema(df)
        gbq.to_gbq(df, table_id,
                   project_id=PROJECT_ID,
                   if_exists='replace',
                   chunksize=1000000,
                   progress_bar=True,
                   credentials=credentials)


if __name__ == '__main__':
    main()
