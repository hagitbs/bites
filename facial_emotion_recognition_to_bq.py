# Video Imports
import cv2
import time
import numpy as np
from fer import FER
import pandas as pd
import os
import vimeo

# Big Query
from google.oauth2 import service_account
from pandas.io import gbq
from decouple import config
import youtube_dl
# Audio Imports


from filter_vimeo_videos import filter_videos

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

# Initiate the Facial Emotion Recognition detector object
fer_detector = FER(mtcnn=False)
print('#### LOGS #### Fer object Initiated')

face_cascade = cv2.CascadeClassifier(r'./object_detection/haarcascade_frontalface_default.xml')

print('#### logs #### - Initiate all the libraries and objects')

# Get relevant list of videos url's:
relavant_videos_df = filter_videos(url=DATABASE_URL)

print('Total filtered Videos from bites DB: ', len(relavant_videos_df))

# Filter out videos that already in the BigQuery DB
try:
    df_vimeo_exist = gbq.read_gbq(credentials=credentials,
                                  query="""SELECT video_vimeo_id FROM
                                bites.video_emotion_rec
                                group by video_vimeo_id""",
                                  project_id='bites-new-api-ga4-report')

    df_vimeo_exist['video_vimeo_id'] = df_vimeo_exist['video_vimeo_id'].astype(str)

    relavant_videos_df = relavant_videos_df[~relavant_videos_df.media_url.isin(df_vimeo_exist.video_vimeo_id)]


except Exception as e1:
    print(e1.message if hasattr(e1, 'message') else e1)
    print('No Data in DB')

print('number of relavant videos for feature extraction: ', len(relavant_videos_df))


# Iterate on all the vimeo_url's

def valid_video(v_id):
    video_vimeo_url_check = rf"https://api.vimeo.com/videos/{v_id}"
    res = client.get(video_vimeo_url_check)
    return 300 > res.status_code >= 200 and res.json()['is_playable']


for v_ind, vimeo_id in enumerate(relavant_videos_df['media_url']):
    bite_id = relavant_videos_df.iloc[v_ind, 0]
    vimeo_id = (str(vimeo_id))
    print('vimeo_id:   ', vimeo_id)
    print('bite_id:   ', bite_id)
    # Check if URL is working

    if not valid_video(vimeo_id):
        print('#### VIDEO IS NOT Playable - Not Valid ####')
        time.sleep(10)
        continue
    else:

        video_url_vimeo = rf"https://vimeo.com/" + str(vimeo_id)
        print('#### logs #### - GOT VIMEO URL')
        # Video #############################################
        print(video_url_vimeo)
        # Check if Video is Valid on Vimeo (URL validation)

        video_path = rf'./videos/{vimeo_id}.mp4'
        # Get Video From vimeo

        try:
            ydl_opts = {'format': "best",
                        'outtmpl': f'./videos/{vimeo_id}.mp4',
                        'verbose': True,
                        'username': VIMEO_USERNAME,
                        'password': VIMEO_PASSWORD,

                        # ########### Filter Vimeo minimum views count < X ##############
                        "match_filter": youtube_dl.utils.match_filter_func("duration > 15 & duration < 200 ")
                        }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                dictMeta = ydl.extract_info("https://vimeo.com/" + str(vimeo_id), download=True)
            time.sleep(3)

        except Exception as e2:
            print(e2.message if hasattr(e2, 'message') else e2)
            print('ERROR WHILE TRYING DOWNLOADING THE VIDEO')
            time.sleep(3)

        print('#### logs #### Video Have been downloaded successfuly')
        print('video path: ', video_path)

        try:
            cap = cv2.VideoCapture(video_path)

            cap.set(10, 150)
            num_of_frames = int(cap.get(7))
            # Video FPS
            video_fps = round(cap.get(cv2.CAP_PROP_FPS), 2)


            # Emotion recognition
            def emotion_cal_Frame(frame_id):
                if not fer_detector.detect_emotions(frame_id):
                    pass
                else:
                    obj_fer = fer_detector.top_emotion(frame_id)
                    if obj_fer[0] == 'happy' and obj_fer[1] > fer_detector_thresh:
                        return 1
                    else:
                        return 0


            # Initialize values:
            fer_detector_thresh = 0.7
            happy_frames_counter = 0.0
            count = 0
            sample_rate = 60
            # Object Detection

            print('#### logs #### - Starting Computer Vision algos')

            for fno in range(0, num_of_frames, sample_rate):
                try:
                    start = time.time()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)

                    _, frame = cap.read()
                    count += 1
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Emotion Rec
                    face = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
                    if face == () or fer_detector.detect_emotions(frame) == []:
                        pass
                    else:
                        happy_frames_counter += emotion_cal_Frame(frame)

                    stop = time.time()
                    duration = stop - start
                    print('complete interation')
                    print('iteration duration: ', duration)

                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break

                except Exception as e5:
                    print(e5.message if hasattr(e5, 'message') else e5)
                    print('Frame didnt loaded properly')

            cap.release()
            cv2.destroyAllWindows()
            video_avg_happiness_rate = round(happy_frames_counter / count, 2)

            print('\n', 'number of frames', count)
            print('emotion counter', video_avg_happiness_rate)


        except Exception as e50:
            video_avg_happiness_rate = 0.0
            print(e50.message if hasattr(e50, 'message') else e50)
            print('Error while trying to extract emotion recognition')

        """
        Creating Data frame
        with all the features
        """

        df_raw = pd.DataFrame(
            [[vimeo_id, bite_id, video_avg_happiness_rate]],
            columns=['video_vimeo_id', 'bite_id', 'video_avg_happiness_rate'])

        # Writing to GBQ video_audio_features Table

        gbq.to_gbq(df_raw, 'bites.video_emotion_rec',
                   project_id='bites-new-api-ga4-report',
                   if_exists='append',
                   chunksize=10000,
                   progress_bar=True,
                   credentials=credentials,
                   table_schema=[{'name': 'video_vimeo_id', 'type': 'INTEGER'},
                                 {'name': 'bite_id', 'type': 'INTEGER'},
                                 {'name': 'video_avg_happiness_rate', 'type': 'FLOAT'},
                                 ])

        if os.path.exists(video_path):
            os.remove(video_path)
        else:
            print('No Video file founded')

        time.sleep(3)
