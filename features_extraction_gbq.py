# Video Imports
import cv2
from skimage.restoration import estimate_sigma
import time
import numpy as np
# from fer import FER
from moviepy.editor import *
import moviepy.editor
import pandas as pd
from myprosody_m import myprosody as mysp
import os
import vimeo
import tensorflow_hub as hub

# Big Query
from google.oauth2 import service_account
from google.cloud import speech
from google.cloud import storage
from pandas.io import gbq
from decouple import config
import youtube_dl
# Audio Imports
import librosa
import io
import nltk
from google.cloud import vision

"""
if os.path.exists('./nltk_data/corpora/pros_cons') \
        and os.path.exists('./nltk_data/corpora/stopwords')\
        and os.path.exists('./nltk_data/corpora/wordnet')\
        and os.path.exists('./nltk_data/taggers/averaged_perceptron_tagger')\
        and os.path.exists('./nltk_data/tokenizers/punkt'):
    print('#### LOGS #### NLTK library already exists')
    pass

else:
    # NLTK imports
"""

for i in ['stopwords', 'pros_cons', 'punkt', 'averaged_perceptron_tagger', 'wordnet']:
    nltk.download(i, "./nltk_data")

nltk.data.path.append("./nltk_data")
print('#### LOGS #### NLTK library downloaded')

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

from filter_vimeo_videos import filter_videos
from ai_core.music_detection import classes_detected_audio
from test_noise_reduction import noise_reduction
from audio_noise_ratio import calc_snr

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
DOLBYIO_API_KEY = config('DOLBYIO_API_KEY')
BUCKET = config('BITES_BUCKET_NAME', default='bites_bucket-prod')
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

vimeo_client = vimeo.VimeoClient(
    token=VIMEO_TOKEN,
    key=VIMEO_CLIENT_ID,
    secret=VIMEO_CLIENT_SECRET
)

# google vision client
client_vision = vision.ImageAnnotatorClient(credentials=credentials)

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Initiate the Facial Emotion Recognition detector object
# fer_detector = FER(mtcnn=True)
print('#### LOGS #### Fer object Initiated')
# Read the coco objects data - for object detection
classNames = []
classFile = r'./object_detection/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Object Detection
configPath = r'./object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = r'./object_detection/frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

face_cascade = cv2.CascadeClassifier(r'./object_detection/haarcascade_frontalface_default.xml')

print('#### logs #### - Initiate all the libraries and objects')

# Get relevant list of videos url's:
relavant_videos_df = filter_videos(url=DATABASE_URL)

print('Total filtered Videos from bites DB: ', len(relavant_videos_df))

# Filter out videos that already in the BigQuery DB
try:
    df_vimeo_exist = gbq.read_gbq(credentials=credentials,
                                  query="""SELECT video_vimeo_id FROM
                                bites.video_audio_features_new
                                group by video_vimeo_id""",
                                  project_id='bites-new-api-ga4-report')

    df_speech_exist = gbq.read_gbq(credentials=credentials,
                                   query="""SELECT video_vimeo_id, max(audio_transcription) audio_transcription FROM
                                    bites.video_audio_features_exist
                                    group by video_vimeo_id""",
                                   project_id='bites-new-api-ga4-report')

    exist_transcription_vimeo_id_lst = list(map(str, df_speech_exist.video_vimeo_id))
    df_vimeo_exist['video_vimeo_id'] = df_vimeo_exist['video_vimeo_id'].astype(str)

    relavant_videos_df = relavant_videos_df[~relavant_videos_df.media_url.isin(df_vimeo_exist.video_vimeo_id)][::-1]


except Exception as e1:
    print(e1.message if hasattr(e1, 'message') else e1)
    print('No Data in DB')

print('number of relavant videos for feature extraction: ', len(relavant_videos_df))


# Iterate on all the vimeo_url's

def valid_video(v_id):
    video_vimeo_url_check = f"https://api.vimeo.com/videos/{v_id}"
    res = vimeo_client.get(video_vimeo_url_check)
    return 300 > res.status_code >= 200 and res.json()['is_playable']


for v_ind, vimeo_id in enumerate(relavant_videos_df['media_url']):
    bite_id = relavant_videos_df.iloc[v_ind, 1]
    vimeo_id = (str(vimeo_id))
    print('vimeo_id:   ', vimeo_id)
    print('bite_id:   ', bite_id)
    # Check if URL is working

    if not valid_video(vimeo_id):
        print('#### VIDEO IS NOT Playable - Not Valid ####')
        time.sleep(10)

    else:

        print('#### logs #### - GOT VIMEO URL')
        # Video #############################################
        print(f"https://vimeo.com/{vimeo_id}")
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
            time.sleep(10)

        except Exception as e2:
            print(e2.message if hasattr(e2, 'message') else e2)
            print('ERROR WHILE TRYING DOWNLOADING THE VIDEO')
            time.sleep(10)

        print('#### logs #### Video Have been downloaded successfuly')
        print('video path: ', video_path)

        # Check if there is an audio in the mp4

        if os.path.exists(video_path):
            print('### logs ### - video found in directory /videos')
            video = moviepy.editor.VideoFileClip(video_path)
            is_audio = video.audio

            if is_audio is None:
                print('AUDIO NOT FOUND')
                audio_path = ""
            else:
                audioclip = AudioFileClip(video_path)
                audio_path = f"./myprosody_m/myprosody/dataset/audioFiles/{vimeo_id}.wav"
                audioclip.write_audiofile(audio_path)

                print('#### logs #### - Extract Audio')

            # Computer vision

            try:
                cap = cv2.VideoCapture(video_path)

                cap.set(10, 150)

                # Extract all static parameters - that wont change from frame to frame
                width = cap.get(3)  # width
                height = cap.get(4)  # height
                num_of_frames = int(cap.get(7))

                # Video Resolution - Width x Height
                video_resolution = [int(cap.get(3)), int(cap.get(4))]


                # Video orientation

                def vid_orient(video_resolution):
                    if (video_resolution[0] / video_resolution[1]) >= 1:
                        orient = 'Horizontal'
                    else:
                        orient = 'Vertical'

                    return orient


                video_orientation = vid_orient(video_resolution)

                # Video FPS
                video_fps = round(cap.get(cv2.CAP_PROP_FPS), 2)


                # Video duration

                def vid_dur(durationInSeconds):
                    fps = cap.get(5)
                    durationInSeconds = round(float(num_of_frames) / float(fps), 2)

                    return durationInSeconds


                video_duration = vid_dur(cap)


                # Face detector
                def detect_face_dim(gray_frame):
                    global face_dim
                    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
                    for (x, y, w, h) in faces:
                        face_dim = w * h
                    return float(face_dim)


                """
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

                """


                # Extract all dynamic parameters - changing from frame to frame

                # plot func - testing
                def plot_img(img):
                    cv2.imshow('random', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


                # Initialize values:
                fer_detector_thresh = 0.7
                estimate_noise_tot = 0.0
                contrast = 0.0
                brightness = 0.0
                saturation = 0.0
                face_dim = 0.0
                Face_Screen_Rate = 0.0
                happy_frames_counter = 0.0
                count = 0
                sample_rate = 120
                # Object Detection
                thres = 0.60  # Threshold to detect object
                nms_threshold = 0.2
                object_lis = []
                obj = []
                ocr_lis = []

                print('#### logs #### - Starting Computer Vision algos')

                for fno in range(0, num_of_frames, sample_rate):
                    try:
                        start = time.time()
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)

                        _, frame = cap.read()
                        count += 1
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
                        # Calculate video average noise
                        estimate_noise_tot += estimate_sigma(frame, multichannel=True, average_sigmas=True)
                        # Calculate Contrast brightness and colorfulness
                        contrast += frame_gray.std()
                        brightness += frame_hsv[:, :, 2].mean()
                        saturation += frame_hsv[:, :, 1].mean()
                        # Face Screen Rate
                        frame_dim = float(frame.shape[0] * frame.shape[1])
                        face_dim = detect_face_dim(frame_gray)
                        if face_dim is not None:
                            Face_Screen_Rate += (face_dim / frame_dim) * 450

                        """
                        # Emotion Rec
                        face = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
                        if face == () or fer_detector.detect_emotions(frame) == []:
                            pass
                        else:
                            happy_frames_counter += emotion_cal_Frame(frame)
                        """
                        # Object Detection
                        classIds, confs, bbox = net.detect(frame, confThreshold=thres)
                        bbox = list(bbox)
                        confs = list(np.array(confs).reshape(1, -1)[0])
                        confs = list(map(float, confs))
                        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
                        for i in indices:
                            i = i[0]
                            box = bbox[i]
                            x, y, w, h = box[0], box[1], box[2], box[3]
                            obj = classNames[classIds[i][0] - 1]
                            if obj not in object_lis:
                                object_lis.append(obj)

                        # ocr - Image Text Detection
                        try:

                            image = vision.Image(content=cv2.imencode('.png', frame)[1].tobytes())
                            response = client_vision.text_detection(image=image)
                            text_lis = response.text_annotations[0].description.split()

                            # Append word detected to list
                            for word in text_lis:
                                if word not in ocr_lis:
                                    ocr_lis.append(word)

                        except Exception as e25:
                            print(e25.message if hasattr(e25, 'message') else e25)
                            print('Didnt detected text in Image')

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

                if ocr_lis is not None:
                    # OCR - NLP clean
                    ocr_lis_words_witout_punc = [word for word in ocr_lis if word.isalpha()]
                    ocr_lis_punc = set([word for word in ocr_lis_words_witout_punc if not word in stopwords.words()])
                    ocr_lis_eng = [word for word in ocr_lis_punc if wordnet.synsets(word)]
                    ocr_lis = [word for word in ocr_lis_eng if len(word) > 2]
                    print('ocr - text detected :', ocr_lis)

                # video average noise - the higher the value mean low noise.
                video_avg_noise = round(estimate_noise_tot / count, 2)

                # Avg contrast of video
                video_avg_contrast = round(contrast / count, 2)

                # Avg brightness of video
                video_avg_brightness = round(brightness / count, 2)

                # Avg saturation of video
                video_avg_saturation = round(saturation / count, 2)

                video_avg_face_screen_rate = round(Face_Screen_Rate / count, 2)

                # video_avg_happiness_rate = round(happy_frames_counter / count, 2)

                print('\n', 'number of frames', count)
                print('object list', object_lis)
                # print('emotion counter', video_avg_happiness_rate)
                print('screen_face_rate', video_avg_face_screen_rate)
                print('saturation', video_avg_saturation)
                print('brightness', video_avg_brightness)
                print('contrast', video_avg_contrast)
                print('noise', video_avg_noise)
                print('dur', video_duration)
                print('fps', video_fps)
                print('res', video_resolution)
                print('orient', video_orientation)
                print('ocr text found', ocr_lis)

            except Exception as e6:
                print(e6.message if hasattr(e6, 'message') else e6)
                print('Computer vision error')

            ##############################################################################################################
            ##############################################################################################################
            if is_audio is None:
                snr = num_syl = num_pauses_filler = speech_rate = articulation_rate = speech_to_non_speech_rate = signal_features_dic = 0.0
                tot_transcript = ''
                transcription_resp_str = None
                video_tags_list = []

                print('#### logs #### - No Audio founded')
            else:
                ### Apply background music detection
                try:

                    audio_class_detect = classes_detected_audio(yamnet_model, audio_path)
                    print(audio_class_detect)

                except Exception as e26:
                    audio_class_detect = []
                    print(e26.message if hasattr(e26, 'message') else e26)
                    print('Error in the audio class detection - YAMNET')

                # Check if there is a background music
                if 'Music' in audio_class_detect:
                    snr = 0
                    file_to_upload_name = f'./myprosody_m/myprosody/dataset/audioFiles/{vimeo_id}.wav'
                else:

                    # Calculating SNR
                    try:

                        snr = calc_snr(audio_path)

                    except Exception as e27:
                        print(e27.message if hasattr(e27, 'message') else e27)
                        print('Error in the audio Signal To Noise Ratio Calculation')

                    # Apply Noise reduction using DOLBY.IO

                    try:

                        noise_reduction(DOLBYIO_API_KEY, vimeo_id)
                        file_to_upload_name = f'./myprosody_m/myprosody/dataset/audioFiles/enhanced/{vimeo_id}_enhanced.wav'


                    except Exception as e28:
                        file_to_upload_name = f'./myprosody_m/myprosody/dataset/audioFiles/{vimeo_id}.wav'
                        print(e28.message if hasattr(e28, 'message') else e28)
                        print('Error in Noise reduction using DOLBY_IO service')

                    # Try to upload enhanced audio file to Google Bucket

                    try:
                        # Upload audio to google bucket
                        file_name = vimeo_id + '_enhanced.wav'

                        gcs = storage.Client(credentials=credentials)
                        bucket = gcs.get_bucket(BUCKET)

                        # upload blob
                        blob = bucket.blob('audio/enhanced/' + file_name)

                        storage.blob._DEFAULT_CHUNKSIZE = 2097152  # 1024 * 1024 B * 2 = 2 MB
                        storage.blob._MAX_MULTIPART_SIZE = 2097152  # 2 MB
                        blob.upload_from_filename(file_to_upload_name)

                    except Exception as e59:
                        print(e59.message if hasattr(e59, 'message') else e59)
                        print('Could not upload enhanced audio to Google Cloud Bucket')

                if vimeo_id not in exist_transcription_vimeo_id_lst:
                    transcript_exists = False

                    ### Upload audio to google bucket
                    file_name = vimeo_id + '.wav'

                    gcs = storage.Client(credentials=credentials)
                    bucket = gcs.get_bucket(BUCKET)

                    # upload blob
                    blob = bucket.blob('audio/' + file_name)

                    storage.blob._DEFAULT_CHUNKSIZE = 2097152  # 1024 * 1024 B * 2 = 2 MB
                    storage.blob._MAX_MULTIPART_SIZE = 2097152  # 2 MB
                    blob.upload_from_filename(file_to_upload_name)

                    # read a blob
                    blob = bucket.blob(f'audio/' + file_name)

                    file_as_string = blob.download_as_string()
                    # convert the string to bytes and then finally to audio samples as floats
                    # and the audio sample rate

                    y, sr = librosa.load(io.BytesIO(file_as_string))

                    # Audio Duration
                    audio_duration = librosa.get_duration(y=y, sr=sr)

                    # Google Speech ###########

                    file_uri = rf'gs://{BUCKET}/audio/{file_name}'


                    def transcribe_gcs(gcs_uri, sr):
                        """Asynchronously transcribes the audio file specified by the gcs_uri."""

                        client = speech.SpeechClient(credentials=credentials)

                        audio = speech.RecognitionAudio(uri=gcs_uri)

                        # speech_context = speech.SpeechContext(phrases=['equashield', 'cstd', 'hazardous', 'drugs'])

                        config = speech.RecognitionConfig(
                            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                            sample_rate_hertz=sr * 2,
                            audio_channel_count=2,
                            language_code="en-US",
                            enable_word_time_offsets=True,
                            model='video',
                            enable_automatic_punctuation=True,
                            # speech_contexts=[speech_context]

                        )

                        operation = client.long_running_recognize(config=config, audio=audio)

                        print("Waiting for operation to complete...")

                        response = operation.result(timeout=audio_duration + 60)

                        # Each result is for a consecutive portion of the audio. Iterate through
                        # them to get the transcripts for the entire audio file.
                        tot_transcript = ''
                        for result in response.results:
                            tot_transcript += result.alternatives[0].transcript
                        """for i, result in enumerate(response.results):
                            alternative = result.alternatives[0]
                            print("-" * 20)
                            print("First alternative of result {}".format(i))
                            print(u"Transcript: {}".format(alternative.transcript))
                            print(u"Channel Tag: {}".format(result.channel_tag))
                            print("Confidence: {}".format(result.alternatives[0].confidence))"""

                        return response, tot_transcript


                    try:
                        resp, tot_transcript = transcribe_gcs(file_uri, sr=sr)
                    except Exception as e3:
                        print(e3.message if hasattr(e3, 'message') else e3)
                        print('No audio founded - Connection timed out for Google Speech')

                    print(tot_transcript)

                    transcription_resp_str = str(resp)

                    # Part of speech Tagging - NLP


                else:
                    print('Speech Exists in BQ already')
                    transcript_exists = True
                    tot_transcript = \
                    df_speech_exist[df_speech_exist['video_vimeo_id'] == int(vimeo_id)]['audio_transcription'].values[0]
                    transcription_resp_str = None
                    print('Existing transcription:\n ', tot_transcript)


                # Create tag list from transcription

                def pos_tagging(text):
                    text_tokens = word_tokenize(text)
                    tokens_low = [w.lower() for w in text_tokens]
                    words_witout_punc = [word for word in tokens_low if word.isalpha()]
                    tokens_without_sw = [word for word in words_witout_punc if not word in stopwords.words()]
                    sentence_cleaned = ' '.join(tokens_without_sw)
                    sen_noun_verbs_tags = [word for (word, pos) in
                                           nltk.pos_tag(nltk.word_tokenize(sentence_cleaned))]
                    is_noun_verb = lambda pos: pos[:2] == 'NN'
                    nouns_verb_list = [word for (word, pos) in nltk.pos_tag(sen_noun_verbs_tags) if
                                       is_noun_verb(pos)]
                    return list(set(nouns_verb_list))


                if tot_transcript is not None:
                    video_tags_list = pos_tagging(tot_transcript)
                    print(video_tags_list)
                else:
                    video_tags_list = []

                ################################
                # Extract Signal features
                signal_features_dic = {}


                def extract_audio_features(y, sr, d):
                    d['chroma_stft'] = librosa.feature.chroma_stft(y=y, sr=sr).mean()
                    d['rms'] = librosa.feature.rms(y=y).mean()
                    d['spec_cent'] = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
                    d['spec_bw'] = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
                    d['rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
                    d['zcr'] = librosa.feature.zero_crossing_rate(y).mean()
                    d['tonnetz'] = librosa.feature.tonnetz(y).mean()
                    d['melspectrogram'] = librosa.feature.melspectrogram(y).mean()
                    d['spectral_centroid'] = librosa.feature.spectral_centroid(y).mean()
                    d['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y).mean()
                    d['spectral_contrast'] = librosa.feature.spectral_contrast(y).mean()
                    d['spectral_flatness'] = librosa.feature.spectral_flatness(y).mean()
                    d['mfcc'] = librosa.feature.mfcc(y=y, sr=sr)

                    return d


                y, sr = librosa.load(audio_path, sr=44100)
                signal_features_dic = (extract_audio_features(y, sr, signal_features_dic))

                # Create dictionary for the mfcc
                mfcc_dic = {}
                for i, val in enumerate(signal_features_dic['mfcc']):
                    mfcc_dic[f'mfcc_{i}'] = val.mean()

                # Add the mfcc_dictionary keys and values into the features_dictionary
                signal_features_dic.pop('mfcc', None)

                for k, val in mfcc_dic.items():
                    signal_features_dic[k] = val

                # speaking features ######################
                if tot_transcript != '':
                    p = vimeo_id
                    c = rf'{os.getcwd()}/myprosody_m/myprosody'

                    # num_syl - Detect and count number of syllables
                    num_syl = mysp.myspsyl(p, c)

                    # num_pauses_filler - Detect and count number of fillers and pauses
                    num_pauses_filler = mysp.mysppaus(p, c)

                    # Measure the rate of speech (speed)
                    speech_rate = mysp.myspsr(p, c)

                    # articulation -Measure the articulation (speed)
                    articulation_rate = mysp.myspatc(p, c)

                    # speech_to_non_speech_rate - Measure ratio between speaking duration and total speaking duration
                    speech_to_non_speech_rate = mysp.myspbala(p, c)
                else:
                    num_syl = num_pauses_filler = speech_rate = articulation_rate = speech_to_non_speech_rate = 0.0

            """
            Creating Data frame
            with all the features
            """

            df_raw = pd.DataFrame(
                [[vimeo_id, bite_id, video_orientation, video_resolution, video_duration, video_avg_noise,
                  video_avg_contrast,
                  video_avg_brightness, video_avg_saturation, video_avg_face_screen_rate, ocr_lis,
                  object_lis, snr, num_syl, num_pauses_filler, speech_rate,
                  articulation_rate, speech_to_non_speech_rate, video_tags_list, tot_transcript,
                  transcription_resp_str]],

                columns=['video_vimeo_id', 'bite_id', 'video_orientation', 'video_resolution', 'video_duration',
                         'video_avg_noise',
                         'video_avg_contrast', 'video_avg_brightness', 'video_avg_saturation',
                         'video_avg_face_screen_rate', 'image_text_detected',
                         'object_lis', 'audio_signal_to_noise_ratio', 'num_syllables',
                         'num_pauses_filler', 'speech_rate', 'articulation_rate',
                         'speech_to_non_speech_rate', 'video_tags_list', 'audio_transcription',
                         'transcription_full_resp'])

            # Writing to GBQ video_audio_features Table

            gbq.to_gbq(df_raw, 'bites.video_audio_features_new',
                       project_id='bites-new-api-ga4-report',
                       if_exists='append',
                       chunksize=10000,
                       progress_bar=True,
                       credentials=credentials,
                       table_schema=[{'name': 'video_vimeo_id', 'type': 'INTEGER'},
                                     {'name': 'bite_id', 'type': 'INTEGER'},
                                     {'name': 'video_orientation', 'type': 'STRING'},
                                     {'name': 'video_resolution', 'type': 'STRING'},
                                     {'name': 'video_duration', 'type': 'FLOAT'},
                                     {'name': 'video_avg_noise', 'type': 'FLOAT'},
                                     {'name': 'video_avg_contrast', 'type': 'FLOAT'},
                                     {'name': 'video_avg_brightness', 'type': 'FLOAT'},
                                     {'name': 'video_avg_saturation', 'type': 'FLOAT'},
                                     {'name': 'video_avg_face_screen_rate', 'type': 'FLOAT'},
                                     {'name': 'image_text_detected', 'type': 'STRING'},
                                     {'name': 'object_lis', 'type': 'STRING'},
                                     {'name': 'audio_signal_to_noise_ratio', 'type': 'FLOAT'},
                                     {'name': 'num_syllables', 'type': 'FLOAT'},
                                     {'name': 'num_pauses_filler', 'type': 'FLOAT'},
                                     {'name': 'speech_rate', 'type': 'FLOAT'},
                                     {'name': 'articulation_rate', 'type': 'FLOAT'},
                                     {'name': 'speech_to_non_speech_rate', 'type': 'FLOAT'},
                                     {'name': 'video_tags_list', 'type': 'STRING'},
                                     {'name': 'audio_transcription', 'type': 'STRING'},
                                     {'name': 'transcription_full_resp', 'type': 'STRING'},
                                     ])
            try:
                video.close()
            except Exception as e7:
                print(e7.message if hasattr(e7, 'message') else e7)
                print('No Video to close')

            try:
                audioclip.close()
            except Exception as e8:
                print(e8.message if hasattr(e8, 'message') else e8)
                print('No Audio to close')

            text_grid_path = rf"./myprosody_m/myprosody/dataset/audioFiles/{vimeo_id}.TextGrid"
            enhanced_audio_path = f'./myprosody_m/myprosody/dataset/audioFiles/enhanced/{vimeo_id}_enhanced.wav'

            if os.path.exists(text_grid_path):
                os.remove(text_grid_path)
            else:
                print('No audio - No textGrid file founded')

            if os.path.exists(video_path):
                os.remove(video_path)
            else:
                print('No Video file founded')

            if os.path.exists(audio_path):
                os.remove(audio_path)
            else:
                print('No Audio file founded')

            if os.path.exists(enhanced_audio_path):
                os.remove(enhanced_audio_path)
            else:
                print('No Enhanced Audio file founded')

            print('#### logs ### Video and Audio - LOCAL FILES - REMOVED')
        else:
            time.sleep(10)
            print('#### logs #### VIDEO not found in directory')
