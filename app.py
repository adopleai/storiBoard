
import streamlit as st
import whisper
import re
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
import math
from stable_whisper import modify_model,results_to_word_srt, results_to_sentence_srt
import asyncio
from deepgram import Deepgram
from typing import Dict
import os
import moviepy.editor as mp
from pytube import YouTube
from time import sleep
import pandas as pd
from difflib import *
import multiprocessing



st.set_page_config(layout="wide")

st.title('StoriBoard AI Video Editor')


@st.cache_resource
#load whisper model
def load_model(model_selected):
  #load medium model
  model = whisper.load_model(model_selected)
  # modify model to get word timestamp
  modify_model(model)
  return model

#transcrib
@st.cache_resource 
def transcribe_video(vid,model_selected):
    model = load_model(model_selected)
    options = whisper.DecodingOptions(fp16=False)
    result = model.transcribe(vid, **options.__dict__)
    result['srt'] = whisper_result_to_srt(result)
    return result

#srt generation
def whisper_result_to_srt(result):
    text = []
    for i,s in enumerate(result['segments']):
        text.append(str(i+1))
        time_start = s['start']
        hours, minutes, seconds = int(time_start/3600), (time_start/60) % 60, (time_start) % 60
        timestamp_start = "%02d:%02d:%06.3f" % (hours, minutes, seconds)
        timestamp_start = timestamp_start.replace('.',',')     
        time_end = s['end']
        hours, minutes, seconds = int(time_end/3600), (time_end/60) % 60, (time_end) % 60
        timestamp_end = "%02d:%02d:%06.3f" % (hours, minutes, seconds)
        timestamp_end = timestamp_end.replace('.',',')        
        text.append(timestamp_start + " --> " + timestamp_end)
        text.append(s['text'].strip() + "\n")
    return "\n".join(text)

#compute speaking_time
async def compute_speaking_time(transcript_data: Dict,data:str) -> None:
   if 'results' in transcript_data:
       transcript = transcript_data['results']['channels'][0]['alternatives'][0]['words']
       total_speaker_time = {}
       speaker_words = []
       current_speaker = -1

       for speaker in transcript:
           speaker_number = speaker["speaker"]

           if speaker_number is not current_speaker:
               current_speaker = speaker_number
               speaker_words.append([speaker_number, [], 0])

               try:
                   total_speaker_time[speaker_number][1] += 1
               except KeyError:
                   total_speaker_time[speaker_number] = [0,1]

           get_word = speaker["word"]
           speaker_words[-1][1].append(get_word)

           total_speaker_time[speaker_number][0] += speaker["end"] - speaker["start"]
           speaker_words[-1][2] += speaker["end"] - speaker["start"]

       for speaker, words, time_amount in speaker_words:
           print(f"Speaker {speaker}: {' '.join(words)}")
           data+=f"\nSpeaker {speaker}: {' '.join(words)}"
           print(f"Speaker {speaker}: {time_amount}")
           data+=f"\nSpeaker {speaker}: {time_amount}"


       for speaker, (total_time, amount) in total_speaker_time.items():
           print(f"Speaker {speaker} avg time per phrase: {total_time/amount} ")
           data+=f"\nSpeaker {speaker} avg time per phrase: {total_time/amount} "
           print(f"Total time of conversation: {total_time}")
           data+=f"\nTotal time of conversation: {total_time}"
   return transcript,data

#extract audio from video
def extract_write_audio(vd):
  my_clip = mp.VideoFileClip(f'{vd}')
  my_clip.audio.write_audiofile(f"audio.wav")

def get_subtitle_by_sequence_number(srt_file_path, sequence_number):
    with open(srt_file_path, 'r') as srt_file:
        srt_content = srt_file.read()

    subtitles = srt_content.strip().split('\n\n')
    for subtitle in subtitles:
        subtitle_lines = subtitle.split('\n')


        subtitle_sequence_number = int(subtitle_lines[0])
        if subtitle_sequence_number == sequence_number:
            subtitle_timestamp = subtitle_lines[1]
            subtitle_text = ' '.join(subtitle_lines[2:])
            return subtitle_timestamp, subtitle_text

    return None

def get_subtitle_by_timestamp(srt_file_path, timestamp):
    with open(srt_file_path, 'r') as srt_file:
        srt_content = srt_file.read()

    subtitles = srt_content.strip().split('\n\n')
    for subtitle in subtitles:
        subtitle_lines = subtitle.split('\n')
        subtitle_timestamp = subtitle_lines[1]
        start_time, end_time = subtitle_timestamp.split(' --> ')
        if start_time <= timestamp <= end_time:
            subtitle_text = ' '.join(subtitle_lines[2:])
            return start_time, end_time, subtitle_text

    return None

def find_missing_word(str1, str2):
    index_value=[]
    str1_list = str1.split()
    str2_list = str2.split()
    str2_list = str2.strip()
    missing_word = [word for word in str1_list if word not in str2_list]
    for i in missing_word:
      if i in str1_list:
        index_value.append(str1_list.index(i)+1)
    return index_value

#speaker diarization workflow
async def speaker_diarization_flow(PATH_TO_FILE):
  audio = extract_write_audio(PATH_TO_FILE)
  data = ''
  DEEPGRAM_API_KEY = "3dc39bf904babb858390455b1a1399e221bf87f8"
  deepgram = Deepgram(DEEPGRAM_API_KEY)
  with open(PATH_TO_FILE, 'rb') as audio:
       source = {'buffer': audio, 'mimetype': 'audio/wav'}
       transcription =  await deepgram.transcription.prerecorded(source, {'punctuate': True, 'diarize': True})
       transcript,final_data =  await compute_speaking_time(transcription,data)
  return final_data

# speaker diarization main funciton
async def speaker_diarization(PATH_TO_FILE):
  data = await speaker_diarization_flow(PATH_TO_FILE)
  print("data is", data)
  return data


#find filler words
def filler_words_finder(result_data):
  word_map_prior_edit=set()
  word_map_after_edit=set()
  #my filler words sample
  filler_words={'um','ah','you know','mmm','er','uh','Hmm','actually','basically','seriously','mhm','uh huh','uh','huh','ooh','aah','ooh', 'likely', 'hmm'}
  filler_words_timestamp=set()
  for keys  in result_data:
    if keys == 'segments':
        prev=0
        for i in result_data[keys]:
            for word in i['whole_word_timestamps']:
                lower_case = re.sub(r'\W','',word['word'].lower())
                word_map_prior_edit.add(word['timestamp'])
                if lower_case in filler_words or lower_case.startswith(('hm','aa','mm','oo')):
                    print(word['word'].lower(),word['timestamp'])
                    filler_words_timestamp.add(word['timestamp'])
                    prev=word['timestamp']
                    continue
                word_map_after_edit.add((prev,word['timestamp']))
                prev=word['timestamp']
  return word_map_after_edit, filler_words_timestamp

from moviepy.editor import AudioClip, VideoFileClip, concatenate_videoclips
import math
import sys
import subprocess
import os
import shutil

def find_speaking(audio_clip, window_size=0.1, volume_threshold=0.01, ease_in=0.25):
    # First, iterate over audio to find all silent windows.
    num_windows = math.floor(audio_clip.end/window_size)
    window_is_silent = []
    for i in range(num_windows):
        s = audio_clip.subclip(i * window_size, (i + 1) * window_size)
        v = s.max_volume()
        window_is_silent.append(v < volume_threshold)
    print("silent window", window_is_silent)
    # Find speaking intervals.
    speaking_start = 0
    speaking_end = 0
    speaking_intervals = []
    for i in range(1, len(window_is_silent)):
        e1 = window_is_silent[i - 1]
        e2 = window_is_silent[i]
        # silence -> speaking
        if e1 and not e2:
            speaking_start = i * window_size
        # speaking -> silence, now have a speaking interval
        if not e1 and e2:
            speaking_end = i * window_size
            new_speaking_interval = [speaking_start - ease_in, speaking_end + ease_in]
            # With tiny windows, this can sometimes overlap the previous window, so merge.
            need_to_merge = len(speaking_intervals) > 0 and speaking_intervals[-1][1] > new_speaking_interval[0]
            if need_to_merge:
                merged_interval = [speaking_intervals[-1][0], new_speaking_interval[1]]
                speaking_intervals[-1] = merged_interval
            else:
                speaking_intervals.append(new_speaking_interval)
    return speaking_intervals

def merge_overlapping_time_intervals(intervals):
    stack = []
    result=[intervals[0]]

    for interval in intervals:
            interval2=result[-1]

            if overlap(interval,interval2):
                result[-1] = [min(interval[0],interval2[0]),max(interval[1],interval2[1])]
            else:
                result.append(interval)
      
    return result

def overlap(interval1,interval2):
            return min(interval1[1],interval2[1])-max(interval1[0],interval2[0]) >= 0

#assembly ai endpoints
import requests
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
upload_endpoint = "https://api.assemblyai.com/v2/upload"

headers = {
	"authorization": "05e515bf6b474966bc48bbdd1448b3cf",
	"content-type": "application/json"
}

def upload_to_AssemblyAI(save_location):
  CHUNK_SIZE = 5242880
  def read_file(filename):
    with open(filename, 'rb') as _file:
      while True:
        print("chunk uploaded")
        data = _file.read(CHUNK_SIZE)
        if not data:
          break
        yield data
  
  upload_response = requests.post(
	  upload_endpoint,
	  headers=headers, data=read_file(save_location)
	)
  print(upload_response.json())	
  audio_url = upload_response.json()['upload_url']
  print('Uploaded to', audio_url)
  return audio_url


def start_analysis(audio_url,type):
	## Start transcription job of audio file
  data = {
	    'audio_url': audio_url,
	    'iab_categories': True,
	    'content_safety': True,
	    "summarization": True,
	    "summary_type": "bullets",
      "summary_model":type
	}
  if type=='conversational':
    data["speaker_labels"]= True

  transcript_response = requests.post(transcript_endpoint, json=data, headers=headers)
  print(transcript_response.json())
  transcript_id = transcript_response.json()['id']
  polling_endpoint = transcript_endpoint + "/" + transcript_id
  print("Transcribing at", polling_endpoint)
  return polling_endpoint

def get_analysis_results(polling_endpoint):	
  status = 'submitted'

  while True:
    print(status)
    polling_response = requests.get(polling_endpoint, headers=headers)
    status = polling_response.json()['status']
	  # st.write(polling_response.json())
	  # st.write(status)
    if status == 'submitted' or status == 'processing' or status == 'queued':
      print('not ready yet')
      sleep(10)
    
    elif status == 'completed':
      print('creating transcript')
      return polling_response
      break
    
    else:
      print('error')
      return False
      break

def pii_redact(audiourl,options):
  print(options,audiourl)
  endpoint = "https://api.assemblyai.com/v2/transcript"
  json = {
    "audio_url": audiourl,
    "redact_pii": True,
    "redact_pii_audio": True,
    "redact_pii_policies": options
  }

  headers = {
      "authorization": "05e515bf6b474966bc48bbdd1448b3cf",
      "content-type": "application/json",
  }

  response = requests.post(endpoint, json=json, headers=headers)
  print(response.json())
  transcript_id = response.json()['id']
  polling_endpoint = endpoint + "/" + transcript_id
  return polling_endpoint

def pii_redact_audio(polling_endpoint):
  status = 'submitted'
  headers = {
      "authorization": "05e515bf6b474966bc48bbdd1448b3cf",
      "content-type": "application/json",
  }
  while True:
    print(status)
    polling_response = requests.get(polling_endpoint, headers=headers)
    status = polling_response.json()['status']
    if status == 'submitted' or status == 'processing' or status == 'queued':
      print('not ready yet')
      sleep(10)
    
    elif status == 'completed':
      print('creating transcript')
      return polling_response
      break
    
    else:
      print('error')
      return False
      break

def download_redact_audio(pooling_enpoint):
  headers = {
      "authorization": "05e515bf6b474966bc48bbdd1448b3cf",
      "content-type": "application/json",
  }

  redacted_audio_response = requests.get(polling_endpoint + "/redacted-audio",headers=headers)
  print(redacted_audio_response.json())
  redacted_audio = requests.get(redacted_audio_response.json()['redacted_audio_url'])
  with open('redacted_audio.mp3', 'wb') as f:
    f.write(redacted_audio.content)

def redact_audio_video_display(vd,audio):
  audioclip = AudioFileClip(audio)
  clip = VideoFileClip(vd)
  videoclip = clip.set_audio(audioclip)
  videoclip.write_videofile("Redacted_video.mp4")
  st.video("Redacted_video.mp4")


#Find Remove Word
def find_missing_word(str1, str2):
    str1_list = str1.split()
    str2_list = str2.split()
    str2_list = str2.strip()
    missing_word = {word.lower() for word in str1_list if word not in str2_list}
    print(missing_word)
    return missing_word


#Find Remove Words Timestamp
def remove_words_finder(result_data,text):
  word_map_prior_edit=set()
  word_map_after_edit=set()
  #my remove words 
  filler_words = text
  filler_words_timestamp=set()
  for keys  in result_data:
    if keys == 'segments':
        prev=0

        for i in result_data[keys]:
            for word in i['whole_word_timestamps']:
                lower_case = re.sub(r'\W','',word['word'].lower())
                word_map_prior_edit.add(word['timestamp'])
                if lower_case in filler_words:
                    print(word['word'].lower(),word['timestamp'])
                    filler_words_timestamp.add(word['timestamp'])
                    prev=word['timestamp']
                    continue
                word_map_after_edit.add((prev,word['timestamp']))
                prev=word['timestamp']
  return word_map_after_edit, filler_words_timestamp

def are_sentences_similar(s1, s2):
    matcher = SequenceMatcher(None, s1, s2)
    similarity_ratio = matcher.ratio()
    if similarity_ratio >= 0.9:
        return True
    else:
        return False

def cut_sentence(result, text_input):
  import pysrt
  subs = pysrt.from_string(result["srt"])
  sub_dict = {}
  for sub in subs:
    sub_dict[sub.text] = [sub.start, sub.end]


  from nltk.tokenize import sent_tokenize
  sentences = sent_tokenize(text_input)

  intervals_to_keep = []

  for text_line in sentences:
    for key, value in sub_dict.items():
      if are_sentences_similar(text_line, key):
        intervals_to_keep.append(value)

  print("intervals to keep", intervals_to_keep)
  return intervals_to_keep

async def main(uploaded_video,model_selected):
    preview = st.video(uploaded_video)
    try:
      vid = uploaded_video.name
      with open(vid, mode='wb') as f:
        f.write(uploaded_video.read()) # save video to disk
    except:
      yt = YouTube(uploaded_video)
      yt.streams.filter(file_extension="mp4").get_by_resolution("360p").download(filename="youtube.mp4")
      vid = "youtube.mp4"
    finally:
      name = vid.split('.')[0]
      #extracting the transcription result
      with st.spinner('Transcribing Video, Wait for it...'):
        result = transcribe_video(vid,model_selected)
        trans_text = st.text_area("Edit Transcript",result["text"])
        col1, col2, col3, col4, col5, col6, col7 ,col8 = st.columns([1,1,1,1,1,1,1,1])
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 ,tab8= st.tabs(["Remove Filler Words","Silence Remover" ,"Download Transcription", "Perform Speaker Diarization","Content Analyzer", "Sentiment Analysis", "Remove or Re-arrange Video by Sentence","Remove words"])
      
      with tab1:
        filler_word = st.button('Edit/Remove Filler Words with a click of a button')
        if filler_word:

          word_map_after_edit, filler_words_timestamp = filler_words_finder(result)
          final_intervals = merge_overlapping_time_intervals(sorted(list(word_map_after_edit)))
          subclips=[]
          for start,end in final_intervals:
              clip = VideoFileClip(vid)
              tmp = clip.subclip(start,end)
              print(start,end,tmp.duration)
              subclips.append(tmp)
          #concatenate subclips without filler words
          final_clip = concatenate_videoclips(subclips)
          final_clip.write_videofile(f"remove_{vid}", codec='libx264')
          preview = st.video(f"remove_{vid}")

      with tab2:
        identify_remove_silence = st.button('Remove Silence Part')

        if identify_remove_silence:
          vid1 = VideoFileClip(vid)
          intervals_to_keep = find_speaking(vid1.audio)
          print("Keeping intervals: " + str(intervals_to_keep))
          keep_clips = [vid1.subclip(start, end) for [start, end] in intervals_to_keep]
          if len(keep_clips) > 0:
            try:
              edited_video = concatenate_videoclips(keep_clips)
              edited_video.write_videofile(f"remove_silence_{vid}", codec='libx264')
              preview = st.video(f"/content/remove_silence_{vid}")
              st.write('Silence Removed!')
            except IndexError as i:
              print("No silence dedected!")
              st.write('No silence detected greater than 0.6 seconds!')
          else:
            print("the duration is empty. the entire audio is silent")
          st.write('Silence Removed!')

      with tab3:
        download = st.download_button('Download Transcription', result["srt"],f'{name}.srt')
        results_to_word_srt(result, "modified_word.srt")        
        if download:
          st.write('Thanks for downloading!')

      
      with tab4:
        identify_download_speaker = st.button('Perform Speaker Diarization')
        if identify_download_speaker:
          results  = await speaker_diarization(vid)
          download_speaker = st.download_button("download speaker_diarization",results,'diarization_stats.txt')
          if download_speaker:
            st.write('Thanks for downloading!')

      with tab5:
        type = st.selectbox('Summary Type?',('informative', 'conversational', 'catchy'))
        Analyze_content = st.button("Start Content Analysis")
        if Analyze_content:
          audio = extract_write_audio(vid)
          audio_url = upload_to_AssemblyAI("audio.wav")
          # start analysis of the file
          polling_endpoint = start_analysis(audio_url,type)
          # receive the results
          results = get_analysis_results(polling_endpoint)

          # separate analysis results
          summary = results.json()['summary']
          content_moderation = results.json()["content_safety_labels"]
          topic_labels = results.json()["iab_categories_result"]

          my_expander1 = st.expander(label='Summary')
          my_expander2 = st.expander(label='Content Moderation')
          my_expander3 = st.expander(label='Topic Discussed')

          with my_expander1:
            st.header("Video summary")
            st.write(summary)

          with my_expander2:
              st.header("Sensitive content")
              if content_moderation['summary'] != {}:
                st.subheader('🚨 Mention of the following sensitive topics detected.')
                moderation_df = pd.DataFrame(content_moderation['summary'].items())
                moderation_df.columns = ['topic','confidence']
                st.dataframe(moderation_df, use_container_width=True)
              else:
                st.subheader('✅ All clear! No sensitive content detected.')

          with my_expander3:
            st.header("Topics discussed")
            topics_df = pd.DataFrame(topic_labels['summary'].items())
            topics_df.columns = ['topic','confidence']
            topics_df["topic"] = topics_df["topic"].str.split(">")
            expanded_topics = topics_df.topic.apply(pd.Series).add_prefix('topic_level_')
            topics_df = topics_df.join(expanded_topics).drop('topic', axis=1).sort_values(['confidence'], ascending=False).fillna('')
            st.dataframe(topics_df, use_container_width=True)

      with tab6:
        sentiment = st.button("Sentiment Analysis")
        if sentiment:
          from transformers import pipeline
          classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
          sequence_to_classify = result["text"]
          candidate_labels = ['positive', 'negative', 'neutral']
          result = classifier(sequence_to_classify, candidate_labels)
          table = pd.DataFrame(result['scores'], index=result['labels'], columns=['scores']).sort_values(by='scores', ascending=False)
          st.table(table)

      with tab7:
        edit_video_by_edit_sentence = st.button('Remove or Re-arrange Video by Sentence')
        if st.session_state.get('button') != True:
          st.session_state['button'] = edit_video_by_edit_sentence # Saved the state
        if st.session_state['button'] == True:
          text_input = st.text_area("Enter the text you want to edit")
          if st.button("Edit"):
            final_intervals = cut_sentence(result, text_input)
            subclips=[]
            for start_time,end_time in final_intervals:
              pattern = r'(\d+):(\d+):(\d+),(\d+)'
              start_h, start_m, start_s, start_ms = map(int, re.search(pattern, str(start_time)).groups())
              end_h, end_m, end_s, end_ms = map(int, re.search(pattern, str(end_time)).groups())
              start_time_seconds = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
              end_time_seconds = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000
              clip = VideoFileClip(vid)
              tmp = clip.subclip(start_time_seconds,end_time_seconds)
              print(start_time_seconds,end_time_seconds)
              subclips.append(tmp)
            final_clip = concatenate_videoclips(subclips)
            final_clip.write_videofile(f"remove_sentence_rearrange_{vid}")
            preview = st.video(f"remove_sentence_rearrange_{vid}")

      with tab8:
        remove_text_input = st.text_area("Enter The Text And Remove Word")
        if st.button("Edit by word"):
          Missing_word = find_missing_word(trans_text,remove_text_input)
          word_map_after_edit, filler_words_timestamp = remove_words_finder(result,Missing_word)
          final_intervals = merge_overlapping_time_intervals(sorted(list(word_map_after_edit)))
          subclips=[]
          for start,end in final_intervals:
              clip = VideoFileClip(vid)
              tmp = clip.subclip(start,end)
              print(start,end,tmp.duration)
              subclips.append(tmp)
          #concatenate subclips without filler words
          final_clip = concatenate_videoclips(subclips)
          final_clip.write_videofile(f"remove_word_{vid}")
          preview = st.video(f"remove_word_{vid}")

#Model_type = st.sidebar.selectbox("Choose Model",('Medium - Use this model for filler word removal'),0)
upload_video = st.sidebar.file_uploader("Upload the file")


#if Model_type.startswith("Medium"):
model_selected = 'large'

if upload_video is not None:
  asyncio.run(main(upload_video,model_selected))
