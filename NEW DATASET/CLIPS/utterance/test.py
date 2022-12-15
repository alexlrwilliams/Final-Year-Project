import os
import speech_recognition as sr
import json

def convert_audio_to_text(audio_file):
  # Create a Recognizer object
  recognizer = sr.Recognizer()

  # Read the audio file
  with sr.AudioFile(audio_file) as source:
    audio = recognizer.record(source)

  try:
    # Convert the audio to text
    transcript = recognizer.recognize_google(audio)
  except:
    transcript = ""

  return transcript

start_index = 13
end_index = 49
data = {}

for i in range(start_index, end_index+1):
  file_path = '3_{}'.format(i)

  if not os.path.exists(file_path + ".wav"):
    command2mp3 = "ffmpeg -i {}.mp4 {}.mp3".format(file_path, file_path)
    command2wav = "ffmpeg -i {}.mp3 {}.wav".format(file_path, file_path)
    os.system(command2mp3)
    os.system(command2wav)

  transcript = convert_audio_to_text("{}.wav".format(file_path))
  data[file_path] = transcript

  # Convert the object to JSON
  json_data = json.dumps(data)

  # Save the JSON to a file
  with open('data.json', 'w') as f:
    f.write(json_data)
