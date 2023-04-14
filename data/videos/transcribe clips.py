import sys
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

def generate_wav_if_file_not_exists(file_path):
    if not os.path.exists(file_path + ".wav"):
        command_convert_wav = "ffmpeg -i {}.mp4 {}.wav".format(file_path, file_path)
        os.system(command_convert_wav)

context_destination_path = "./context"
context_file_name = "3_{}_c"
utterance_destination_path = "./utterance"
utterance_file_name = "3_{}"
start_index = 50
end_index = 122
data = {}

for i in range(start_index, end_index+1):
    context_file_path = context_destination_path + '/' + context_file_name.format(i)
    utterance_file_path = utterance_destination_path + '/' + utterance_file_name.format(i)

    generate_wav_if_file_not_exists(context_file_path)
    generate_wav_if_file_not_exists(utterance_file_path)

    context = convert_audio_to_text("{}.wav".format(context_file_path))
    utterance = convert_audio_to_text("{}.wav".format(utterance_file_path))

    data[utterance_file_name.format(i)] = {
        'utterance': utterance,
        'speaker': 'PERSON 1',
        'context': [context],
        'context_speakers': [''],
        'sarcasm': True
    }

    # Convert the object to JSON
    json_data = json.dumps(data)

    # Save the JSON to a file
    with open('sarcasm_data.json', 'w') as f:
        f.write(json_data)
