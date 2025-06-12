!pip3 install whisper-timestamped

import whisper_timestamped
help(whisper_timestamped.transcribe)

!git lfs install

!git clone https://huggingface.co/openai/whisper-large-v3

!git lfs pull

import whisper_timestamped as whisper

import os

model = whisper.load_model("/whisper-large-v3", device="cuda")

# 資料夾路徑
folder_path = '/AICUP_DATA'

# 讀出所有 .wav 檔案
wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]

# 根據數字排序（把字串轉成 int 排序）
wav_files_sorted = sorted(wav_files, key=lambda x: int(os.path.splitext(x)[0]))


with open("task1.txt", "w", encoding="utf-8") as f:

  word_dict = {}
  temp = []

  for wav_file in wav_files_sorted:

      name = wav_file.replace(".wav", "")
      audio_path = f"/AICUP_DATA/{wav_file}"
      audio = whisper.load_audio(audio_path)
      result = whisper.transcribe(model, audio, language="en")

      temp = []
      full_text = result["text"].strip()

      print( name + "\t" + full_text )

      f.write(name + "\t" + full_text + "\n")