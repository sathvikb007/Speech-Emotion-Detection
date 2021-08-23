
# import required modules
from os import path
from pydub import AudioSegment
import glob, os
from tqdm import tqdm
  
# assign files
input_file = "hello.mp3"
output_file = "result.wav"
  
# convert mp3 file to wav file
# sound = AudioSegment.from_mp3(input_file)
# sound.export(output_file, format="wav")


os.chdir("./TrainAudioFiles/")
for file in tqdm(glob.glob("*.mp3")):
    input_file = file
    wav_file = file.replace(".mp3", ".wav")
    output_file = os.path.join('../wavdatatrain2/', wav_file)
    sound = AudioSegment.from_mp3(input_file)
    sound.set_channels(1)
    sound.export(output_file, format="wav")