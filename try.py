import librosa
import numpy as np

def calculate_pitch_and_f0(audio):
    """
    Print pitch and fundamental frequency of audio, see terminal for results

    :param str filepath: path to the .wav file
    :param list audio: list of frequencies
    :param str digit: digit that is pronounced in the audio
    :param str voicing: voiced or voiceless
    :param str letter: the specific ortographic letter that is pronounced
    """
    f0_list = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length = 1024)
    f0 = np.average(f0_list)
    print(f"fundamental freq: {f0}")
    print(f"pitch: {1/f0}")
    print("----------------------------")

y, sr = librosa.load("./recordings/2_voiced_oo.wav")
print(sr)
print(len(y))
calculate_pitch_and_f0(y)
