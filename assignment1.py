"""
This script is based on the MatLab code by Shekhar Nayak, Campus Fryslân, University of Groningen.

Run in command line with optional parameters 'orig_sr', 'target_sr' and/or 'model_orders'
(see bottom of file for default values of these parameters).
Examples:
$ python assignment1.py
$ python assignment1.py --orig_sr 22050
$ python assignment1.py --orig_sr 22050 --target_sr 4000
$ python assignment1.py --orig_sr 22050 --target_sr 4000 --model_orders 8 10 12
$ python assignment1.py --model_orders 4 8 16 32 

See the output folder and the terminal for the results.
"""
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import numpy as np
import os
from pathlib import Path
from scipy.signal import lfilter, freqz

def plot_waveform(digit, audio, target_sr):
    """
    Plots waveform of audio, see results in ./output/waveforms

    :param str digit: digit that is pronounced in the audio
    :param list audio: list of frequencies
    :param int target_sr: desired sampling rate
    """
    L = len(audio)
    t = np.linspace(0, L/target_sr, L)

    plt.plot(t, audio)
    plt.title(f"Waveform of digit {digit}")
    plt.savefig(f"./output/waveforms/waveform{digit}.png")
    plt.close()

def calculate_pitchperiod_and_f0(audio, digit, voicing, letter):
    """
    Print pitch period and fundamental frequency of audio, see terminal for results

    :param list audio: list of frequencies
    :param str digit: digit that is pronounced in the audio
    :param str voicing: voiced or voiceless
    :param str letter: the specific ortographic letter that is pronounced
    """
    f0_list = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length = 1024)
    f0 = np.average(f0_list)
    print(f"Pitch period and F0 for {voicing} {letter} in digit {digit}:\n")
    print(f"fundamental freq: {f0}")
    print(f"pitch period: {1/f0}")
    print("----------------------------")


def autocorrelation(inp, lags, order="", title=""):
    """
    Plot autocorrelation

    :param ...
    """
    autocorr = plt.acorr(inp, usevlines=False, maxlags= lags-1)
    plt.close()
    lags, acs = autocorr[0], autocorr[1]
    plt.plot(lags, acs)
    str_inp = str(inp)
    plt.savefig(f"./output/autocorrelation/autocorrelation_{order}_{title}.png")
    plt.close()

def plot_spectrum(audio, target_sr, digit, voicing, letter, lpc_envelope=False, orders=None):
    """
    Plots magnitude spectrum of audio (with or without LPC envelope), see ./output/spectra and
    ./output/spectra_with_lpc for results

    :param list audio: list of frequencies
    :param int target_sr: desired sampling rate
    :param str digit: digit that is pronounced in the audio
    :param str voicing: voiced or voiceless
    :param str letter: the specific ortographic letter that is pronounced
    :param bool lpc_envelope: boolean for whether or not to plot the LPC envelope
    :param list orders: model orders that need to be plotted
    """
    # get middle of audio
    sound_middle = len(audio)//2
    # one second contains 40 frames of 25ms, i.e. sr should be divided by 40
    frame_samples_n = target_sr//40
    # take the samples for the frame from the middle
    y_frame_s = audio[sound_middle:sound_middle+frame_samples_n]

    y_mag_spec = 20*np.log10(abs(np.fft.fft(y_frame_s)))
    y_mag_spec_final=np.fft.fftshift(y_mag_spec)
    f = np.linspace(-(target_sr//2), (target_sr//2), y_frame_s.shape[0])

    plt.plot(f, y_mag_spec_final, color='k')
    plt.ylabel("Magnitude (dB)")
    plt.xlabel("Frequency (Hz)")
    if lpc_envelope:
        for order in orders:
            a = librosa.lpc(y_frame_s, order=order)
            # create numerator coefficient (equivalent of [0 -a(2:end)] in MatLab)
            numerator = np.concatenate([[0], [-(i) for i in a[1:]]])

            est_y_frame_s = lfilter(numerator, 1, y_frame_s) 
            e = y_frame_s - est_y_frame_s
            g = math.sqrt(sum(np.square(e)))

            autocorrelation(e, y_frame_s.shape[0], order=order, title="error")
            autocorrelation(y_frame_s, y_frame_s.shape[0], order=order, title="frame")

            h = freqz(g, a, f, fs=target_sr)
            plt.plot(f, y_mag_spec_final)
            plt.plot(f, 20*np.log10(abs(h[1])), label=f"LPC order {order}", alpha=1)
            plt.legend()

        plt.title(f"Magnitude spectrum for {voicing} '{letter}' in digit {digit} with LPC envelope")
        plt.savefig(f"./output/spectra_with_lpc/spectrum_{digit}_{voicing}_{letter}.png") 

    # else plot magnitude spectrum without lpc envelope
    else:
        plt.title(f"Magnitude spectrum for {voicing} '{letter}' in digit {digit}")
        plt.savefig(f"./output/spectra/spectrum_{digit}_{voicing}_{letter}.png")
    plt.close()

def main(orig_sr, target_sr, model_orders):

    data_dir = Path("recordings")

    for filepath in data_dir.glob('[0-9]*.wav'):
        y, _ = librosa.load(filepath, sr=orig_sr)
        # downsample audiofile to target_sr
        downsampled_y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

        basename = os.path.basename(filepath)

        # voiced and unvoiced regions of audiofiles
        if len(basename.split("_")) == 3:
            digit, voicing, letter = basename.replace(".wav", "").split("_")
            # print pitch period and f0 of voiced regions
            if voicing == "voiced":
                calculate_pitchperiod_and_f0(y, digit, voicing, letter)
            # plot magnitude spectrum for all 4 regions
            plot_spectrum(downsampled_y, target_sr, digit, voicing, letter)
            # plot magnitude spectrum with lpc envelope (various model orders) for all 4 regions
            plot_spectrum(downsampled_y, target_sr, digit, voicing, letter, lpc_envelope=True, orders=model_orders)

        # plot waveforms of complete audiofiles
        else:
            digit = basename.replace(".wav", "")
            plot_waveform(digit, y, target_sr)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--orig_sr', default=22050, type=int)
    argparser.add_argument('--target_sr', default=8000, type=int)
    argparser.add_argument('--model_orders', '--arg', nargs='+', type=int, default=[12])
    args = argparser.parse_args()

    main(
        args.orig_sr,
        args.target_sr,
        args.model_orders
    )