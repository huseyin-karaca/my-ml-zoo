import torch
import os
import random
import torchaudio
import augment
from glob import glob
from torchaudio.transforms import Resample
import numpy as np
import torch


def augment_pitch(folder_path,fraction = 0.3):

    # parameters
    files = glob(os.path.join(folder_path, '*.wav'))
    sample_rate = 8000


    selected_files = random.sample(files, int(len(files) * fraction))
    for file_path in selected_files:

        ###
        effect_name = "pitch"
        random_pitch_shift = lambda: np.random.randint(-400, +400)
        random_pitch_shift = random_pitch_shift()
        effect_chain = augment.EffectChain().pitch(random_pitch_shift).rate(sample_rate)
        ###

        waveform, sr = torchaudio.load(file_path)
        augmented_waveform = effect_chain.apply(waveform, src_info={'rate': sr})
        os.makedirs(os.path.join(folder_path, f"{effect_name}"),exist_ok= True)
        new_file_path = os.path.join(folder_path, f"{effect_name}",f"{os.path.splitext(os.path.basename(file_path))[0]}_{effect_name}_{random_pitch_shift}.wav")
        torchaudio.save(new_file_path, augmented_waveform, sample_rate)

def augment_reverb(folder_path,fraction = 0.3):

    # parameters
    files = glob(os.path.join(folder_path, '*.wav'))
    sample_rate = 8000


    selected_files = random.sample(files, int(len(files) * fraction))
    for file_path in selected_files:

        ###
        effect_name = "reverb"
        x1 = lambda: np.random.randint(0, +50)
        x1 = x1()
        x2 = lambda: np.random.randint(0, +50)
        x2 = x2()
        x3 = lambda: np.random.randint(0, +50)
        x3 = x3()
        effect_chain = augment.EffectChain().reverb(x1,x2,x3).rate(sample_rate)
        ###

        waveform, sr = torchaudio.load(file_path)
        augmented_waveform = effect_chain.apply(waveform, src_info={'rate': sr})
        os.makedirs(os.path.join(folder_path, f"{effect_name}"),exist_ok= True)
        new_file_path = os.path.join(folder_path, f"{effect_name}",f"{os.path.splitext(os.path.basename(file_path))[0]}_{effect_name}_{x1}-{x2}-{x3}.wav")
        torchaudio.save(new_file_path, augmented_waveform, sample_rate)

def augment_speed(folder_path,fraction = 0.3):

    # parameters
    files = glob(os.path.join(folder_path, '*.wav'))
    sample_rate = 8000


    selected_files = random.sample(files, int(len(files) * fraction))
    for file_path in selected_files:

        ###
        effect_name = "speed"
        random_speed_factor = lambda: np.random.uniform(0.85, 1.15)
        random_speed_factor = random_speed_factor()
        effect_chain = augment.EffectChain().speed(random_speed_factor).rate(sample_rate)
        ###

        waveform, sr = torchaudio.load(file_path)
        augmented_waveform = effect_chain.apply(waveform, src_info={'rate': sr})
        os.makedirs(os.path.join(folder_path, f"{effect_name}"),exist_ok= True)
        new_file_path = os.path.join(folder_path, f"{effect_name}",f"{os.path.splitext(os.path.basename(file_path))[0]}_{effect_name}_{random_speed_factor}.wav")
        torchaudio.save(new_file_path, augmented_waveform, sample_rate)

def augment_vol(folder_path,fraction = 0.3):

    # parameters
    files = glob(os.path.join(folder_path, '*.wav'))
    sample_rate = 8000


    selected_files = random.sample(files, int(len(files) * fraction))
    for file_path in selected_files:

        ###
        effect_name = "vol"
        random_vol_factor = lambda: np.random.uniform(1, 30)
        random_vol_factor = random_vol_factor()
        effect_chain = augment.EffectChain().vol(random_vol_factor).rate(sample_rate)
        ###

        waveform, sr = torchaudio.load(file_path)
        augmented_waveform = effect_chain.apply(waveform, src_info={'rate': sr})
        os.makedirs(os.path.join(folder_path, f"{effect_name}"),exist_ok= True)
        new_file_path = os.path.join(folder_path, f"{effect_name}",f"{os.path.splitext(os.path.basename(file_path))[0]}_{effect_name}_{random_vol_factor}.wav")
        torchaudio.save(new_file_path, augmented_waveform, sample_rate)

def augment_noise(folder_path,fraction = 0.3):

    # parameters
    files = glob(os.path.join(folder_path, '*.wav'))
    sample_rate = 8000


    selected_files = random.sample(files, int(len(files) * fraction))
    for file_path in selected_files:

        waveform, sr = torchaudio.load(file_path)

        ###
        effect_name = "noise"
        random_snr = np.random.randint(10,15)
        noise_generator = lambda: torch.zeros_like(waveform).uniform_()
        effect_chain = augment.EffectChain().additive_noise(noise_generator, snr=random_snr)
        ###


        augmented_waveform = effect_chain.apply(waveform, src_info={'rate': sr})
        os.makedirs(os.path.join(folder_path, f"{effect_name}"),exist_ok= True)
        new_file_path = os.path.join(folder_path, f"{effect_name}",f"{os.path.splitext(os.path.basename(file_path))[0]}_{effect_name}_{random_snr}.wav")
        torchaudio.save(new_file_path, augmented_waveform, sample_rate)

