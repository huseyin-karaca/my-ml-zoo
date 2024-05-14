import tensorflow as tf
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav

def read_and_decode(filename):
    filename = filename.numpy().decode()  # Eager execution için TensorFlow tensörünü byte-string'den normal string'e çevir
    fs, audio = wav.read(filename)
    mfcc_feat =  mfcc(audio, 
                      samplerate=fs,
                      winlen=0.04,
                      winstep=0.02,
                      numcep=21,
                      appendEnergy = False,
                      nfft = 1024,
                      ceplifter=22,
                      preemph=0.97,
                      lowfreq=20,
                      highfreq=4000) 
    return mfcc_feat[:,1:].astype(np.float32)  # TensorFlow ile uyumlu olması için dtype'ı tf.float32 olarak belirt

def tf_read_and_decode(filename):
    # tf.py_function içinde kullanılacak wrapper fonksiyon
    [mfcc_features] = tf.py_function(read_and_decode, [filename], [tf.float32])
    return mfcc_features

def create_dataset(folder_path):
    filenames = []
    labels = []
    class_names = ['silence', 'other', 'mama', 'papa', 'cry']

    for class_index, class_name in enumerate(class_names):
        class_files = tf.io.gfile.glob(f'{folder_path}/{class_name}/*.wav')
        filenames.extend(class_files)
        labels.extend([class_index] * len(class_files))

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # Dosyaları oku ve ön işlemleri uygula
    dataset = dataset.map(lambda filename, label: (tf_read_and_decode(filename), label),
                          num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def prepare_for_training(dataset, batch_size=100):
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
