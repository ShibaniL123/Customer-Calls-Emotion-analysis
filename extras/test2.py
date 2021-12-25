from numpy import loadtxt
import numpy as np
import librosa

mean_bisat = loadtxt('REQ/bisat/mean_bisat.csv', delimiter=',')
std_bisat = loadtxt('REQ/bisat/std_bisat.csv', delimiter=',')

X, sample_rate = librosa.load('CallRecordings/1001_DFA_HAP_XX.wav'
								,res_type='kaiser_fast'
								,duration=2.5
								,sr=44100
								,offset=0.5
								)
                            
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=np.array(sample_rate), n_mfcc=13),axis=0)
mfccs = (mfccs - mean_bisat)/std_bisat

print(mfccs)