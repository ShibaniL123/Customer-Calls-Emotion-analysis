# Importing required libraries 
# Keras
import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Other  
import librosa
import librosa.display
import json
import numpy as np
import pandas as pd
# import seaborn as sns
import os
import pickle
# from tqdm import tqdm
from numpy import savetxt
from numpy import loadtxt

import wave
import pydub

from splitterkit import readwave, writewave, split_s, merge

#########     EMOTION     ###################################
# load json and create emotion model
json_file = open('REQ/emo/arch_emo.json', 'r')
model_emo_json = json_file.read()
json_file.close()
model_emo = model_from_json(model_emo_json)

# load weights into emotion model
model_emo.load_weights("REQ/emo/weights_emo.h5")

mean_emo = loadtxt('REQ/emo/mean_emo.csv', delimiter=',')
std_emo = loadtxt('REQ/emo/std_emo.csv', delimiter=',')
##############################################################


#########     BINARY#SATISFACTION     ########################
# load json and create bisat model
json_file = open('REQ/bisat/arch_bisat.json', 'r')
model_bisat_json = json_file.read()
json_file.close()
model_bisat = model_from_json(model_bisat_json)

# load weights into bisat model
model_bisat.load_weights("REQ/bisat/weights_bisat.h5")

mean_bisat = loadtxt('REQ/bisat/mean_bisat.csv', delimiter=',')
std_bisat = loadtxt('REQ/bisat/std_bisat.csv', delimiter=',')
###############################################################


# #########     DECEPTION     ###################################
# # load json and create dec model
# json_file = open('REQ/dec/arch_dec.json', 'r')
# model_dec_json = json_file.read()
# json_file.close()
# model_dec = model_from_json(model_dec_json)

# # load weights into dec model
# model_dec.load_weights("REQ/dec/weights_dec.h5")

# mean_dec = loadtxt('REQ/dec/mean_dec.csv', delimiter=',')
# std_dec = loadtxt('REQ/dec/std_dec.csv', delimiter=',')
# ###############################################################

satis = pd.read_csv('satis.csv')

def extract_features_emo(path):
	X, sample_rate = librosa.load(path
								,res_type='kaiser_fast'
								,duration=2.5
								,sr=44100
								,offset=0.5
								)

	sample_rate = np.array(sample_rate)
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=np.array(sample_rate), n_mfcc=100).T,axis=0)

	mfccs = (mfccs - mean_emo)/std_emo

	newdf = pd.DataFrame(data=mfccs).T
	return newdf

def extract_features_bisat(path):
	X, sample_rate = librosa.load(path
								,res_type='kaiser_fast'
								,duration=2.5
								,sr=44100
								,offset=0.5
								)

	sample_rate = np.array(sample_rate)
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=np.array(sample_rate), n_mfcc=13),axis=0)

	mfccs = np.pad(mfccs, (0, 216-mfccs.shape[0]), 'constant')
	mfccs = (mfccs - mean_bisat)/std_bisat

	newdf = pd.DataFrame(data=mfccs).T
	return newdf

def wave_file_length(wave_file_path):
	wave_file = wave.open(wave_file_path, mode = 'rb')
	length = float(wave_file.getnframes()/wave_file.getframerate())
	return length

def generate_wave_file_chunk(file, start, end):
	wave_file = pydub.AudioSegment.from_wav(file)
	return wave_file[start:end]

def chunk_wave_file(wave_file_path):
	length = wave_file_length(wave_file_path)
	wave_file_chunks = []
	#wave_file = pydub.AudioSegment.from_wav(wave_file_path)
	start = 0
	end = 10
	i=0
	while(end > start):
		if(end > length):
			end = length
		if(end > start):
			print(type(end))
			#wave_file_chunks.append(generate_wave_file_chunk(wave_file_path, int(start), int(end)))
			wave_file = pydub.AudioSegment.from_wav(wave_file_path)
			wave_file = wave_file[start:end]
			wave_file.export('temp'+str(i), format='wav')
			print("Chunking" + str(start)+ "to" + str(end))
			print(wave_file_chunks)
		start += 10
		end += 10
		i += 1
	return wave_file_chunks

def predict_chunks(wave_file_chunks, model):
	predictions = []
	for each in wave_file_chunks:
		predictions.append(model.predict(extract_features_emo(each)))
	return predictions

def save_wave_files(file_list):
	i = 0
	filename_list = []
	for each in file_list:
		each.export('temp'+str(i), format='wav')
		filename_list.append('temp'+str(i))
		i += 1
	return filename_list


def normalize_predictions(predictions):
	length = len(predictions)
	predictions_sum = np.zeroes(length)
	quarter_1_mark = 0.25 * length
	quarter_3_mark = 0.75 * length
	for i in range(length):
		if(i<=quarter_1_mark):
			predictions_sum += predictions[i]
		elif(i>quarter_1_mark and i<=quarter_3_mark):
			predictions_sum += 2*predictions[i]
		else:
			predictions_sum += 3*predictions[i]
		predictions_normalized = predictions_sum / 6
	return predictions_normalized

def delete_files(filename_list):
	for each in filename_list:
		os.remove(each)

for i in os.listdir('CallRecordings'):

	path = 'CallRecordings/' + i

	chunk_wave_file(path)
	#delete_files(filename_list)

	# #### EMOTION PREDICTION
	# newdf = extract_features_emo(path)

	# # Apply emo predictions 
	# newdf= np.expand_dims(newdf, axis=2)
	# newpred = model_emo.predict(newdf, 
	# 						batch_size=16, 
	# 						verbose=1)

	filename = 'REQ/emo/labels_emo'
	infile = open(filename,'rb')
	lb = pickle.load(infile)
	infile.close()

	# Get the final predicted emotion
	emo = newpred.argmax(axis=1)
	emo = emo.astype(int).flatten()
	emo = (lb.inverse_transform((emo)))
	print(emo)



	#### BINARY SATISFACTION PREDICTION
	newdf = extract_features_bisat(path)

	# Apply bisat predictions 
	newdf= np.expand_dims(newdf, axis=2)
	newpred = model_bisat.predict(newdf, 
							batch_size=16, 
							verbose=1)

	filename = 'REQ/bisat/labels_bisat'
	infile = open(filename,'rb')
	lb = pickle.load(infile)
	infile.close()

	# Get the final predicted bisat
	bisat = newpred.argmax(axis=1)
	bisat = bisat.astype(int).flatten()
	bisat = (lb.inverse_transform((bisat)))
	print(bisat)



	#### DECEPTION PREDICTION


	satis = satis.append({'Filename' : path , 'Emotion' : emo , 'Satisfaction' : bisat , 'Deception' : '-'}, ignore_index=True)
	
satis.to_csv('satis.csv', index=False)	

