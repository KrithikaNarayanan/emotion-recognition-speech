import librosa
import numpy as np
import csv

filename="/home/krithika/"

with open("wave_file_list.txt",'r') as f:
    for line in f:
	temp = filename+line[:-1]
	#print line
	
	#load audio file
	y, sr = librosa.load(temp,sr=16000,duration=1)

	#generate spectrogram
	spec=librosa.stft(y=y,n_fft=400,hop_length=160)

	#generate mfcc
	mfcc = librosa.feature.mfcc(y=y,sr=sr,n_fft=400,hop_length=160,n_mfcc=13)

	#make the 13x101 array into a 1D array (1x1313)
	array2 = np.ravel(mfcc)

	#compute energy from spectrogram
	S, phase = librosa.magphase(librosa.stft(y=y,n_fft=400,hop_length=160))
	rms = librosa.feature.rmse(S=S)
	deri1=librosa.feature.delta(rms)
	deri1= np.ravel(deri1)
	deri2=librosa.feature.delta(rms, order=2)
	deri2= np.ravel(deri2)

	#set label 
	label_A=np.array([1,0,0,0,0])
	label_E=np.array([0,1,0,0,0])
	label_N=np.array([0,0,1,0,0])
	label_P=np.array([0,0,0,1,0])
	label_R=np.array([0,0,0,0,1])


	#concatenate all 4 arrays
	outputArray=np.concatenate((array2,deri1,deri2,label_A),axis=0)
	outputArray=np.hstack((array2,deri1,deri2,label_A))

	#write output array into csv file
	f= open('/home/krithika/chumma.csv','a')
	writer = csv.writer(f,delimiter=',')
	writer.writerow(outputArray)
	

	
