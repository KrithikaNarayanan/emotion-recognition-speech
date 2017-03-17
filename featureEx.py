import librosa
import numpy as np
import csv

#load audio file
y, sr = librosa.load("/home/krithika/fau/test-neg/Mont_01_071_00.wav",sr=16000,duration=1)

#generate spectrogram
spec=librosa.stft(y=y,n_fft=400,hop_length=160)

#generate mfcc
mfcc = librosa.feature.mfcc(y=y,sr=sr,n_fft=400,hop_length=160,n_mfcc=13)

#make the 13x101 array into a 1D array (1x1313)
array2 = np.ravel(mfcc)

# zero pad to get uniform length input vectors
#array3=np.zeros(1313)
#array3[:array2.shape[0]]=array2
#array3.tofile('sample.csv',sep=',')

#compute energy from spectrogram
S, phase = librosa.magphase(librosa.stft(y=y,n_fft=400,hop_length=160))
rms = librosa.feature.rmse(S=S)
deri1=librosa.feature.delta(rms)
deri1= np.ravel(deri1)
deri2=librosa.feature.delta(rms, order=2)
deri2= np.ravel(deri2)

#concatenate all 3 arrays
outputArray=np.concatenate((array2,deri1,deri2),axis=0)
outputArray=np.hstack((array2,deri1,deri2))

#write output array into csv file
#outputArray.tofile('sample.csv',sep=',')

f= open('/home/krithika/trial.csv','a')
writer = csv.writer(f,delimiter=',')
writer.writerow(outputArray)

