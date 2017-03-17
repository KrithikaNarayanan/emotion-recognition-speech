from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import csv

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

# load dataset for testing
dataset = numpy.loadtxt("my-trail.csv",delimiter=",",skiprows=5)
numpy.random.shuffle(dataset)
numpy.random.shuffle(dataset)
numpy.random.shuffle(dataset)
numpy.random.shuffle(dataset)

# split into input (X) and output (Y) variables
X = dataset[:,0:1515]
Y = dataset[:,1515:1520]

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

preds= model.predict_proba(X, batch_size=1, verbose=1)
#newPreds = numpy.zeros_like(preds)
#newPreds[numpy.arange(len(preds)), preds.argmax(1)] = 1

#print "\n"
#print newPreds
print preds

# evaluate the model
scores = model.evaluate(X, Y,verbose=1)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
