from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset for training
dataset = numpy.loadtxt("midni_trail.csv",delimiter=",")
numpy.random.shuffle(dataset)
numpy.random.shuffle(dataset)
numpy.random.shuffle(dataset)
numpy.random.shuffle(dataset)

# split into input (X) and output (Y) variables
X = dataset[:,0:1515]
Y = dataset[:,1515:1520]

# create model
model = Sequential()
model.add(Dense(1515, input_dim=1515, init='he_uniform', activation='sigmoid'))
model.add(Dense(1000, init='he_uniform', activation='sigmoid'))
#model.add(Dense(600, init='he_uniform', activation='relu'))
model.add(Dense(5, init='he_uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
# Fit the model
model.fit(X, Y, validation_split=0.05, nb_epoch=20, batch_size=5,verbose=1)

# load dataset for testing
dataset = numpy.loadtxt("midni_trail_test.csv",delimiter=",")
numpy.random.shuffle(dataset)
numpy.random.shuffle(dataset)
numpy.random.shuffle(dataset)
numpy.random.shuffle(dataset)

# split into input (X) and output (Y) variables
X = dataset[:,0:1515]
Y = dataset[:,1515:1520]

preds= model.predict_proba(X, batch_size=1, verbose=1)
newPreds = numpy.zeros_like(preds)
newPreds[numpy.arange(len(preds)), preds.argmax(1)] = 1

#print "\n"
#print newPreds

# evaluate the model
scores = model.evaluate(X, Y,verbose=1)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
