from keras.models import Sequential
from keras.layers import Dense
import numpy
import os
import csv 

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataset = numpy.loadtxt("input.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:1313]
Y = dataset[:,1313]

# create model
model = Sequential()
model.add(Dense(1313, input_dim=1313, init='uniform', activation='tanh'))
model.add(Dense(1000, init='uniform', activation='relu'))
model.add(Dense(620, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)

dataset = numpy.loadtxt("testing.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:1313]
Y = dataset[:,1313]

# evaluate the model
scores = model.evaluate(X, Y)

result= model.predict_classes(X, batch_size=1, verbose=1)
print "\n"
print result
print len(result)

f= open('/home/krithika/test-result.csv','a')
writer = csv.writer(f,delimiter=',')
for row in result:
	writer.writerow(result[row])

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# serialize model to JSON
model_json = model.to_json()
with open("model-2Cl.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("mode-2Cl.h5")
print("Saved model to disk")
