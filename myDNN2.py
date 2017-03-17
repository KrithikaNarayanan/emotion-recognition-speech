from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("trail-2.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:1515].astype(float)
Y = dataset[:,1515]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# create model
model = Sequential()
model.add(Dense(1515, input_dim=1515, init='he_uniform', activation='tanh'))
model.add(Dense(1000, init='he_uniform', activation='relu'))
model.add(Dense(600, init='he_uniform', activation='relu'))
model.add(Dense(5, init='he_uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['accuracy'])

# Fit the model
model.fit(X, dummy_y, validation_split=0.10, nb_epoch=20, batch_size=10)

# load dataset for testing
dataset = numpy.loadtxt("TraincTest.csv",delimiter=",",skiprows=2)

# split into input (X) and output (Y) variables
X = dataset[:,0:1515]
Y = dataset[:,1515:1520]

# evaluate the model
#scores = model.evaluate(X, Y)

result= loaded_model.predict_proba(X, batch_size=1, verbose=1)

print "\n"
print result

f= open('/home/krithika/test-result.csv','a')
writer = csv.writer(f,delimiter=',')
for row in result:
	writer.writerow(result[row])

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
