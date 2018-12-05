import keras
import k_model
import numpy as np
import gc
import pickle

def getData():
	num_classes = 10
	img_rows, img_cols = 28, 28

	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

	if keras.backend.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	return x_train, y_train, x_test, y_test, input_shape


def trainModel(savePath):
	xTrain, yTrain, xTest, yTest, inputShape = getData()
	model = k_model.getModel(inputShape)
	model.summary()
	checkpoint = keras.callbacks.ModelCheckpoint(savePath, monitor = 'val_acc', verbose = 1, save_best_only = True,
	                                             mode = 'max')
	model.fit(x = xTrain, y = yTrain, validation_data = (xTest, yTest), batch_size = 64, epochs = 10, verbose = 1, callbacks = [checkpoint])


def loadModel(savePath):
	return keras.models.load_model(savePath)


def extractModelWeights(model):
	weights = []
	for i in range(len(model.layers)):
		weights.append(model.layers[i].get_weights())
	return weights


def setModelWeights(model, weights):
	for i in range(len(model.layers)):
		model.layers[i].set_weights(weights[i])


def evaluateCreatures(creaturePool, x, y):
	for individual in creaturePool:
		individual.evaluateFitness(x, y)





def main():
	import creature
	xTrain, yTrain, xTest, yTest, inputShape = getData()

	# trainModel('model.h5')
	model = loadModel('model.h5')
	model.summary()

	weights = model.get_weights()
	print(model.evaluate(xTest, yTest))
	population = 0
	generations = 0
	elitismRatio = 0.4
	mutationRate = 0.01
	creaturePool = []
	for i in range(population):
		print('Creating Creature ' + str(i))
		individual = creature.Creature(inputShape, weights)
		individual.pruneWeights()
		individual.evaluateFitness(xTest, yTest)
		creaturePool.append(individual)
		if i%10 == 0:
			keras.backend.clear_session()


	# evaluateCreatures(creaturePool, xTest, yTest)

	progress = []

	for gen in range(generations):
		print('Generation ' + str(gen))
		print('Max prev: ' + str(creaturePool[0].getFitness()))
		progress.append(creaturePool[0].getFitness())
		creaturePool.sort(key=lambda x: x.fitness, reverse=True)
		keptIndex = round(elitismRatio * population)
		for i in range(keptIndex):
			mateIndex = i
			while mateIndex == i:
				mateIndex = np.random.random_integers(0, keptIndex - 1)
			print('Mating creature ' + str(i) + ' and ' + str(mateIndex) + ' to ' + str(i + keptIndex))
			childCreature = creaturePool[i].crossover(creaturePool[mateIndex], mutationRate)
			childCreature.pruneWeights()
			childCreature.evaluateFitness(xTest, yTest)
			creaturePool[i + keptIndex] = childCreature
			if i > 0:
				creaturePool[i].mutate(mutationRate)
			if i % 10 == 0:
				keras.backend.clear_session()
		for i in range(keptIndex * 2, population):
			print('Creating Creature ' + str(i))
			newCreature = creature.Creature(inputShape, weights)
			newCreature.pruneWeights()
			newCreature.evaluateFitness(xTest, yTest)
			creaturePool[i] = newCreature
			if i % 10 == 0:
				keras.backend.clear_session()
		gc.collect()
		#keras.backend.clear_session()
		with open('best.creature', 'wb') as f:
			pickle.dump(creaturePool[0].prob, f)
		np.savetxt('progress.csv', np.asarray(progress), fmt = '%f', encoding = 'utf8')
	print("End")


if __name__ == "__main__":
	main()
