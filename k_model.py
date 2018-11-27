import keras

def getModel(inputShape):
	L = keras.layers.Input(inputShape)
	inputLayer = L
	L = keras.layers.Conv2D(32, (3, 3), activation = 'relu')(L)
	L = keras.layers.MaxPooling2D(2, 2)(L)
	L = keras.layers.Dropout(0.2)(L)
	L = keras.layers.Conv2D(64, (3, 3), activation = 'relu')(L)
	L = keras.layers.MaxPooling2D(2, 2)(L)
	L = keras.layers.Dropout(0.2)(L)
	L = keras.layers.Flatten()(L)
	L = keras.layers.Dense(128, activation = 'relu')(L)
	L = keras.layers.Dropout(0.2)(L)
	L = keras.layers.Dense(10, activation = 'softmax')(L)
	outputLayer = L
	model = keras.models.Model(inputLayer, outputLayer)
	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
	return model
