import numpy as np
import k_model
from copy import deepcopy


class Creature:

	def __init__(self, inputShape, weights, prob = None, **kwargs):
		self.inputShape = deepcopy(inputShape)
		self.model = k_model.getModel(inputShape)
		self.weights = deepcopy(weights)
		self.model.set_weights(self.weights)
		if prob is None:
			self.prob = self.createProbabilityMatrix()
		else:
			self.prob = prob
		self.initalWeightCount = sum([np.size(x) for x in self.prob])
		self.accuracy = 0
		self.efficiency = 0
		self.fitness = 0
		super(Creature, self).__init__(**kwargs)

	def createProbabilityMatrix(self):
		prob = []
		weights = self.model.get_weights()
		for i in range(0, len(weights), 2):
			weightShape = weights[i].shape
			prob.append(np.random.uniform(0.0, 1.0, weightShape))
		return prob

	def countZeroWeights(self):
		weights = self.model.get_weights()
		count = 0
		for i in range(0, len(weights), 2):
			count += (weights[i] == 0).sum()
		return count

	def evaluateFitness(self, x, y):
		accuracy = self.model.evaluate(x, y, verbose = 0)[1]
		efficiency = self.countZeroWeights()/self.initalWeightCount
		self.accuracy = accuracy
		self.efficiency = 1/(1-efficiency)
		self.fitness = accuracy
		return self.fitness

	def getFitness(self):
		return [self.fitness, self.accuracy, self.efficiency]

	def pruneWeights(self):
		weights = self.model.get_weights()
		for i in range(0, len(weights), 2):
			result = np.round(self.prob[int(i/2)])
			weights[i] = weights[i] * result
		self.model.set_weights(weights)

	def mutate(self, mutationRate):
		for i in range(0, len(self.prob)):
			weightShape = self.prob[i].shape
			mutationDiceRoll = (np.random.uniform(0.0, 1.0, weightShape) > mutationRate).astype(int)
			mutatedValues = np.random.uniform(0.0, 1.0, weightShape)
			matedValues = self.prob[i] * mutationDiceRoll
			matedValues = matedValues + ((mutationDiceRoll == 0).astype(int) * mutatedValues)
			self.prob[i] = matedValues

	def crossover(self, mate, mutationRate):
		crossoverMidpoint = 0.5
		newProbabilities = []
		for i in range(0, len(self.prob)):
			weightShape = self.prob[i].shape
			crossoverDiceRoll = np.random.uniform(0.0, 1.0, weightShape)
			valuesFromSelf = (crossoverDiceRoll < crossoverMidpoint).astype(int) * deepcopy(self.prob[i])
			valuesFromMate = (crossoverDiceRoll > crossoverMidpoint).astype(int) * deepcopy(mate.prob[i])
			matedValues = valuesFromMate + valuesFromSelf
			newProbabilities.append(matedValues)
		newCreature = Creature(self.inputShape, self.weights, prob = newProbabilities)
		newCreature.mutate(mutationRate)
		return newCreature

	def __repr__(self):
		return str(self.fitness)