import tensorflow as tf
import numpy as np
from model import Model


def getData():
	from tensorflow.examples.tutorials.mnist import input_data
	return input_data.read_data_sets('MNIST_data', one_hot = True, reshape = False)


def plot_test_acc(plot_handles):
	import matplotlib.pyplot as plt
	from IPython import display
	plt.legend(handles = plot_handles, loc = "center right")
	plt.xlabel("Iterations")
	plt.ylabel("Test Accuracy")
	plt.ylim(0, 1)
	display.display(plt.gcf())
	display.clear_output(wait = True)


def train_task(model, num_iter, disp_freq, trainX, trainY, testsX, testsY, x, y_, sess, lams = [0], batchSize = 64):
	import matplotlib.pyplot as plt
	for l in range(len(lams)):
		# lams[l] sets weight on old task(s)
		model.restore(sess)  # reassign optimal weights from previous training session
		if (lams[l] == 0):
			model.set_vanilla_loss()
		else:
			model.update_ewc_loss(lams[l])
		# initialize test accuracy array for each task
		test_accs = []
		for task in range(len(testsY)):
			test_accs.append(np.zeros(int(num_iter / disp_freq)))
		# train on current task
		iter = 0
		for offset in range(0, len(trainY), batchSize):
			end = offset + batchSize
			batch_x, batch_y = trainX[offset:end], trainY[offset:end]
			model.train_step.run(feed_dict = {x: batch_x, y_: batch_y})
			if iter % disp_freq == 0:
				plt.subplot(1, len(lams), l + 1)
				plots = []
				colors = ['r', 'b', 'g']
				for task in range(len(testsY)):
					feed_dict = {x: testsX[task], y_: testsY[task]}
					print('Accuracy: ' + str(model.accuracy.eval(feed_dict = feed_dict)))
					# test_accs[task][int(offset / disp_freq)] = model.accuracy.eval(feed_dict = feed_dict)
					# c = chr(ord('A') + task)
				# 	plot_h, = plt.plot(range(1, iter + 2, disp_freq), test_accs[task][:int(iter / disp_freq) + 1],
				# 	                   colors[task],
				# 	                   label = "task " + c)
				# 	plots.append(plot_h)
				# plot_test_acc(plots)
				if l == 0:
					plt.title("vanilla sgd")
				else:
					plt.title("ewc")
				plt.gcf().set_size_inches(len(lams) * 5, 3.5)
			iter += 1


def mnist_test():
	mnist = getData()
	sess = tf.InteractiveSession()

	X_train, y_train = mnist.train.images, mnist.train.labels
	X_validation, y_validation = mnist.validation.images, mnist.validation.labels
	X_test, y_test = mnist.test.images, mnist.test.labels

	# Pad images with 0s
	X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
	X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
	X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

	assert (len(X_train) == len(y_train))
	assert (len(X_validation) == len(y_validation))
	assert (len(X_test) == len(y_test))

	x = tf.placeholder(tf.float32, shape = [None, 32, 32, 1])
	y_ = tf.placeholder(tf.float32, shape = [None, 10])

	model = Model(x, y_)
	# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(model, one_hot_y)
	# loss_operation = tf.reduce_mean(cross_entropy)
	# optimizer = tf.train.AdamOptimizer(learning_rate = rate)
	# training_operation = optimizer.minimize(loss_operation)

	sess.run(tf.global_variables_initializer())

	train_task(model, 800, 20, X_train, y_train, [X_test], [y_test], x, y_, sess, lams = [0])
	model.compute_fisher(X_test, sess, num_samples = 200, plot_diffs = True)
	fisher = model.F_accum


def main():
	t1 = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
	test = t1.values
	print("End")


if __name__ == "__main__":
	main()
