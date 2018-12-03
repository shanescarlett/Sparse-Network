import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display


# variable initialization functions
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)


class Model:
	def __init__(self, x, y_):

		in_dim = int(x.get_shape()[1])  # 784 for MNIST
		out_dim = int(y_.get_shape()[1])  # 10 for MNIST

		self.x = x  # input placeholder

		mu = 0
		sigma = 0.1
		layer_depth = {
			'layer_1': 6,
			'layer_2': 16,
			'layer_3': 120,
			'layer_f1': 84
		}

		# TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
		conv1_w = tf.Variable(tf.truncated_normal(shape = [5, 5, 1, 6], mean = mu, stddev = sigma))
		conv1_b = tf.Variable(tf.zeros(6))
		conv1 = tf.nn.conv2d(x, conv1_w, strides = [1, 1, 1, 1], padding = 'VALID') + conv1_b
		# TODO: Activation.
		conv1 = tf.nn.relu(conv1)

		# TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
		pool_1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

		# TODO: Layer 2: Convolutional. Output = 10x10x16.
		conv2_w = tf.Variable(tf.truncated_normal(shape = [5, 5, 6, 16], mean = mu, stddev = sigma))
		conv2_b = tf.Variable(tf.zeros(16))
		conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1, 1, 1, 1], padding = 'VALID') + conv2_b
		# TODO: Activation.
		conv2 = tf.nn.relu(conv2)

		# TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
		pool_2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

		# TODO: Flatten. Input = 5x5x16. Output = 400.
		fc1 = tf.contrib.layers.flatten(pool_2)

		# TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
		fc1_w = tf.Variable(tf.truncated_normal(shape = (400, 120), mean = mu, stddev = sigma))
		fc1_b = tf.Variable(tf.zeros(120))
		fc1 = tf.matmul(fc1, fc1_w) + fc1_b

		# TODO: Activation.
		fc1 = tf.nn.relu(fc1)

		# TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
		fc2_w = tf.Variable(tf.truncated_normal(shape = (120, 84), mean = mu, stddev = sigma))
		fc2_b = tf.Variable(tf.zeros(84))
		fc2 = tf.matmul(fc1, fc2_w) + fc2_b
		# TODO: Activation.
		fc2 = tf.nn.relu(fc2)

		# TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
		fc3_w = tf.Variable(tf.truncated_normal(shape = (84, 10), mean = mu, stddev = sigma))
		fc3_b = tf.Variable(tf.zeros(10))
		self.y = tf.matmul(fc2, fc3_w) + fc3_b

		# simple 2-layer network
		# W1 = weight_variable([in_dim, 50])
		# b1 = bias_variable([50])
		#
		# W2 = weight_variable([50, out_dim])
		# b2 = bias_variable([out_dim])
		#
		# h1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # hidden layer
		# self.y = tf.matmul(h1, W2) + b2  # output layer

		self.var_list = [conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_b, fc3_w]

		# vanilla single-task loss
		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = self.y))
		self.set_vanilla_loss()

		# performance metrics
		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	def compute_fisher(self, imgset, sess, num_samples = 200, plot_diffs = False, disp_freq = 10):
		# computer Fisher information for each parameter

		# initialize Fisher information for most recent task
		self.F_accum = []
		for v in range(len(self.var_list)):
			self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

		# sampling a random class from softmax
		probs = tf.nn.softmax(self.y)
		class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

		if (plot_diffs):
			# track differences in mean Fisher info
			F_prev = deepcopy(self.F_accum)
			mean_diffs = np.zeros(0)

		for i in range(num_samples):
			# select random input image
			im_ind = np.random.randint(imgset.shape[0])
			# compute first-order derivatives
			ders = sess.run(tf.gradients(tf.log(probs[0, class_ind]), self.var_list),
			                feed_dict = {self.x: imgset[im_ind:im_ind + 1]})
			# square the derivatives and add to total
			for v in range(len(self.F_accum)):
				self.F_accum[v] += np.square(ders[v])
			if (plot_diffs):
				if i % disp_freq == 0 and i > 0:
					# recording mean diffs of F
					F_diff = 0
					for v in range(len(self.F_accum)):
						F_diff += np.sum(np.absolute(self.F_accum[v] / (i + 1) - F_prev[v]))
					mean_diff = np.mean(F_diff)
					mean_diffs = np.append(mean_diffs, mean_diff)
					for v in range(len(self.F_accum)):
						F_prev[v] = self.F_accum[v] / (i + 1)
					plt.plot(range(disp_freq + 1, i + 2, disp_freq), mean_diffs)
					plt.xlabel("Number of samples")
					plt.ylabel("Mean absolute Fisher difference")

					display.display(plt.gcf())
					display.clear_output(wait = True)

		plt.show()
		# divide totals by number of samples
		for v in range(len(self.F_accum)):
			self.F_accum[v] /= num_samples

	def star(self):
		# used for saving optimal weights after most recent task training
		self.star_vars = []

		for v in range(len(self.var_list)):
			self.star_vars.append(self.var_list[v].eval())

	def restore(self, sess):
		# reassign optimal weights for latest task
		if hasattr(self, "star_vars"):
			for v in range(len(self.var_list)):
				sess.run(self.var_list[v].assign(self.star_vars[v]))

	def set_vanilla_loss(self):
		self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)

	def update_ewc_loss(self, lam):
		# elastic weight consolidation
		# lam is weighting for previous task(s) constraints

		if not hasattr(self, "ewc_loss"):
			self.ewc_loss = self.cross_entropy

		for v in range(len(self.var_list)):
			self.ewc_loss += (lam / 2) * tf.reduce_sum(
				tf.multiply(self.F_accum[v].astype(np.float32), tf.square(self.var_list[v] - self.star_vars[v])))
		self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.ewc_loss)
