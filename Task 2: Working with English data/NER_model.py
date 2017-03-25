# Keras imports
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dropout
from keras.regularizers import l2
from keras import metrics

import numpy as np


class NER():
	def __init__(self, data_reader):
		self.data_reader = data_reader
		self.x, self.y = data_reader.get_data();
		self.model = None
		self.x_train = None
		self.y_train = None
		self.x_test = None
		self.y_test = None

	def make_and_compile(self, units = 100, dropout = 0.2, regul_alpha = 0.0):
		self.model = Sequential()
		# Bidirectional LSTM with 100 outputs/memory units
		self.model.add(Bidirectional(LSTM(units, 
										  return_sequences=True,
										  W_regularizer=l2(regul_alpha),
										  b_regularizer=l2(regul_alpha)),
									input_shape = [self.data_reader.max_len, 
												   self.data_reader.LEN_WORD_VECTORS]))
		self.model.add(TimeDistributed(Dense(self.data_reader.LEN_NAMED_CLASSES, 
											 activation='softmax',
											 W_regularizer=l2(regul_alpha),
											 b_regularizer=l2(regul_alpha))))
		self.model.add(Dropout(dropout))
		self.model.compile(loss='categorical_crossentropy',
						   optimizer='adam',
						   metrics=['accuracy'])
		print self.model.summary()

	def train(self, train_split = 0.8, epochs = 20, batch_size = 50):
		split_mask = np.random.rand(len(self.x)) < (train_split)
		self.x_train = self.x[split_mask]
		self.y_train = self.y[split_mask]
		self.x_test = self.x[~split_mask]
		self.y_test = self.y[~split_mask]

		self.model.fit(self.x_train, self.y_train, nb_epoch=epochs, batch_size=batch_size)

	def evaluate(self):
		score = self.model.evaluate(self.x_test, self.y_test)
		print score

	def predict_tags(self, sentence):
		sentence_list = sentence.strip().split()
		sent_len = len(sentence_list)
		# Get padded word vectors 
		x = self.data_reader.encode_sentence(sentence)
		tags = self.model.predict(x, batch_size = 1)[0]

		tags = tags[-sent_len:]
		pred_tags = self.data_reader.decode_result(tags)

		for s,tag in zip(sentence_list,pred_tags):
			print s + "/" + tag
