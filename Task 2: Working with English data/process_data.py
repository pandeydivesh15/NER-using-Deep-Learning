from keras.preprocessing import sequence
import numpy as np
import pandas as pd

# For getting English word vectors
from get_word_vectors import get_word_vectors


class Get_data():
	"""
	Class for handling all data processing and preparing training/testing data"""

	def self.__init__(datapath):
		# Default values
		self.LEN_NAMED_CLASSES = 5 #4 names and 1 null class
		self.LEN_WORD_VECTORS = 300

		self.tags = []
		# string tags mapped to int and one hot vectors 
		self.tag_id_map = {}
		self.tag_to_one_hot_map = {}

		# All data
		self.x = []
		self.y = []

		self.read_data(datapath)

	def read_data(self, datapath):
		id = 0
		sentence = []
		with open(datapath, 'r') as f:
			for l in f:
				line = line.strip().split()
				if line:
					word, named_tag = line[0], line[3]
					if named_tag not in self.tags:
						self.tags.append(named_tag)
						self.tag_id_map[named_tag] = id

						one_hot_vec = np.zeros(self.LEN_NAMED_CLASSES, dtype = np.int32)
						one_hot_vec[id] = 1
						self.tag_to_one_hot_map[named_tag] = one_hot_vec

						id+=1;
					# Get word vectors for given word	
					


