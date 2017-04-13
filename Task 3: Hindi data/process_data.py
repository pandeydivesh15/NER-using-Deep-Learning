import numpy as np
from keras.preprocessing import sequence
# For getting English word vectors
from get_word_vectors import get_word_vector, get_sentence_vectors
import codecs


class DataHandler():
	"""
	Class for handling all data processing and preparing training/testing data"""

	def __init__(self, datapath):
		# Default values
		self.LEN_NAMED_CLASSES = 12 # 4 names and 1 null class
		self.NULL_CLASS = "O"
		self.LEN_WORD_VECTORS = 50

		self.tags = []
		# string tags mapped to int and one hot vectors 
		self.tag_id_map = {}
		self.tag_to_one_hot_map = {}

		# All data(to be filled by read_data method)
		self.x = []
		self.y = []

		self.read_data(datapath)

	def read_data(self, datapath):
		_id = 0
		sentence = []
		sentence_tags = []
		all_data = []
		pos = 0
		with codecs.open(datapath, 'r') as f:
			for l in f:
				if pos > 100000:
					break;
				pos+=1
				line = l.strip().split()
				if line:
					try:
						word, named_tag = line[0], line[1]
					except:
						continue

					if named_tag not in self.tags:
						self.tags.append(named_tag)
						self.tag_id_map[_id] = named_tag
						one_hot_vec = np.zeros(self.LEN_NAMED_CLASSES, dtype = np.int32)
						one_hot_vec[_id] = 1
						self.tag_to_one_hot_map[named_tag] = one_hot_vec

						_id+=1;

					# Get word vectors for given word	
					sentence.append(get_word_vector(word)[:self.LEN_WORD_VECTORS])
					sentence_tags.append(self.tag_to_one_hot_map[named_tag])
				else:
					all_data.append( (sentence, sentence_tags) );
					sentence_tags = []
					sentence = []

		#Find length of largest sentence
		self.max_len = 0
		for pair in all_data:
			if self.max_len < len(pair[0]):
				self.max_len = len(pair[0])

		for vectors, one_hot_tags in all_data:
			# Pad the sequences and make them all of same length
			temp_X = np.zeros(self.LEN_WORD_VECTORS, dtype = np.int32)
			temp_Y = np.array(self.tag_to_one_hot_map[self.NULL_CLASS])
			pad_length = self.max_len - len(vectors)

			#Insert into main data list
			self.x.append( ((pad_length)*[temp_X]) + vectors)
			self.y.append( ((pad_length)*[temp_Y]) + one_hot_tags)

		self.x = np.array(self.x)
		self.y = np.array(self.y)

	def get_data(self):
		# Returns proper data for training/testing
		return (self.x, self.y)

	def encode_sentence(self, sentence):
		vectors = get_sentence_vectors(sentence)
		vectors = [v[:self.LEN_WORD_VECTORS] for v in vectors]
		return sequence.pad_sequences([vectors], maxlen=self.max_len, dtype=np.float32)

	def decode_result(self, result_sequence):
		pred_named_tags = []
		for pred in result_sequence:
			_id = np.argmax(pred)
			pred_named_tags.append(self.tag_id_map[_id])
		return pred_named_tags






