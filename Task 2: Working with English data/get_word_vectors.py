# Impor  Spacy and create Word Vector Model (GLOVE Model)
import spacy
# The next step takes some time to execute.
NLP = spacy.load("en")

def get_word_vectors(sentence):
	"""
	Returns word vectors for complete sentence / single word
	as a python list"""

	s = NLP(unicode(sentence))
	vec = [ word.vector for word in s ]
	return vec


	