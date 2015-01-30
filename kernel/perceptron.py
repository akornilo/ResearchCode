from feature import *
import json
import sys
import itertools
from random import shuffle
import json
import matplotlib.pyplot as plt
from io import *


category = "atheistchristian"

labels = [0,1]

SV = SupportVector()

error_rate = []
error_rate2 = []

def ymax(features):
	global labels, SV
	maxV = float(-sys.maxint - 1)
	maxLabel = -1

	for l in labels:
		total = SV.dot_prod(FeatureVector(features, l))
		if maxV < total:
			maxV = total
			maxLabel = l
	return maxLabel

def ymaxCA(features, yi):
	global labels, SV
	maxV = float(-sys.maxint - 1)
	maxLabel = -1
	cost = 100

	for l in labels:
		total = SV.dot_prod(FeatureVector(features, l))
		if l != yi:
			total+=cost
		if maxV < total:
			maxV = total
			maxLabel = l
	return maxLabel


def test_sent(data, keys):
	total = 0
	mistakes = 0
	for sent in data:
		y = keys[sent[0]]
		yArg = ymax(sent[1])
		if not (y == yArg):
			mistakes += 1
		total+=1
	return (mistakes, total)



keys = get_keys(category)
train = get_train(category, 50)
dev = get_dev(category)

i = 0
for x in range(3):

	shuffle(train) 
	
	for sent in train:

		yActual = keys[sent[0]]

		yArg = ymaxCA(sent[1], yActual)

		# if we guessed wrong, add vectors
		if yArg != yActual:
			SV.add_FVs(FeatureVector(sent[1], yActual), FeatureVector(sent[1], yArg))

		#SV.iterate_alphas()

		(mistakes, total) = test_sent(train, keys)
		error_rate +=[mistakes/(total*1.0)]
		print "train",i, mistakes/(total*1.0)

		(mistakes, total) = test_sent(dev, keys)
		print "dev", i, mistakes/(total*1.0)
		error_rate2+=[mistakes/(total*1.0)]
		i+=1

plt.plot(range(len(error_rate)), error_rate)
plt.plot(range(len(error_rate2)), error_rate2)

plt.show()
