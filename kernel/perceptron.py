from feature import *
import json
import sys
import itertools
from random import shuffle
import json
import matplotlib.pyplot as plt

alpha = 0.5

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
	#print "ymax output:",features, maxLabel, maxV
	return maxLabel



# gets actual labels for sentences
keyFile = open("../20N/"+category+".response")
key = {}

for l in keyFile:
	data = l.rstrip().split("\t")
	key[data[0]] = int(data[1])


# get sentences
train = open("../20N/"+category+".train")
allS = [t for t in train]
train.close()

shuffle(allS)
mySet = allS[:100]

for x in range(3):
#while True
	# pick random order for sentences
	shuffle(mySet) 
	i = 0

	for sent in mySet:
		sent = sent.rstrip().split("\t")
		data = json.loads(sent[1])

		yActual = key[sent[0]]

		yArg = ymax(data)

		#SV.iterate_alphas()

		if yArg != yActual:
			SV.add_FVs(FeatureVector(data, yActual), FeatureVector(data, yArg), alpha)

		# # look at dev set
		dev = open("../20N/"+category+".dev")

		mistakes = 0
		total = 0
		for sent in dev:
			sent = sent.rstrip().split("\t")
			data = json.loads(sent[1])

			y = key[sent[0]]
			yArg = ymax(data)

			if not (y == yArg):
				mistakes += 1
			total+=1
		#print mistakes, total
		error_rate +=[mistakes]
		print "dev",i, mistakes/(total*1.0)


		total = 0
		mistakes=0
		for sent in mySet:
			sent = sent.rstrip().split("\t")
			data = json.loads(sent[1])
			y = key[sent[0]]
			yArg = ymax(data)

			#print "correct label",y,"predict", yArg
			if not (y == yArg):
				mistakes += 1
			total+=1
		print "train", i, mistakes/(total*1.0)
		error_rate2+=[mistakes]

		i+=1


# look at test set

# test = open("../20N/"+category+"test.json")

# mistakes = 0
# total = 0
# for sent in test:
# 	sent = sent.rstrip().split("\t")
# 	data = json.loads(sent[1])

# 	y = key[sent[0]]
# 	yArg = ymax(data)

# 	if not (y == yArg):
# 		mistakes += 1
# 	total+=1
# print "results from training set"
# print mistakes, total

plt.plot(range(len(error_rate)), error_rate)
plt.plot(range(len(error_rate2)), error_rate2)

plt.show()
