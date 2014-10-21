
import json
import sys
import itertools
from random import shuffle

weights = {}

alpha = 0.5

category = "atheistchristian"

labels = [0,1]

#calculates the predicted label for a sentence
def ymax(sent):
	global labels, weights
	maxV = -sys.maxint - 1
	maxLabel = -1

	for l in labels:
		total = 0
		for (f,v) in sent.items():
			if f in weights:
				total += weights[f][l] * int(v)
		if maxV < total:
			maxV = total
			maxLabel = l

	return maxLabel


# gets actual labels for sentences
keyFile = open("20N/"+category+".response")
key = {}

for l in keyFile:
	data = l.rstrip().split("\t")
	key[data[0]] = int(data[1])

# get sentences
train = open("20N/"+category+".train")
allS = [t for t in train]
train.close()
prevMistakes = sys.maxint

for x in range(10):
#while True
	# pick random order for sentences
	shuffle(allS) 
	for sent in allS:
		sent = sent.rstrip().split("\t")
		data = json.loads(sent[1])

		y = key[sent[0]]
		yArg = ymax(data)

		if not (y == yArg):
			for (f,v) in data.items():
				if f not in weights:
					weights[f] = [0,0]
				weights[f][y] += alpha * int(v)
				weights[f][yArg] -= alpha * int(v)	

	# test on dev set - needed if updating based on how much mistakes decrease
	dev = open("20N/"+category+".dev")
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
	if prevMistakes <= mistakes:
		break
	prevMistakes = mistakes
	dev.close()


# look at test set

test = open("20N/"+category+"test.json")

mistakes = 0
total = 0
for sent in test:
	sent = sent.rstrip().split("\t")
	data = json.loads(sent[1])

	y = key[sent[0]]
	yArg = ymax(data)

	if not (y == yArg):
		mistakes += 1
	total+=1
print "results from training set"
print mistakes, total

	
