from random import shuffle
import json


def get_keys(category):
	keyFile = open("../20N/"+category+".response")
	key = {}

	for l in keyFile:
		data = l.rstrip().split("\t")
		key[data[0]] = int(data[1])

	return key

def get_train(category, n):
	train = open("../20N/"+category+".train")
	allS = [t for t in train]
	train.close()

	shuffle(allS)
	initialSet = allS[:n]
	finalSet = []

	for sent in initialSet:
		sent = sent.rstrip().split("\t")
		data = json.loads(sent[1])
		finalSet+=[(sent[0], data)]

	return finalSet

def get_dev(category):
	train = open("../20N/"+category+".dev")
	allS = [t for t in train]
	train.close()
	finalSet = []

	for sent in allS:
		sent = sent.rstrip().split("\t")
		data = json.loads(sent[1])
		finalSet+=[(sent[0], data)]

	return finalSet

