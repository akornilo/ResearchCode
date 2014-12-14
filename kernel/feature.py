from collections import defaultdict
from random import sample
from scipy import stats


class FeatureVector:

	features = defaultdict(int)
	label = 0

	def __init__(self, features, label):
		self.features = features
		self.features["BIAS"] = 1
		self.label = label

	def dot_prod(self, fv2):

        # kernels

		total = 0
		if self.label != fv2.label:
			return 0
		for x in self.features:
			if x in fv2.features:			
				total+=self.features[x]*fv2.features[x]
		return total

	def add_fv(self, fv2):
		if self.label != fv2.label:
			return
		for x in fv2.features:
			self.features[x]+=fv2.features[x]


class SupportVector:
	FVs = []
	Alphas = []
	total = 0
	N = 50
	beta = 0.5

	def add_FVs(self, fv1, fv2, alpha):
		self.FVs.append(fv1)
		self.FVs.append(fv2)
		self.Alphas.append(alpha)
		self.Alphas.append(-1 * alpha)
		total = sum([abs(x) for x in self.Alphas])

	def dot_prod(self, fv2):
		total = 0

		# stats.rv_discrete(values=(range(len(self.Alphas),[abs(x) for x in self.Alphas])))


		#if (len(self.FVs) > self.N):
		#	subset = sample(xrange(len(self.FVs)), self.N)
		#else:
		subset = range(len(self.FVs))

		for x in subset:
			total+= self.Alphas[x]*(self.FVs[x].dot_prod(fv2))
		return total

	def iterate_alphas(self):
		for i in range(len(self.Alphas)):
			self.Alphas[i]-=self.Alphas[i]*self.beta
		total = sum([abs(x) for x in self.Alphas])


