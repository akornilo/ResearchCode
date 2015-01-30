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
	N = 100
	beta = 1
	alpha = 0.5


	def add_FVs(self, fv1, fv2):
		self.FVs.append(fv1)
		self.FVs.append(fv2)
		self.Alphas.append(self.alpha)
		self.Alphas.append(-1 * self.alpha)
		self.total = sum([abs(x) for x in self.Alphas])

	def dot_prod(self, fv2):
		total = 0


		if len(self.Alphas) > 0 : 

			# negative half
			odds = range(1, len(self.Alphas), 2)
			d = stats.rv_discrete(values=(odds,
				                  [abs(self.Alphas[x])/(self.total/2.0) for x in odds]))
 			for i in range(min(len(self.FVs)/2, self.N/2)):
				i = d.rvs()
				total-= (self.FVs[i].dot_prod(fv2))


			evens = range(0, len(self.Alphas), 2)
			d = stats.rv_discrete(values=(evens,
				                  [abs(self.Alphas[x])/(self.total/2.0) for x in evens]))
 			for i in range(min(len(self.FVs)/2, self.N/2)):
				i = d.rvs()
				total+= (self.FVs[i].dot_prod(fv2))
			return total

		else:
			return 0
	
	def iterate_alphas(self):
		for i in range(len(self.Alphas)):
			self.Alphas[i]*=self.beta
		self.total = sum([abs(x) for x in self.Alphas])


