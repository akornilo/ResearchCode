from collections import defaultdict
from random import sample, shuffle
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
	SVs = []
	Alphas = []
	total = 0
	N = 30
	l2reg = 0.02
	alpha = 1

	scale_factor = 1.0


	def update_scalefactor(self, step_size=1):
		# call before add_SVs step!!!!!!!!!!!
		self.scale_factor *= (1-self.l2reg*step_size)

	def add_SVs(self, svActual, svPredicted, step_size=1):

		self.SVs.append(svActual)
		self.SVs.append(svPredicted)

		self.Alphas.append(step_size/self.scale_factor)
		self.Alphas.append(-1 * step_size / self.scale_factor)


	def dot_prod(self, fv2):
		total = 0


		l = min(self.N, len(self.Alphas)/2)


		values = range(0, len(self.Alphas), 2)
		shuffle(values)
		values = values[:l]
		for i in values:
			total += self.Alphas[i]*self.SVs[i].dot_prod(fv2)

		values = range(1, len(self.Alphas), 2)
		shuffle(values)
		values = values[:l]
		for i in values:
			total += self.Alphas[i]*self.SVs[i].dot_prod(fv2)
		
		return total*self.scale_factor

		# if len(self.Alphas) > 0 : 

		# 	# negative half
		# 	odds = range(1, len(self.Alphas), 2)
		# 	d = stats.rv_discrete(values=(odds,
		# 		                  [abs(self.Alphas[x])/(self.total/2.0) for x in odds]))
 	# 		for i in range(min(len(self.FVs)/2, self.N/2)):
		# 		i = d.rvs()
		# 		total-= (self.FVs[i].dot_prod(fv2))


		# 	evens = range(0, len(self.Alphas), 2)
		# 	d = stats.rv_discrete(values=(evens,
		# 		                  [abs(self.Alphas[x])/(self.total/2.0) for x in evens]))
 	# 		for i in range(min(len(self.FVs)/2, self.N/2)):
		# 		i = d.rvs()
		# 		total+= (self.FVs[i].dot_prod(fv2))
		# 	return total

		# else:
		# 	return 0
	
	def iterate_alphas(self):
		for i in range(len(self.Alphas)):
			self.Alphas[i] 
		self.total = sum([abs(x) for x in self.Alphas])


