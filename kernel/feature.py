from collections import defaultdict

class FeatureVector:

	features = defaultdict(int)
	label = 0

	def __init__(self, features, label):
		self.features = features
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

	def add_FVs(self, fv1, fv2, alpha):
		self.FVs.append(fv1)
		self.FVs.append(fv2)
		self.Alphas.append(alpha)
		self.Alphas.append(-1 * alpha)

	def dot_prod(self, fv2):
		total = 0
		for x in range(len(self.FVs)):
			total+= self.Alphas[x]*(self.FVs[x].dot_prod(fv2))
		return total
