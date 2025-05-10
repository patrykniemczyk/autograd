class Variable:
    
	def __init__(self, data):
		self.data = data

	def __add__(self, other):
		
		if isinstance(other, Variable):
			out = Variable(self.data + other.data)
		else:
			out = Variable(self.data + other)

		return out

	def __mul__(self, other):
		if isinstance(other, Variable):
			out = Variable(self.data * other.data)
		else:
			out = Variable(self.data * other)

		return out

	def __pow__(self, other):
		if isinstance(other, Variable):
			out = Variable(self.data ** other.data)
		else:
			out = Variable(self.data ** other)

		return out

	def __neg__(self):
		return self * -1

	def __sub__(self, other):
		return self + (-other)

	def __truediv__(self, other):
		return self * (other ** -1)

	def __radd__(self, other):
		return self + other

	def __rsub__(self, other):
		return -self + other

	def __rmul__(self, other):
		return self * other

	def __rpow__(self, other):
		return Variable(other) ** self

	def __rtruediv__(self, other):
		return Variable(other) / self

	def __repr__(self):
		return f"Variable(data={self.data})"