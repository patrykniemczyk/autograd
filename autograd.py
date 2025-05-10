import math

class Variable:
	
	def __init__(self, data, prev=()):
		self.data = data
		self.grad = 0.0
		self._backward = lambda: None
		self.prev = prev
  
	def zero_grad(self):
		visited = set()
  
		def zero(v):
			if v not in visited:
				visited.add(v)
				v.grad = 0.0
				for p in v.prev:
					zero(p)
     
		zero(self)
   
	def backward(self):
		order = []
		visited = set()
		def topo(v):
			if v not in visited:
				visited.add(v)
				for p in v.prev:
					topo(p)
				order.append(v)
		topo(self)
  
		self.grad = 1.0
		for v in reversed(order):
			v._backward()

	def __add__(self, other):
		
		if isinstance(other, Variable):
			out = Variable(self.data + other.data, prev=(self, other))
		else:
			out = Variable(self.data + other, prev=(self,))
   
		def _backward():
			self.grad += out.grad
			if isinstance(other, Variable):
				other.grad += out.grad

		out._backward = _backward
		return out

	def __mul__(self, other):
		if isinstance(other, Variable):
			out = Variable(self.data * other.data, prev=(self, other))
		else:
			out = Variable(self.data * other, prev=(self,))
   
		def _backward():
			if isinstance(other, Variable):
				self.grad += out.grad * other.data
				other.grad += out.grad * self.data
			else:
				self.grad += out.grad * other
	
		out._backward = _backward
		return out

	def __pow__(self, other):
		if isinstance(other, Variable):
			out = Variable(self.data ** other.data, prev=(self, other))
		else:
			out = Variable(self.data ** other, prev=(self,))
	
		def _backward():
			if isinstance(other, Variable):
				self.grad += out.grad * other.data * self.data ** (other.data - 1)
				other.grad += out.grad * self.data ** other.data * math.log(self.data)
			else:
				self.grad += out.grad * other * self.data ** (other - 1)

		out._backward = _backward
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
		return f"Variable(data={self.data}, grad={self.grad})"