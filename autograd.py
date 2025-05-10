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

	def __neg__(self):
		return self * -1

	def __sub__(self, other):
		return self + (-other)

	def __radd__(self, other):
		return self + other

	def __rsub__(self, other):
		return -self + other

	def __rmul__(self, other):
		return self * other

	def __repr__(self):
		return f"Variable(data={self.data})"

# Example usage:

a = Variable(5)
b = Variable(10)

print(a)         # Variable(data=5)
print(b)         # Variable(data=10)

# Addition
print(a + b)     # Variable(data=15)
print(a + 3)     # Variable(data=8)
print(3 + a)     # Variable(data=8)

# Multiplication
print(a * b)     # Variable(data=50)
print(a * 2)     # Variable(data=10)
print(2 * a)     # Variable(data=10)

# Negation
print(-a)        # Variable(data=-5)

# Subtraction
print(b - a)     # Variable(data=5)
print(a - 2)     # Variable(data=3)
print(20 - a)    # Variable(data=15)
