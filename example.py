from autograd import Variable

# Example usage:

a = Variable(2)
b = Variable(3)

print(a ** b)    # Variable(data=8)
print(a ** 2)    # Variable(data=4)
print(2 ** a)    # Variable(data=4)

print(a / b)     # Variable(data=0.6666666666666666)
print(a / 2)     # Variable(data=1.0)
print(2 / a)     # Variable(data=1.0)
