from autograd import Variable

# Example 1: Testing addition
x = Variable(2.0)
y = Variable(3.0)
z = x + y
z.grad = 1.0  # Assigning the gradient to the output variable

# Backpropagate the gradients
z._backward()
print(f"Gradients after addition: x.grad = {x.grad}, y.grad = {y.grad}")  # Expected: 1.0, 1.0

# Example 2: Testing multiplication
x = Variable(2.0)
y = Variable(3.0)
z = x * y
z.grad = 1.0  # Assigning the gradient to the output variable

# Backpropagate the gradients
z._backward()
print(f"Gradients after multiplication: x.grad = {x.grad}, y.grad = {y.grad}")  # Expected: 3.0, 2.0

# Example 3: Testing exponentiation
x = Variable(2.0)
z = x ** 2
z.grad = 1.0  # Assigning the gradient to the output variable

# Backpropagate the gradient
z._backward()
print(f"Gradient after exponentiation: x.grad = {x.grad}")  # Expected: 4.0 (2 * 2)

# Example 4: Testing exponentiation with variable exponent
x = Variable(2.0)
y = Variable(3.0)
z = x ** y
z.grad = 1.0  # Assigning the gradient to the output variable

# Backpropagate the gradients
z._backward()
print(f"Gradients after exponentiation with variable exponent: x.grad = {x.grad}, y.grad = {y.grad}")  
# Expected: x.grad = 3.0 * 2^2 = 12.0, y.grad = 2^3 * log(2) ≈ 5.5451