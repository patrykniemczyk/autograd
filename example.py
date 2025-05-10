from autograd import Variable

# Example 1: Testing a chain of operations (addition and multiplication)
x = Variable(1.0)
y = Variable(2.0)
z = Variable(3.0)

# z = (x + y) * z
result = (x + y) * z
result.grad = 1.0  # Assigning the gradient to the output variable

# Backpropagate the gradients
result.backward()
print(f"Gradients after chain of addition and multiplication: x.grad = {x.grad}, y.grad = {y.grad}, z.grad = {z.grad}")  
# Expected: x.grad = 3.0, y.grad = 3.0, z.grad = 3.0 (because z is multiplied by (x + y) which equals 3)

# Example 2: Testing exponentiation and multiplication in a chain
x = Variable(2.0)
y = Variable(3.0)
z = Variable(4.0)

# z = (x ** y) * y
result = (x ** y) * y
result.grad = 1.0  # Assigning the gradient to the output variable

# Backpropagate the gradients
result.backward()
print(f"Gradients after exponentiation and multiplication: x.grad = {x.grad}, y.grad = {y.grad}, z.grad = {z.grad}")  
# Expected: 
# x.grad = y * y * x^(y-1) = 3 * 3 * 2^2 = 3 * 3 * 4 = 36.0
# y.grad = x^y * (1 + y * ln(x)) = 2^3 * (1 + 3 * ln(2)) = 8 * (1 + 3 * 0.693) = 8 * 3.079 ≈ 24.635
# z.grad = 0.0 (z does not affect the result)

# Example 3: Nested operations (exponentiation and addition)
x = Variable(2.0)
y = Variable(1.0)
z = Variable(3.0)

# z = (x ** y) + (x + y)
result = (x ** y) + (x + y)
result.grad = 1.0  # Assigning the gradient to the output variable

# Backpropagate the gradients
result.backward()
print(f"Gradients after nested exponentiation and addition: x.grad = {x.grad}, y.grad = {y.grad}, z.grad = {z.grad}")  
# Expected:
# x.grad = y * x^(y-1) + 1 = 1 * 2^0 + 1 = 1 + 1 = 2.0
# y.grad = x^y * ln(x) + 1 = 2^1 * ln(2) + 1 = 2 * 0.693 + 1 ≈ 2.386
# z.grad = 0.0 (z does not affect the result)

# Example 4: A more complex chain with division and subtraction
x = Variable(5.0)
y = Variable(3.0)
z = Variable(2.0)

# z = (x - y) / (x + y)
result = (x - y) / (x + y)
result.grad = 1.0  # Assigning the gradient to the output variable

# Backpropagate the gradients
result.backward()
print(f"Gradients after division and subtraction: x.grad = {x.grad}, y.grad = {y.grad}, z.grad = {z.grad}")  
# Expected:
# x.grad = (1 * (x + y) - (x - y) * 1) / (x + y)^2 = (8 - 2) / 64 = 6 / 64 = 0.09375
# y.grad = (-1 * (x + y) - (x - y) * 1) / (x + y)^2 = (-8 - 2) / 64 = -10 / 64 = -0.15625
# z.grad = 0.0 (z does not affect the result)

# Example 5: Using a combination of power and multiplication in a deep chain
x = Variable(2.0)
y = Variable(3.0)

# z = (x ** 2) * (y ** 3)
result = (x ** 2) * (y ** 3)
result.grad = 1.0  # Assigning the gradient to the output variable

# Backpropagate the gradients
result.backward()
print(f"Gradients after deep chain (power and multiplication): x.grad = {x.grad}, y.grad = {y.grad}")  
# Expected:
# x.grad = 2 * x * (y^3) = 2 * 2 * 27 = 4 * 27 = 108.0
# y.grad = 3 * (y^2) * (x^2) = 3 * 9 * 4 = 27 * 4 = 108.0

# Test the zero_grad method in the Variable class

# Example 1: Test zero_grad on a simple addition
x = Variable(2.0)
y = Variable(3.0)
z = x + y
z.grad = 1.0  # Assigning the gradient to the output variable

# Backpropagate the gradients
z.backward()
print(f"Before zero_grad: x.grad = {x.grad}, y.grad = {y.grad}")  
# Expected: x.grad = 1.0, y.grad = 1.0

# Zero the gradients
z.zero_grad()

# Check if gradients have been reset to zero
print(f"After zero_grad: x.grad = {x.grad}, y.grad = {y.grad}")  
# Expected: x.grad = 0.0, y.grad = 0.0

# Example 2: Test zero_grad after more complex chain
x = Variable(2.0)
y = Variable(3.0)
z = Variable(4.0)

# z = (x + y) * z
result = (x + y) * z
result.grad = 1.0  # Assigning the gradient to the output variable

# Backpropagate the gradients
result.backward()
print(f"Before zero_grad: x.grad = {x.grad}, y.grad = {y.grad}, z.grad = {z.grad}")  
# Expected: x.grad = 4.0, y.grad = 4.0, z.grad = 5.0 (since z is multiplied by (x + y) = 5)

# Zero the gradients
result.zero_grad()

# Check if gradients have been reset to zero
print(f"After zero_grad: x.grad = {x.grad}, y.grad = {y.grad}, z.grad = {z.grad}")  
# Expected: x.grad = 0.0, y.grad = 0.0, z.grad = 0.0

# Example 3: Test zero_grad with more complex operations
x = Variable(2.0)
y = Variable(3.0)

# z = (x ** 2) * (y ** 3)
result = (x ** 2) * (y ** 3)
result.grad = 1.0  # Assigning the gradient to the output variable

# Backpropagate the gradients
result.backward()
print(f"Before zero_grad: x.grad = {x.grad}, y.grad = {y.grad}")  
# Expected: x.grad = 108.0, y.grad = 108.0

# Zero the gradients
result.zero_grad()

# Check if gradients have been reset to zero
print(f"After zero_grad: x.grad = {x.grad}, y.grad = {y.grad}")  
# Expected: x.grad = 0.0, y.grad = 0.0