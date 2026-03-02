import numpy as np

# # # from a list
# a = np.array([1, 2, 3, 4, 5, 6, 7])

# # # from zeros
# # b = np.zeros((2, 2))
# # print("Array from zeros:", b)

# # # from ones
# # c = np.ones((2, 2))
# # print("Array from ones:", c)

# # Identity matrix - eye returns a square matrix with ones on the main diagonal and zeros elsewhere
# d = np.eye(4)
# print("Identity matrix:", d)

# # from a range
# e = np.arange(10)
# print("Array from range:", e)

# # from a range with a step - (start, stop, step)
# f = np.arange(0, 10, 2)
# print("Array from range with step:", f)

# # random values
# f = np.random.rand(2, 2)
# print("Array from random values:", f)

# # random integers
# g = np.random.randint(0, 10, (4, 4))
# print("Array from random integers:", g)

# # evenly spaced values
# h = np.linspace(0, 1, 5)
# print("evenly spaced values:", h)

# Create a 2x3 array
# array_2d = np.array([[1, 2, 3], [4, 5, 6]])
# print("Array from 2x3 array:", array_2d)

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])

# c = a + b
# print("Array from addition:", c)

# # Element-wise subtraction
# d = a - b
# print("Array from subtraction:", d)

# # Element-wise multiplication
# e = a * b
# print("Array from multiplication:", e)

# # Element-wise division
# f = a / b
# print("Array from division:", f)

# Indexing and Slicing

# arr_1d = np.array([1, 2, 3, 4, 5])
# print("Array from 1d array:", arr_1d)

# # Indexing
# print("Element at index 2:", arr_1d[2])

# # Slicing
# print("Elements from index 1 to 3:", arr_1d[1:4])

# arr_2D = np.array([[1, 2, 3, 6, 7, 8, 6], [4, 5, 6, 7, 3, 5, 5]])
# print("Array from 2D array:", arr_2D)

# # # Indexing and Slicing for 2D array
# # print("Element at index 1, 2:", arr_2D[1, 2])

# # Slicing row for 2D array
# print("Elements from index 0 to 1, 0 to 2:", arr_2D[0:6, 0:4])

# # # # Slicing columns
# # # print("Elements from index 0 to 1, 0 to 2:", arr_2D[0:2, 1])

# # # Combining indexing and slicing in 2D array
# # print(arr_2D[1, 0:2])
#
# a = np.arange(1, 13)

# reshaped_a = a.reshape(4, 3)
# print("Reshaped array:\n", reshaped_a)


# Universal Functions
# arr = np.array([1, 4, 9, 16])

# # Square root
# sqrt_arr = np.sqrt(arr)
# print("Square root array:", sqrt_arr)

# # Rounding to the nearest integer
# rounded_arr = np.round([1.3, 2.7, 4.1])
# print("Rounded:", rounded_arr)

# # Absolute value abs_arr
# abs_arr = np.abs([-1, -2, -3])
# print("Absolute value:", abs_arr)

# # Trigonometric functions

# sin_arr = np.sin(arr)
# print("Sine array:", sin_arr)

# cos_arr = np.cos(arr)
# print("Cosine array:", cos_arr)

# tan_arr = np.tan(arr)
# print("Tangent array:", tan_arr)

# # Logarithmic functions

# log_arr = np.log(arr)
# print("Logarithmic array:", log_arr)

# log10_arr = np.log10(arr)
# print("Logarithmic base 10 array:", log10_arr)

# # Exponential functions

# exp_arr = np.exp(arr)
# print("Exponential array:", exp_arr)

# exp2_arr = np.exp2(arr)
# print("Exponential base 2 array:", exp2_arr)

# power_arr = np.power(arr, 2)
# print("Power array:", power_arr)

# # mean
# mean_arr = np.mean(arr)
# print("Mean array:", mean_arr)


# Masking and Boolean Indexing
arr = np.array([1, 4, 8, 9, 16])

mask = arr > 5
print("Masked array:", arr[mask])

mask = arr % 2 == 0
print("Masked array:", arr[mask])

mask = arr % 2 != 0
print("Masked array:", arr[mask])

# Broadcasting with a scalar
arr = np.array([1, 4, 8, 9, 16])

result = arr * 2
print("Broadcasted array:", result)

# Broadcasting with different shaped arrays
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
c = np.array([1, 2, 3])
result = b + c
print("Broadcasting with different shaped arrays:", result)

# Broadcasting a column vector across a 2D array
d = np.array([[1], [2], [3]])
result = b + d
print("Broadcasting a column vector across a 2D array:", result)
