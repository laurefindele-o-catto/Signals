"""
Python Lists vs Dictionaries vs NumPy Arrays
=============================================
A tour of the operations you'll actually use when doing 2D FFT work:
declaring, modifying, iterating, indexing, slicing, reversing,
row<->column conversion, and 2D matrix access.

Run this file top to bottom - each section prints its own output.
"""

import numpy as np

print("=" * 60)
print("1. LISTS (Python's built-in 'array')")
print("=" * 60)

# --- Declaring ---
empty_list = []                    # empty list
also_empty = list()                # same thing, more explicit
numbers = [10, 20, 30, 40, 50]     # list declared with values

print("empty_list      :", empty_list)
print("numbers         :", numbers)

# --- Modifying a declared list ---
numbers.append(60)          # add to the end -> [10,20,30,40,50,60]
numbers.insert(0, 5)        # insert 5 at index 0 -> [5,10,20,30,40,50,60]
numbers[2] = 999            # overwrite the value at index 2
numbers.remove(999)         # remove first occurrence of 999
numbers.pop()                # remove and return the last element
print("after modifying :", numbers)

# --- Accessing an index ---
print("numbers[0]      :", numbers[0])   # first element
print("numbers[-1]     :", numbers[-1])  # last element (negative indexing)

# --- Iterating over a list ---
print("iterating:")
for value in numbers:
    print("  value =", value)

# also with index, using enumerate
for i, value in enumerate(numbers):
    print(f"  index {i} -> {value}")


print()
print("=" * 60)
print("2. DICTIONARIES")
print("=" * 60)

# --- Declaring ---
empty_dict = {}
person = {"name": "Alice", "age": 30, "role": "engineer"}

print("empty_dict      :", empty_dict)
print("person          :", person)

# --- Modifying ---
person["age"] = 31          # update existing key
person["city"] = "Dhaka"    # add a new key
del person["role"]          # remove a key
print("after modifying :", person)

# --- Accessing ---
print("person['name']  :", person["name"])
print("person.get('x') :", person.get("x", "not found"))  # safe access, no crash

# --- Iterating over a dictionary ---
print("iterating keys:")
for key in person:
    print("  key =", key)

print("iterating items (key, value):")
for key, value in person.items():
    print(f"  {key} -> {value}")

print()
print("KEY DIFFERENCE:")
print("  List/array : ordered, accessed by INTEGER INDEX (0,1,2,...)")
print("  Dictionary : accessed by KEY (any hashable type, e.g. strings)")


print()
print("=" * 60)
print("3. SLICING (list[start:stop:step])")
print("=" * 60)

data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print("data            :", data)
print("data[2:5]       :", data[2:5])     # elements at index 2,3,4
print("data[:4]        :", data[:4])      # from start up to (not incl) 4
print("data[6:]        :", data[6:])      # from 6 to the end
print("data[::2]       :", data[::2])     # every 2nd element
print("data[::-1]      :", data[::-1])    # REVERSED list (whole array flipped)

# reversing an array (two common ways)
reversed_list = data[::-1]
reversed_builtin = list(reversed(data))
print("reversed (slice):", reversed_list)
print("reversed (fn)   :", reversed_builtin)


print()
print("=" * 60)
print("4. ROW <-> COLUMN with NumPy (this is what you'll use for FFT)")
print("=" * 60)
# Plain Python lists don't really have a "shape", so for row/column
# conversion and 2D matrix work you use NumPy arrays - the same
# structure numpy.fft.fft2 expects.

row = np.array([1, 2, 3, 4])
print("row (1D)        :", row, " shape:", row.shape)

# Turn a row into a column vector -> shape goes from (4,) to (4,1)
column_a = row.reshape(-1, 1)      # -1 means "figure out this dimension"
column_b = row[:, None]            # equivalent trick using slicing + np.newaxis
print("row.reshape(-1,1) as column:\n", column_a)
print("row[:, None] as column:\n", column_b)

# Turning it back into a row
back_to_row = column_a.reshape(-1)     # or column_a.flatten()
print("back to row     :", back_to_row)


print()
print("=" * 60)
print("5. 2D MATRIX ACCESS (relevant for 2D FFT)")
print("=" * 60)

# A 2D "list of lists" (pure python) vs a numpy 2D array
matrix_list = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
matrix_np = np.array(matrix_list)

print("pure python list-of-lists access -> matrix_list[row][col]")
print("  matrix_list[1][2] =", matrix_list[1][2])   # row 1, col 2 -> 6

print()
print("numpy access -> matrix_np[row, col]  (preferred, faster, used by FFT code)")
print("  matrix_np[1, 2]   =", matrix_np[1, 2])       # same element, cleaner syntax
print("  full matrix:\n", matrix_np)

# Slicing a whole row or column out of a 2D numpy array
print("  row 0    matrix_np[0, :] =", matrix_np[0, :])
print("  col 1    matrix_np[:, 1] =", matrix_np[:, 1])

# Reversing a 2D matrix
print("  flip rows (upside down)   :\n", matrix_np[::-1, :])
print("  flip cols (mirror)        :\n", matrix_np[:, ::-1])
print("  flip both (180 rotation)  :\n", matrix_np[::-1, ::-1])


print()
print("=" * 60)
print("6. WHY THIS MATTERS FOR 2D FFT")
print("=" * 60)
# numpy.fft.fft2 works on a 2D array (rows = one axis, cols = other axis).
# A common technique ("row-column algorithm") is:
#   1) run a 1D FFT on every ROW
#   2) run a 1D FFT on every COLUMN of the result
# That's exactly the row/column slicing and row<->column reshaping shown above.

signal = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
], dtype=float)

# Step 1: FFT each row (iterate over rows using slicing)
row_fft = np.array([np.fft.fft(signal[r, :]) for r in range(signal.shape[0])])

# Step 2: FFT each column of that result (iterate over columns using slicing)
full_fft = np.array([np.fft.fft(row_fft[:, c]) for c in range(row_fft.shape[1])]).T

# Sanity check against numpy's built-in 2D FFT
builtin_fft2 = np.fft.fft2(signal)

print("Manual row-then-column FFT matches np.fft.fft2:",
      np.allclose(full_fft, builtin_fft2))
