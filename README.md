# matrix-calc
A matrix calculator built in Python.

## Current features
1. Matrix addition, subtraction, multiplication (with '+', '-', '*')
2. RREF
3. Determinant
4. Cofactor
5. Inverse

## Usage

```bash
# Start interactive python session
python3 -i matrix.py
```

```python
# Create Matrix object
a = Matrix([1, 2, 3], [4, 5, 6], [7, 8, 9])
b = Matrix([10, 2, 3, 1], [1, 2, 0, -2], [2, 3, 1, 5])
c = Matrix([2, 1, 0], [2, 10, 4], [1, 2, 0])

print(a)
# ⎡1 2 3⎤
# ⎢4 5 6⎥
# ⎣7 8 9⎦

print(a * a)
# ⎡ 30  36  42⎤
# ⎢ 66  81  96⎥
# ⎣102 126 150⎦

print(a * b)
# ⎡18 15  6 12⎤
# ⎢57 36 18 24⎥
# ⎣96 57 30 36⎦

print(a * c * b + b)
# ⎡143  98  38  -4⎤
# ⎢347 242  92 -10⎥
# ⎣561 387 150  -6⎦

print(c.inv())
# ⎡0.6666666666666666  -0.0 -0.3333333333333333⎤
# ⎢0.3333333333333333  -0.0 -0.6666666666666666⎥
# ⎣               0.5 -0.25                -1.5⎦

c.det()
# -12

print(a.swagchelon())
# ⎡1 0 -1⎤
# ⎢0 1  2⎥
# ⎣0 0  0⎦

# Lists steps of RREF
print(a.swagchelon(True))
# ⎡1.0 2.0 3.0⎤
# ⎢  4   5   6⎥
# ⎣  7   8   9⎦
# ⎡1.0  2.0  3.0⎤
# ⎢0.0 -3.0 -6.0⎥
# ⎣  7    8    9⎦
# ⎡1.0  2.0   3.0⎤
# ⎢0.0 -3.0  -6.0⎥
# ⎣0.0 -6.0 -12.0⎦
# ⎡ 1.0  2.0   3.0⎤
# ⎢-0.0  1.0   2.0⎥
# ⎣ 0.0 -6.0 -12.0⎦
# ⎡ 1.0 2.0 3.0⎤
# ⎢-0.0 1.0 2.0⎥
# ⎣ 0.0 0.0 0.0⎦
# ⎡ 1.0 0.0 -1.0⎤
# ⎢-0.0 1.0  2.0⎥
# ⎣ 0.0 0.0  0.0⎦
# ⎡1 0 -1⎤
# ⎢0 1  2⎥
# ⎣0 0  0⎦
```
