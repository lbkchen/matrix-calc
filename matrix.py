from operator import add, sub, mul
num_add = add
num_mul = mul
#########################
# OBJECTIVE ABSTRACTION #
#########################
class Matrix(object):
    def __init__(self, *rows):
        assert all(len(rows[0]) == len(r) for r in rows), "Malformed matrix."
        self.lists = [r for r in rows]

    def __str__(self):
        max_char_in_col = [max([len(str(n)) for n in self.n_col(i)]) for i in range(self.cols)]
        result = []
        for i in range(self.rows):
            this_row = " ".join([" " * (max_char_in_col[j] - len(str(self.n_row(i)[j])))
                        + str(self.n_row(i)[j]) for j in range(self.cols)])
            if self.rows == 1:
                result.append("[" + this_row + "]")
            elif i == 0:
                result.append("⎡" + this_row + "⎤")
            elif i == self.rows - 1:
                result.append("⎣" + this_row + "⎦")
            else:
                result.append("⎢" + this_row + "⎥")
            result.append("\n")
        result.pop()
        return "".join(result)

    def __repr__(self):
        return "Matrix(*" + str(self.lists) + ")"

    @property
    def rows(self):
        return len(self.lists)

    @property
    def cols(self):
        return len(self.lists[0])

    def n_row(self, n):
        """Returns a list that is 0-indexed n-th row of the given matrix."""
        return self.lists[n]

    def n_col(self, n):
        """Returns a list that is 0-indexed n-th col of the given matrix."""
        return [row[n] for row in self.lists]

    def rep_row(self, n, new):
        """Replaces row n of the matrix by the new row."""
        assert len(new) == self.cols, "Cannot replace row. Bad dimensions."
        self.lists = [new if i == n else self.n_row(i) for i in range(self.rows)]

    def rep_col(self, n, new):
        """Replaces col n of the matrix by the new col.""" #FIXME not efficient cause of multiple calls to self.n_row(i)
        assert len(new) == self.rows, "Cannot replace row. Bad dimensions."
        self.lists = [[new[i] if j == n else self.n_row(i)[j] for j in range(self.cols)] for i in range(self.rows)]

    def add(self, m):
        assert self.rows == m.rows and self.cols == m.cols, "Cannot add. Bad dimsensions."
        return Matrix(*[[add(x, y) for x, y in zip(self.n_row(i), m.n_row(i))] for i in range(self.rows)])

    def __add__(self, m):
        return self.add(m)

    def sub(self, m):
        return self.add(m.kmul(-1))

    def __sub__(self, m):
        return self.sub(m)

    def mul(self, m):
        if type(m) == int or type(m) == float:
            return self.kmul(m)
        else:
            assert self.cols == m.rows, "Cannot multiply. Bad dimensions."
            return Matrix(*[[sum([mul(x, y) for x, y in zip(self.n_row(i), m.n_col(j))])
                            for j in range(m.cols)] for i in range(self.rows)])

    def __mul__(self, m):
        return self.mul(m)

    def __rmul__(self, m):
        return self.mul(m)

    def kmul(self, k):
        """Returns a new matrix, which is the old one multiplied by a constant k."""
        return Matrix(*[[k * x for x in self.n_row(i)] for i in range(self.rows)])

    #############################
    # ELEMENTARY ROW OPERATIONS #
    #############################

    def scale_row(self, row, k):
        """Returns a new matrix, with row #row scaled by k."""
        return Matrix(*[[k * entry for entry in self.n_row(i)]
                if row == i else self.n_row(i) for i in range(self.rows)])

    def swap_rows(self, row1, row2):
        """Returns a new matrix, with row1 and row2 swapped"""
        new_matrix = Matrix(*self.lists[:])
        new_matrix.lists[row1], new_matrix.lists[row2] = new_matrix.n_row(row2), new_matrix.n_row(row1) #FIXME Abstraction violation whoops
        return new_matrix

    def rep_mul(self, row_replaced, other_row, k_rep, k_other):
        """Returns a new matrix, with row_replaced replaced with a linear
        combination of itself and other_row with weights given by k_rep and k_other.
        """
        new_matrix = Matrix(*self.lists[:])
        new_matrix.lists[row_replaced] = [x + y for x, y in zip(self.scale_row(row_replaced, k_rep).n_row(row_replaced),
                                                                self.scale_row(other_row, k_other).n_row(other_row))]
        return new_matrix

    ###################
    # SWAG OPERATIONS #
    ###################
    def swagchelon(self, steps=False):
        """Returns the rref of matrix, and sets an instance attribute echelon to
        be equal to the result."""
        i, j = 0, 0
        echelon = Matrix(*self.lists[:])
        while i < echelon.rows and j < echelon.cols:

            # 1. If aij = 0 swap the i-th row with some other row below to guarantee that aij != 0.
            # The non-zero entry in the (i, j)-position is called a pivot. If all entries in the column
            # are zero, increase j by 1.
            other = i
            while echelon.n_row(i)[j] == 0:
                other += 1
                if other >= echelon.rows: # last row or no more non-zero rows
                    j += 1
                    other = i
                if j >= echelon.cols:
                    break
                echelon = echelon.swap_rows(i, other)
                if steps:
                    print(echelon)

            if j >= echelon.cols:
                break

            # 2. Divide the i-th row by aij to make the pivot entry = 1.
            echelon = echelon.scale_row(i, 1/echelon.n_row(i)[j])
            if steps:
                print(echelon)

            # 3. Eliminate all other entries in the j-th column by subtracting suitable multiples of the
            # i-th row from the other rows.
            counter, this = 1, (i + 1) % echelon.rows
            while counter < echelon.rows:
                echelon = echelon.rep_mul(this, i, 1, - echelon.n_row(this)[j] / echelon.n_row(i)[j])
                counter, this = counter + 1, (this + 1) % echelon.rows
                if steps:
                    print(echelon)

            # 4. Increase i by 1 and j by 1 to choose the new pivot element. Return to Step 1.
            i, j = i + 1, j + 1

        self.echelon = echelon.matrix_round()
        return echelon.matrix_round()

    def transpose(self):
        """Returns the transpose of the matrix."""
        return Matrix(*[self.n_col(j) for j in range(self.cols)])

    def det(self):
        """Returns the determinant of the matrix."""
        assert self.rows == self.cols, "Not a square matrix."
        if self.rows == 1:
            return self.n_row(0)[0]
        else:
            return sum([(-1) ** j * self.n_row(0)[j] * self.remove_ij(0, j).det()
                        for j in range(self.rows)])

    def inv(self):
        """Returns the inverse of the matrix."""
        determinant = self.det()
        assert determinant != 0, "Matrix is not invertible."
        return 1/determinant * self.adjoint()

    def eigenvalues(self):
        pass

    ###########
    # UTILITY #
    ###########
    def matrix_round(self, error=0.0001):
        """Returns a new matrix with rounded values."""
        new_matrix = Matrix(*self.lists[:])
        for i in range(new_matrix.rows):
            new_row = [round(entry) if abs(entry - round(entry)) <= error
                                    else entry for entry in new_matrix.n_row(i)]
            new_matrix.rep_row(i, new_row)
        return new_matrix

    def remove_ij(self, i, j):
        """Returns the matrix with row i and column j removed."""
        assert i < self.rows and j < self.cols, "Invalid indices."
        return Matrix(*[[self.n_row(y)[x] for x in range(self.cols) if x != j] for y in range(self.rows) if y != i])

    def cofactor(self):
        """Returns the cofactor matrix."""
        assert self.rows == self.cols, "Not a square matrix."
        return Matrix(*[[self.remove_ij(i, j).det() for j in range(self.cols)] for i in range(self.rows)])

    def adjoint(self):
        """Returns the adjoint matrix."""
        return self.cofactor().transpose()

    def symmetric(self):
        """Returns whether the matrix is symmetric."""
        assert self.rows == self.cols, "Not a square matrix."
        return self.transpose() == self
##########################
# FUNCTIONAL ABSTRACTION #
##########################

def matrix():
    print("Hello friend, please write the rows of your matrix below, separated by either commas or whitespace. When you have written all rows, hit enter.")
    matrix = []
    row_num = 1
    while True:
        this_row = str_to_list(input("Enter row #%s: " %row_num))
        if not this_row:
            break
        if not row_num == 1:
            assert len(this_row) == len(matrix[-1]), 'Your matrix is wonkily sized.'
        matrix.append(this_row)
        row_num += 1
    return matrix

def print_matrix(matrix):
    """Prints the swag matrix in swag form.
    >>> print_matrix(a)
    ⎡  1 2   3 ⎤
    ⎣ -4 0 -12 ⎦
    >>> print_matrix(I5)
    ⎡ 1 0 0 0 0 ⎤
    ⎢ 0 1 0 0 0 ⎥
    ⎢ 0 0 1 0 0 ⎥
    ⎢ 0 0 0 1 0 ⎥
    ⎣ 0 0 0 0 1 ⎦
    """
    max_char_in_col = [max([len(str(n)) for n in n_col(matrix, i)]) for i in range(col_count(matrix))]

    for i in range(len(matrix)):
        this_row = " ".join([" " * (max_char_in_col[j] - len(str(n_row(matrix, i)[j]))) + str(n_row(matrix, i)[j]) for j in range(col_count(matrix))])

        if i == 0:
            print("⎡", this_row, "⎤")
        elif i == len(matrix) - 1:
            print("⎣", this_row, "⎦")
        else:
            print("⎢", this_row, "⎥")

def add_matrix(m1, m2):
    assert row_count(m1) == row_count(m2) and col_count(m1) == col_count(m2), "Cannot add. Bad dimensions."
    return [[x + y for x, y in zip(n_row(m1, i), n_row(m2, i))] for i in range(row_count(m1))]

def sub_matrix(m1, m2):
    return add_matrix(m1, kmul_matrix(m2, -1))

def mul_matrix(m1, m2):
    assert col_count(m1) == row_count(m2), "Cannot multiply. Bad dimensions."
    return [[sum([x * y for x, y in zip(n_row(m1, i), n_col(m2, j))]) for j in range(col_count(m2))] for i in range(row_count(m1))]

def kmul_matrix(matrix, k):
    """Returns a new matrix, which is the old one multiplied by a constant k."""
    return [[k * x for x in n_row(matrix, i)] for i in range(row_count(matrix))]

def echelon(matrix):
    if row_count(matrix) == 1 or col_count(matrix) == 0:
        return matrix

    new_matrix = matrix[:]
    num_cols = col_count(new_matrix)
    for i in range(num_cols):
        if any(n_col(new_matrix, i)):

            # pushing all 0-heading rows in this column to the bottom
            row_index = 0
            while row_index < col_count(matrix) and not n_col(new_matrix, i)[row_index]:
                new_matrix = push_row(new_matrix, row_index)
                row_index += 1

            # scale first non-zero row to 1
            first_nonzero = 0
            for j in range(len(n_col(new_matrix, i))):
                if n_col(new_matrix, i)[j] == 0:
                    first_nonzero += 1
                else:
                    break
            this_entry = n_col(new_matrix, i)[0]
            new_matrix = scale_row(new_matrix, first_nonzero, 1 / int(this_entry))
            print('YO')

            # replace other rows to get 0s in column
            for j in range(first_nonzero, len(n_col(new_matrix, i))):
                if n_row(new_matrix, j)[i] != 0:
                    new_matrix = replace_multiple(new_matrix, j, 0, 1, -n_row(new_matrix, j)[i])

            # new_matrix = [replace_multiple(new_matrix, r, 0, 1, -n_row(new_matrix, r)[i])
            #                 if n_row(new_matrix, r)[i] != 0 else n_row(new_matrix, r) for r in range(1, len(new_matrix))]
    return new_matrix
def rref(matrix):
    pass

"""###############################"""
"""## Elementary Row Operations ##"""
"""###############################"""

def scale_row(matrix, row, k):
    """Returns a new matrix, with row #row scaled by k."""
    return [[k * e if row == i else e for e in n_row(matrix, i)] for i in range(len(matrix))]

def swap_rows(matrix, row1, row2):
    """Returns a new matrix, with row1 and row2 swapped"""
    new_matrix = matrix[:]
    new_matrix[row1], new_matrix[row2] = n_row(new_matrix, row2), n_row(new_matrix, row1) # Abstraction violation whoops
    return new_matrix

def replace_multiple(matrix, row_replaced, other_row, k_rep, k_other):
    """Returns a new matrix, with row_replaced replaced with a linear
    combination of itself and other_row with weights given by k_rep and k_other.
    """
    new_matrix = matrix[:]
    new_matrix[row_replaced] = [x + y for x, y in zip(n_row(scale_row(matrix, row_replaced, k_rep), row_replaced),
                                                      n_row(scale_row(matrix, other_row, k_other), other_row))]
    return new_matrix

"""################################"""
"""## Utility/Selector Functions ##"""
"""################################"""

def n_row(matrix, n):
    "Returns the nth row of matrix as a list"
    return matrix[n]

def n_col(matrix, n):
    "Returns the nth column of matrix as a list"
    return [matrix[i][n] for i in range(len(matrix))]

def row_count(matrix):
    return len(matrix)

def col_count(matrix):
    return len(matrix[0])

def is_piv_row(matrix, row):
    pass

def is_piv_col(matrix, col):
    pass

def gen_I(n):
    """Returns an nxn identity matrix."""
    return [[min(x // y, y // x) for x in range(1, n + 1)] for y in range(1, n + 1)]

def push_row(matrix, row):
    """Pushes row with index row to the bottom of the matrix, and shifts all other rows up"""
    return [matrix[i] for i in range(len(matrix)) if i != row] + [matrix[row]]

def str_to_list(string):
    """Turns a string input and converts it to a number list.
    >>> str_to_list("3, 4, 2   -24, 3 ")
    [3, 4, 2, -24, 3]
    """
    string = " ".join(string.replace(",", "").split())
    return [int(s) if s[0] != "-" else -1 * int(s[1:]) for s in string.split()]

"""############################"""
"""## Sample/Useful Matrices ##"""
"""############################"""

I1 = [1]
I2 = gen_I(2)
I3 = gen_I(3)
I4 = gen_I(4)
I5 = gen_I(5)

a = [[1, 2, 3], [-4, 0, -12]]
b = [[0, 4], [-1, 12], [-9, 8]]

a = Matrix([1, 2, -3, 10], [-10, 0, 0, 12], [2, 3, -1, 0], [0, 1, 0, 5])
b = Matrix([2, 3, 0], [0, -1, 2])
c = Matrix([2, 5], [0, 1], [6, -3])
d = Matrix([4, 3], [0, -1])
