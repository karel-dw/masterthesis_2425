import numpy as np



def jacobian_dh_to_dF(cell_0):
    """
    Compute Jacobian of h = h0 @ F, where F is flattened in row-major order.
    Returns 9x9 matrix where rows = vec(h), cols = vec(F) [row-major]
    """
    J = np.zeros((9, 9))
    for i in range(3):        # row in h
        for j in range(3):    # col in h (same as col in F)
            for k in range(3):  # summation index over h0 rows and F cols
                row = 3*i + j       # index in vec(h)
                col = 3*k + j       # index in vec(F), row-major
                J[row, col] = cell_0[i, k]
    return J

def cell_jac(coo,cell_0,num_atoms):
    if coo[-1:] == 'h':
        return np.eye(9)
    if coo[-1:] == 'F':
        return jacobian_dh_to_dF(cell_0)
    if coo[-2:] == 'FN':
        return jacobian_dh_to_dF(cell_0) / np.cbrt(num_atoms)
    
def pos_jac(coo,cell,cell_0,num_atoms):
    if coo[:1] == 's':
        return np.copy(cell.T)
    if coo[:1] == 'r':
        return np.eye(3)
    if coo[:1] == 'd':
        return np.copy((np.linalg.inv(cell_0)@cell).T)
    if coo[:2] == 'Ns':
        pos_reg = np.copy(cell.T) / np.cbrt(num_atoms)
        return np.copy(pos_reg)
    

def transform_pos_different_to_regular(
        coo, #string to indicate type of transformation
        pos_diff,  #(n,3) array with positions in a non regular coordinate
        cell,       #(3,3) array for the cell h
        cell_0,     #(3,3) array for the cell h of optimized geometry
        num_atoms=None,
    ):
    if coo[:1] == 's':
        pos_reg = pos_diff @ cell
        return np.copy(pos_reg)
    if coo[:1] == 'r':
        return np.copy(pos_diff)
    if coo[:1] == 'd':
        pos_reg = pos_diff @ np.linalg.inv(cell_0) @ cell
        return np.copy(pos_reg)
    if coo[:2] == 'Ns':
        pos_reg = pos_diff @ cell / np.cbrt(num_atoms)
        return np.copy(pos_reg)


def transform_pos_regular_to_different(
        coo, #string to indicate type of transformation
        pos_reg,  #(n,3) array with positions in a non regular coordinate
        cell,       #(3,3) array for the cell h
        cell_0,
        num_atoms=None,
    ):
    if coo[:1] == 's':
        pos_diff = pos_reg @ np.linalg.inv(cell)
        return np.copy(pos_diff)
    if coo[:1] == 'r':
        return np.copy(pos_reg)
    if coo[:1] == 'd':
        pos_diff = pos_reg @ np.linalg.inv(cell) @ cell_0
        return np.copy(pos_diff)
    if coo[:2] == 'Ns':
        pos_diff = pos_reg @ np.linalg.inv(cell) * np.cbrt(num_atoms)
        return np.copy(pos_diff)

def transform_cell_different_to_regular(
        coo, #string to indicate type of transformation
        cell_diff,       #(3,3) array for the cell h
        cell_0,     #(3,3) array for the cell h of optimized geometry
        num_atoms=None,
    ):
    if coo[-1:] == 'h':
        return np.copy(cell_diff)
    if coo[-1:] == 'F':
        return np.copy(cell_0 @ cell_diff)
    if coo[-2:] == 'FN':
        return np.copy(cell_0 @ cell_diff / np.cbrt(num_atoms))

def transform_cell_regular_to_different(
        coo, #string to indicate type of transformation
        cell_reg,       #(3,3) array for the cell h
        cell_0,
        num_atoms=None,
    ):
    if coo[-1:] == 'h':
        return np.copy(cell_reg)
    if coo[-1:] == 'F':
        return np.copy(np.linalg.inv(cell_0) @ cell_reg)
    if coo[-2:] == 'FN':
        return np.copy(np.linalg.inv(cell_0) @ cell_reg * np.cbrt(num_atoms))