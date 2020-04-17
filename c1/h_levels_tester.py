import numpy as np
import sys
import time
from scipy.integrate import quad
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from matplotlib import pyplot as plt

# Constants
c1 = 0.0380998  # nm^2 eV
c2 = 1.43996  # nm eV
r0 = 0.0529177  # nm
h = 6.62606896e-34  # J s
c = 299792458.  # m/s

def get_force(r, alpha=0):
    """
    Method: calculates the electrostatic force between a proton and
    an electron with charge ±e respectively

    :param r:  radial distance
    :param alpha:  perturbation
    :return force: value for force as a float
    """
    force = -c2 * 1 / (r)**2 * (r / r0)**alpha
    return force

def get_potential(a, b, alpha=0):
    """
    Method: calculates the electrostatic potential for a pairwise
    system through the integration of the force between the pair

    :param r: radial distance
    :param a: lower limit of integration
    :param b: upper limit of integration
    :param alpha:  perturbation
    :return potential: potential value for corresponding radial distance
    """
    potential = np.array([])
    potential = np.append(potential, quad(get_force, a, b, args=(alpha))[0])
    return potential

def get_error(radii, numerical_array, alpha=0):
    """
    Method: quantifies the accuracy the numerical solution
    compared to an analytical value

    :param radii:  numPy array of radial distances
    :param numerical_array:  numPy array of numerical solutions
    :param alpha:  perturbation
    :return error_array: numPy array whose elements consist
    of the differences between the numerical and analytical
    solution
    """
    exact_array = np.array([])
    for i in range(radii.size):
        exact_array = np.append(exact_array, -c2 * 1 / (r0)**alpha * ((radii[i])**(alpha - 1)) / (alpha - 1))
    error_array = abs(exact_array + numerical_array)
    return error_array

def get_hamiltonian(radii, dr, alpha=0):
    """
    Method: determines the Hamiltonian using sparse matrices
    where H = -hbar**2/2m * (d/dr)**2 + V(r) 

    :param radii: numPy array of radial distances
    :param dr: step
    :return H: Hamiltonian represented as a sparse matrix
    """
    dr_array = np.arange(dr, radii.size * dr + dr, dr)
    pot_diag = get_potential(dr_array, alpha)
    pot_matrix = diags(pot_diag)

    diagonals = [np.full(radii.size, -2),
                 np.full(radii.size, 1),
                 np.full(radii.size, 1)]
    hh = diags(diagonals, [0, -1, 1])
    delta = 1 / (dr)**2 * hh

    H = -c1 * delta + pot_matrix
    return H

def get_evals(matrix):
    """
    :Method: determines the 2 smallest eigenvalues of a matrix

    :param matrix: matrix used to determine eigenvalues
    :return E[0], E[1]: the two smallest eigenvalues
    """
    evals, evecs = eigs(matrix, k=2, which="SR")
    E = np.sort(np.real(evals))
    return E[0], E[1]

def main():

    N = 2500  # Number of integration steps
    radii = np.linspace(0.01, 1.0, num=N)  # Defining radial distances
    dr = max(radii) / N  # Defining step spacing
    alpha_values = np.array([0.0]) # Defining pertubations to Coulomb potential
    delta_E_array = np.array([]) # Defining array for energy spacing between n=2 and n=1
    delta_E_max = 10.21286573 # maximum energy diffence between n=2 and n=1 levels
    delta_E_min = 10.19606828 # minimum energy difference between n=2 and n=1 levels

    for alpha in alpha_values: # Looping over different pertubation strengths
        potential = get_potential(radii.all(), np.inf, alpha)  # Finding potential of H atom
        error_array = get_error(radii, potential, alpha)  # Determinig accuraxy of numerical method
        print(f"The accuracy of our potential calculation is {np.amax(error_array)}")
    
   
        #H = get_hamiltonian(radii, dr, alpha)  # Forming Hamiltonian
        #eval_1, eval_2 = get_evals(H)  # Calculating eigenvalues
        #delta_E = eval_2 - eval_1
        #if delta_E > delta_E_max or delta_E < delta_E_min:
        #    pass
        #else:
        #    print(f"alpha={alpha} corresponds to λ = 121.5 ± 0.1nm")
        #delta_E_array = np.append(delta_E_array, delta_E)
        #print(f"The eigenvalues for alpha = {alpha} are {eval_1} and {eval_2}")
        

    # Plotting electrostatic potential vs radial distance for H atom
    plt.rcParams['figure.figsize'] = (10, 6)  # inches
    plt.rcParams['font.size'] = 14
    plt.title("Electrostatic Potential vs. Radial Distance")
    plt.xlabel("Radial Distance (nm)")
    plt.xlim(right=1.0)
    plt.ylabel("Potential (eV)")
    plt.plot(radii, potential)
    plt.show()

    # Plotting how the energy spacing of H atom varies with different perturbations
    #plt.rcParams['figure.figsize'] = (10, 6)  # inches
    #plt.rcParams['font.size'] = 14
    #plt.title("En=2 - En=1 for the Hydrogen atom with a perturbed potential")
    #plt.xlabel("Alpha (Perturbation)")
    #plt.ylabel("Delta E (eV)")
    #plt.plot(alpha_values, delta_E_array)
    #plt.show()

main()
