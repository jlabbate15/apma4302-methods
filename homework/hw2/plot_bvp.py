import numpy as np
from petsc4py import PETSc
import subprocess
import matplotlib.pyplot as plt

def read_hdf5_vec(filename, vec_name):
    """
    Read PETSc HDF5 viewer output and convert to numpy arrays.
    
    Parameters:
    filename: str - path to the HDF5 file
    vec_name: str - name of the vector to read  
    
    Returns:
    numpy array containing the data
    """
    # Create a viewer for reading HDF5 files
    viewer = PETSc.Viewer().createHDF5(filename, 'r')   
    
    # Create a Vec to load the data
    vec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    vec.setName(vec_name)
    vec.load(viewer)
   
    # Convert to numpy array
    array = vec.getArray()
    
    # Clean up
    vec.destroy()
    viewer.destroy()
    
    return array.copy()


def plot_bvp_solution(x, u_numeric, u_exact):
    """
    Plot the numerical and exact solutions of the BVP.
    
    Parameters:
    x: numpy array - grid points
    u_numeric: numpy array - numerical solution
    u_exact: numpy array - exact solution
    """
    import matplotlib.pyplot as plt
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(x, u_numeric, 'b-', label='Numerical Solution', linewidth=2)
    ax1.plot(x, u_exact, 'r--', label='Exact Solution', linewidth=2)
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('u(x)', fontsize=14)
    ax1.set_title('BVP Numerical vs Exact Solution', fontsize=16)
    ax1.legend(fontsize=12)
    ax1 .grid(True)

    ax2.plot(x, u_numeric - u_exact, 'g--', label='Error', linewidth=1)
    ax2.set_ylabel('Error', fontsize=14)
    ax2.legend(loc='lower right', fontsize=12)
    ax2.set_ylim(-np.max(np.abs(u_numeric - u_exact)) * 3., np.max(np.abs(u_numeric - u_exact)) * 3.)

    plt.show()

if __name__ == "__main__":


    # Example usage
    # read numerical and exact solutions from HDF5 files
    h5_filename = 'bvp_solution.h5'  # Update with your actual filename
    # u = read_hdf5_vec(h5_filename, 'u') 
    # u_exact = read_hdf5_vec(h5_filename, 'uexact')

    # x = np.linspace(0, 1, len(u))  
    
    # plot_bvp_solution(x, u, u_exact)


    # Part c usage
    num_procs = 4
    ks = np.array([1,5,10])
    ms = np.linspace(40,1280,32,dtype=int)
    hs = 1/ms
    err = np.zeros([len(ms)])
    i=0
    fig,ax = plt.subplots()
    for k in ks:
        j=0
        for m in ms:
            with open("options_file", "w") as f:
                f.write("-bvp_m "+str(m)+"\n")
                f.write("-bvp_gamma 0.\n")
                f.write("-bvp_k "+str(k)+"\n")
                f.write("-bvp_c 3.\n")
                f.write("-ksp_rtol 1.e-8\n")
                f.write("-ksp_atol 1.e-10\n")
                f.write("-ksp_monitor\n")
                f.write("-ksp_type preonly\n")
                f.write("-pc_type lu\n")
                f.write("-pc_factor_solver_type mumps\n")

            subprocess.run([
                "mpiexec",
                "-np",
                str(num_procs),
                "./bvp",
                "-options_file",
                "options_file",
            ])

            u = read_hdf5_vec(h5_filename, 'u') 
            u_exact = read_hdf5_vec(h5_filename, 'uexact')
            err[j] = np.linalg.norm(u-u_exact)
            j+=1

        ax.plot(hs,err,label='k='+str(k))
        i+=1
    ax.set_title("Convergence of Error as a Function of Grid Spacing")
    ax.set_xlabel("Grid Spacing")
    ax.set_ylabel(r'$|u-u_{exact}|$')
    ax.legend()
    plt.savefig('convergence_4f4.png')