import math
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the test problem:
# only u_t + a u_x = 0


def get_testproblem(testfunction):
    testproblem = {}

    testproblem["a"] = 1.0

    # t_0: initial time
    testproblem["t0"] = 0.0

    # xL,xR : domain [xL,xR]
    testproblem["xL"] = 0.0
    testproblem["xR"] = 1.0

    # tend: final time
    testproblem["tend"] = 1.0

    # choice
    testproblem["choice"] = testfunction

    # u0: initial data
    if (testfunction == "smooth"):
        testproblem["u0"] = lambda x: np.sin(2 * math.pi * x)
    elif (testfunction == "disc"):
        testproblem["u0"] = lambda x: np.where(abs(x - 0.5) < 0.25, 1.0, 0.0)
    else:
        raise Exception(
            'Stop in testproblem. Choice of test problem does not exist')

    # uexact: exact solution
    testproblem["uexact"] = lambda x, t: testproblem["u0"](
        (x - t * testproblem["a"]) % 1.0)

    return testproblem


# Set parameters for solving the problem
def define_default_parameters():
    parameters = {}

    # nrefine: how many refinements do we do?
    parameters["Nrefine"] = 0

    # N: number of grid points (on coarsest grid)
    parameters["N"] = 80

    # max_steps: maximal number of time steps
    parameters["max_steps"] = 10000

    # CFL: CFL number used
    parameters["CFL"] = 0.4

    # plot_freq: how often do we plot?
    parameters["plot_freq"] = 10

    return parameters


def graph(U_comp, x_plot, time, uexact, xL, xR, method, testfunction, initialize, final_time):
    # evaluate true solution
    U_true = uexact(x_plot, time)

    if initialize:
        plt.figure(1)
        plt.ion()

    plt.figure(1)
    plt.plot(x_plot, U_comp, 'r.', markersize=4, label='computed solution')
    plt.plot(x_plot, U_true, 'k-', label='true solution')
    plt.title(method)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.xlim(xL, xR)

    if final_time == 1:
        plt.figure(2)
        plt.plot(x_plot, U_comp, 'r.', markersize=4,
                 label='computed solution')
        plt.plot(x_plot, U_true, 'k-', label='true solution')
        plt.title(method)
        plt.xlabel('x')
        plt.ylabel('u')
        plt.xlim(xL, xR)
        plt.legend()
        save_str = 'results/' + method + '_' + testfunction + '.jpg'
        plt.savefig(save_str)

    plt.show()
    plt.pause(0.1)
    plt.clf()


# Upwind flux 
def upwind(UL, UR, a):
    if a >= 0:
        return a * UL
    else:
        return a * UR



# Compute minmod of two input arguments.
def minmod2(V1, V2):
    ## TODO
    if np.sign(V1) == np.sign(V2):
        if abs(V1)<abs(V2):
            return V1
        else:
            return V2
    else:
        return 0




# Compute minmod of three input arguments.
def minmod3(V1, V2, V3):
    ## TODO
    if np.sign(V1) == np.sign(V2) == np.sign(V3):
        return min([V1, V2, V3], key=abs)
    else:   
        return 0


# Compute slope of inner cells 
def slope_limiters(limiter, U, dx):
    N = len(U) - 4 
    n_ghost = 2
    u_inner = U[n_ghost-1 : N+n_ghost+1]
    
    d_back = (u_inner[1:-1] - u_inner[:-2]) / dx
    d_forw = (u_inner[2:] - u_inner[1:-1]) / dx
    d_cent = (u_inner[2:] - u_inner[:-2]) / (2 * dx)

    slopes = np.zeros(N) 

    for i in range(N):
        if limiter == 'const':
            slopes[i] = 0.0
        elif limiter == 'nolim':
            slopes[i] = d_cent[i]
        elif limiter == 'minmod':
            slopes[i] = minmod2(d_back[i], d_forw[i])
        elif limiter == 'MC':
            slopes[i] = minmod3(d_cent[i], 2 * d_back[i], 2 * d_forw[i])
            
    return slopes[0] if N == 1 else slopes


# Advance in time with forward Euler.
def advance_in_time_euler(U, fu_rhs, dt):
    return U + dt * fu_rhs(U)


# Advance in time with Heun.
def advance_in_time_heun(U, fu_rhs, dt):
    ## TODO
    U1 = U + dt*fu_rhs(U)
    U2 = U1 + dt*fu_rhs(U1)
    Un_1 = 0.5*(U+U2)
    return Un_1


# Driver
#
# Output:
# - errL1: error in L1 norm
# - errLmax: error in maximum norm


def my_driver(limiter, scheme_order, testproblem, parameters, N):

    # Extract problem information
    a = testproblem["a"]
    xL = testproblem["xL"]
    xR = testproblem["xR"]
    t0 = testproblem["t0"]
    tend = testproblem["tend"]
    u0 = testproblem["u0"]
    uexact = testproblem["uexact"]
    testfunction = testproblem["choice"]

    max_steps = parameters["max_steps"]
    CFL = parameters["CFL"]
    plot_freq = parameters["plot_freq"]

    # Grid generation

    # calculate dx: cell length in space
    dx = (xR - xL) / N

    # number of ghost cells
    n_ghost = scheme_order

    # create mesh with ghost cells
    x = np.linspace(xL - dx * n_ghost, xR + dx * (n_ghost - 1), N + 2*n_ghost)
    x = x + 0.5 * dx
    ind_dof = range(n_ghost, N + n_ghost)

    # Initialization

    # initialize U
    U = u0(x)

    # start time marching
    time = t0
    done = 0

    # do output
    print('\n# START PROGRAM')

    # Plot initial data
    if plot_freq != 0:
        graph(U[ind_dof], x[ind_dof], time, uexact, xL, xR,
              limiter, testfunction, True, False)

    # function to set periodic boundary conditions
    def set_pbc(U):
        for i in range(0, n_ghost):
            U[i] = U[N + i]
            U[N + n_ghost + i] = U[n_ghost + i]

    # Time stepping
    for j in range(1, max_steps + 1):
        # note: initialization of boundary conditions at
        # the beginning of the time-step loop has been
        # moved to the update step so that both schemes
        # can use the same framework

        # impose CFL condition to find dt
        smax = np.max(np.abs(U[n_ghost:-n_ghost]))
        dt = CFL * (dx / smax)

        # check that time 'tend' has not been exceeded
        if (time + dt) > tend:
            dt = tend - time
            done = 1
        time = time + dt

        # do output to screen if wished
        if (plot_freq != 0) and (j % plot_freq) == 0:
            print('Taking time step %i: \t update from %f \t to %f' %
                  (j, time - dt, time))

        if scheme_order == 1:
            def fu_rhs(U):
                # step 1: update ghost values {0, N+1}
                set_pbc(U)

                # step 2: compute left and right velocities and
                # compute upwind fluxes
                UL = U[0:-1]
                UR = U[1:]

                flux = upwind(UL, UR, a)

                # step 3: compute right-hand side [1:N+1[
                rhs = np.zeros(N+2)
                rhs[1: -1] = - (flux[1:] - flux[0: -1]) / dx

                return rhs

            U = advance_in_time_euler(U, fu_rhs, dt)

        elif scheme_order == 2:
            def fu_rhs(U):
                # ghost cell values
                set_pbc(U)
                N = len(U) -4
                inner_slopes = slope_limiters(limiter, U, dx)
                slopes = np.zeros_like(U)
                slopes[2 : N+2] = inner_slopes
                slopes[0] = slopes[N]
                slopes[1] = slopes[N+1]
                slopes[N+2] = slopes[2]
                slopes[N+3] = slopes[3]
                
                u_l = U[1 : N+2] + 0.5 * dx * slopes[1 : N+2]
                u_r = U[2 : N+3] - 0.5 * dx * slopes[2 : N+3]
                flux = upwind(u_l, u_r, a)
                rhs = np.zeros(N + 4)
                rhs[2 : N+2] = - (flux[1:] - flux[:-1]) / dx
                return rhs

            U = advance_in_time_heun(U, fu_rhs, dt)

        else:
            raise Exception(
                'Stop in my_dirver. Choice of scheme order does not exist')

        # draw graph if wished
        if (plot_freq != 0) and (j % plot_freq) == 0:
            graph(U[ind_dof], x[ind_dof], time, uexact, xL, xR,
                  limiter, testfunction, False, False)

        # if we have done the calculation for tend, we can stop
        if done == 1:
            print('Have reached time tend; stop now')
            break

    if j >= max_steps:
        print('Stopped after %i steps.' % max_steps)
        print('Did not suffice to reach the end time %f.' % tend)

    if plot_freq != 0:
        graph(U[ind_dof], x[ind_dof], time, uexact, xL, xR,
              limiter, testfunction, False, True)

    # Compute error
    true_sol = uexact(x[ind_dof], time)
    err = U[ind_dof] - true_sol
    err_L1 = np.sum(np.abs(err) * dx)
    err_max = np.max(np.abs(err))

    print('Error in L1:\t\t %3.2e' % err_L1)
    print('Error in L^infty:\t %3.2e\n' % err_max)

    return err_L1, err_max

# Main file for solving
# linear advection equation u_t + a u_x = 0
# using Godunov of second order with
#  -- unlimited
#  -- minmod
#  -- MC
# using periodic b.c.


def main():
    if not os.path.exists("results"):
        os.makedirs("results")
    print("")

    # Choose test problem:
    #
    # Options:
    # 'disc': 0-1-0 discontinuity
    # 'smooth': sine function
    testfunction = 'smooth'

    # Choose limiter:
    # Options:
    # 'const'    : slope = 0
    # 'nolim'    : unlimited central difference quotients
    # 'minmod'   : minmod limiter
    # 'MC'       : MC limiter
    limiter = 'const'

    # Scheme order:
    # scheme_order = 1:  piecewise constant data, Euler time stepping
    # scheme_order = 2:  slope limiter reconstruction, 2nd order Heun
    scheme_order = 1

    # read in test problem:
    testproblem = get_testproblem(testfunction)

    # read in problem parameters:
    parameters = define_default_parameters()

    # call driver
    if parameters["Nrefine"] < 0:
        raise Exception("Stop in main. Nrefine negative!")

    errL1_vec = np.zeros(parameters["Nrefine"] + 1)
    errLmax_vec = np.zeros(parameters["Nrefine"] + 1)
    N_vec = np.zeros(parameters["Nrefine"] + 1)

    # call the driver routine for different grid sizes
    N = parameters["N"]
    for k in range(parameters["Nrefine"] + 1):
        [errL1, errLmax] = my_driver(
            limiter, scheme_order, testproblem, parameters, N)

        N_vec[k] = N
        errL1_vec[k] = errL1
        errLmax_vec[k] = errLmax

        N = N * 2

    # compute convergence rate
    rate = np.diff(np.log(errL1_vec)) / np.diff(np.log(1. / N_vec))
    rateLmax = np.diff(np.log(errLmax_vec)) / np.diff(np.log(1. / N_vec))

    # write results to file
    with open("results/error.txt", "w") as f:
        f.write('\nL1 error:\n')
        f.write('%i \t %5.3e \t NaN\n' % (N_vec[0], errL1_vec[0]))
        for i in range(1, parameters["Nrefine"] + 1):
            f.write('%i \t %5.3e \t %3.2f\n' %
                    (N_vec[i], errL1_vec[i], rate[i - 1]))

        f.write('\nLmax error:\n')
        f.write('%i \t %5.3e \t NaN\n' % (N_vec[0], errLmax_vec[0]))
        for i in range(1, parameters["Nrefine"] + 1):
            f.write('%i \t %5.3e \t %3.2f\n' %
                    (N_vec[i], errLmax_vec[i], rateLmax[i - 1]))

    # write results to file for use in latex table
    with open("results/error_latex.txt", "w") as f:
        f.write('N  & L1-err & L1-ord & Lmax-err & Lmax-ord\n')
        f.write('%i & %5.3e & -- & %5.3e & --\n' %
                (N_vec[0], errL1_vec[0], errLmax_vec[0]))
        for i in range(1, parameters["Nrefine"] + 1):
            f.write('%i & %5.3e & %3.2f & %5.3e & %3.2f\n' % (
                N_vec[i], errL1_vec[i], rate[i - 1], errLmax_vec[i], rateLmax[i - 1]))

def main_test():
    if not os.path.exists("results"):
        os.makedirs("results")

    test_functions = ['smooth', 'disc']
    limiters = ['const', 'nolim', 'minmod', 'MC']
    
    # Open a summary file to store all results for part (f)
    with open("results/full_convergence_report.txt", "w") as report:
        for func in test_functions:
            report.write(f"\n{'='*80}\n")
            report.write(f"TEST PROBLEM: {func.upper()}\n")
            report.write(f"{'='*80}\n")
            
            for lim in limiters:
                params = define_default_parameters()
                params["N"] = 20          # Base grid
                params["Nrefine"] = 7     # Refines up to 2560
                
                # Order 1 for 'const', Order 2 for others
                order = 1 if lim == 'const' else 2
                testproblem = get_testproblem(func)

                report.write(f"\nLimiter: {lim} | Order: {order}\n")
                report.write(f"{'N':<8} & {'L1-err':<10} & {'L1-ord':<8} & {'Lmax-err':<10} & {'Lmax-ord':<8}\n")
                report.write("-" * 65 + "\n")

                errL1_vec = []
                errLmax_vec = []
                N_vec = []

                N_current = params["N"]
                for k in range(params["Nrefine"] + 1):
                    # Enable plotting ONLY for N=80 to answer part (e)
                    params["plot_freq"] = 10 if N_current == 80 else 0
                    
                    e1, e_max = my_driver(lim, order, testproblem, params, N_current)
                    
                    errL1_vec.append(e1)
                    errLmax_vec.append(e_max)
                    N_vec.append(N_current)
                    
                    if k == 0:
                        report.write(f"{N_current:<8} & {e1:.3e} & {'--':<8} & {e_max:.3e} & {'--':<8}\n")
                    else:
                        # Calculate rates
                        r1 = np.log(errL1_vec[k-1]/errL1_vec[k]) / np.log(2.0)
                        rmax = np.log(errLmax_vec[k-1]/errLmax_vec[k]) / np.log(2.0)
                        report.write(f"{N_current:<8} & {e1:.3e} & {r1:<8.2f} & {e_max:.3e} & {rmax:<8.2f}\n")
                    
                    N_current *= 2
                report.write("\n")

    print("\nDone! Full results saved to 'results/full_convergence_report.txt'")

if __name__ == '__main__':
    main()
