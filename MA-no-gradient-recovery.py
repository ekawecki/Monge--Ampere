"""This program is currently set up to approximate the uniformly convex solution of the Monge--Ampere optimal mass transport equation, coinciding with the prescribed Gaussian curvature equation
    det( D^2 u(x, y) ) = f(x,y)*(|grad(u)|^2+1)^2,
    on the unit disk, with the boundary condition |grad(u)|^2-1=0, as well as some other domains such as an oval, and the union of a half oval and half unit disk.
    You can change boundary conditions, and domains, the specific choice of domain is based on the lack of a computable distance function to define the boundary conditions for polyhedral domains.
    This code uses an L^2 gradient recovery method producing optimal results for piecewise linear approximations.
    If you do use this code, please acknowledge the authors Ellya Kawecki, Omar Lakkis & Tristan Pryer.
    This code is compatible with an earlier release of FEniCS than the latest available version, so they may be some issues with domains, etc, please contact me if you have any problems. To download FEniCS, simply google "fenics", I recommend using either a Linux or Mac system.
    All options are at the top of the file are integer switches, change these to
    change the options.
    """

__author__ = "Ellya Kawecki (ekawecki@cct.lsu.edu / kawecki@maths.ox.ac.uk)"
__date__ = "09-11-2018"
__copyright__ = "Copyright (C) 2015 Ellya Kawecki"
__license__  = "GNU LGPL Version 2.1 (or whatever is the latest one)"


from numpy import *
from dolfin import *
from time import clock, time
import matplotlib.pyplot as plt
from mshr import *
import mshr as mshr
from fenics import Point, mesh
from mshr import Sphere, generate_mesh



# Adaptivity options
N = 6 #maximum number of sets of Newton iterations.
experiment = False #True = Testing without mesh refinement.
if experiment == True:
    itermax = 1
else:
    itermax = N
tol = 5e-14        # Error tolerance
# maximal number of iterations (in h)
hm = zeros(itermax)
benchmark = True
print("Representing solution and Hessian via a 2-0 mixed system.")

#Choose a domain
domain = 0 #Choose a domain 0 = unit circle, 1 = Unit square, 2 = Square centred at 0 width 1, 4 = curved square, 5 = non symmetric domain, 6 = oval

#Choose an initial guess, i.e. u_0
prob = 2

# Function spaces for the solution and Hessian
solution_space = "CG"
Hessian_solution_space = "CG"
e_L2 = []; e_H1 = []; e_H2 = []; e_D2 = []; ndof = []; one = [];

# Create the triangulation of the computational domain
if domain==0:
    dom = Circle(Point(0, 0), 1)
    mesh = generate_mesh(dom, 4)
    parameters['allow_extrapolation'] = True
    print(mesh.hmax())
    PENALTY = False #toggling boundary condition penalty parameter
#    plot(mesh,interactive = True)
elif domain==1:
    mesh = UnitSquareMesh(50, 50)
    PENALTY = 1 #toggling boundary condition penalty parameter
elif domain==2:
    mesh = RectangleMesh(-0.5,-0.5,0.5,0.5,50,50,'left')
    parameters['allow_extrapolation'] = True
    PENALTY = True #toggling boundary condition penalty parameter
elif domain==3:
    mesh = RectangleMesh(0.5,0.5,1.0,1.0,4, 4,'left')
    parameters['allow_extrapolation'] = True
    PENALTY = 1 #toggling boundary condition penalty parameter
elif domain==4:
    mesh = Mesh("curved_square_mesh_ref60.xml")
    plot(mesh,interactive = True)
    PENALTY = False
elif domain==5:
    mesh = Mesh("nonsymmetric_domain_ref45.xml")
    PENALTY = 1 #toggling boundary condition penalty parameter
elif domain == 6:
    mesh = Mesh("oval_mesh_ref45.xml")
    PENALTY = 1 #toggling boundary condition penalty parameter
else:
    print("No domain with that associated value, fool!")

#choosing integration quadrature degree, choose 4 for P^2 elements, 16 for P^3 elements.
parameters["form_compiler"]["quadrature_degree"] = 8

#Defining finite element spaces, the last input is for the piece-wise polynomial degree.
deg = 2
FES = FunctionSpace(mesh, solution_space, deg)
FESH = FunctionSpace(mesh, Hessian_solution_space, deg)
CG = FunctionSpace(mesh, "CG", deg)

#Defining easy to call coordinates.
x0 = Expression('x[0]', degree = 1)
x1 = Expression('x[1]', degree = 1)
x = project(x0,CG)
y = project(x1,CG)

#calculating mesh size h
h = CellDiameter(mesh)
hmin = mesh.hmin()


#Defining initial guess choices, note that the constants must be changed when the domain changes, as we seek a solution with zero integral. Also we calculate first and second derivatives for use in setting up benchmarks.

if prob == 0:
    u = 0.5*(x**2+y**2)-1.0/4.0#1.0/12.0#
    du0 = x
    du1 = y
    d2udxx = 1.0
    d2udxy = Constant(0.0)
    d2udyy = 1.0
    C = Constant(0.0)
elif prob == 1:
    u = exp(0.5*(x**2+y**2))/sqrt(exp(1))-(2.0/exp(0.5))*(exp(0.5)-1)
    du0 = x*exp(0.5*(x**2+y**2))/sqrt(exp(1))
    du1 = y*exp(0.5*(x**2+y**2))/sqrt(exp(1))
    d2udxx = exp(0.5*(x**2+y**2))*(x**2+1)/sqrt(exp(1))
    d2udxy = exp(0.5*(x**2+y**2))*(x*y)/sqrt(exp(1))
    d2udyy = exp(0.5*(x**2+y**2))*(y**2+1)/sqrt(exp(1))
    C = Constant(-(2.0/exp(1))*(exp(0.5)-1))
elif prob == 2:
    rho = pow(x,2) + pow(y,2)
    u = -sqrt(2-rho)+(2.0/3.0)*(2.0*sqrt(2)-1)
    du0 = x/sqrt(2-rho)
    du1 = y/sqrt(2-rho)
    d2udxx = -(-2+pow(y,2))/(pow((2-rho),1.5))
    d2udyy = -(-2+pow(x,2))/(pow((2-rho),1.5))
    d2udxy = -(-x*y)/(pow(2-rho,1.5))
    C = Constant((2.0/3.0)*(2.0*sqrt(2)-1))

#Defining some elementary matrix and vector operations
def tensorp(p, q):
    ptq = as_matrix( [ [p[0]*q[0], p[0]*q[1] ], [p[1]*q[0], p[1]*q[1] ] ] )
    return ptq
def determinant(M):
    deter = M[0][0]*M[1][1] - M[0][1]*M[1][0]
    return deter
def elarea(p):
    el = 1 + p[0]**2 + p[1]**2
    return el
def frob(M, N):
    frobMN = M[0][0]*N[0][0] + M[1][1]*N[1][1] + M[0][1]*N[0][1] + M[1][0]*N[1][0]
    return frobMN
def divmat(M):
    divmatM = as_vector( [ M[0][0].dx(0) + M[0][1].dx(1), M[1][0].dx(0) + M[1][1].dx(1)] )
    return divmatM
def matnormal(M,n):
    matnormM = as_vector( [ M[0][0]*n[0] + M[0][1]*n[1], M[1][0] + M[1][1].dx(1)] )

#Defining distance functions
#defining maximum and minimum function using conditional UFL statements (these can be used to define distance functions that can be used in the variational formulation).
def makemin(a,b):
    mkmin = conditional(le(a,b),a,0)+conditional(le(b,a),b,0)
    return mkmin
def makemax(a,b):
    mkmax = conditional(ge(a,b),a,0)+conditional(ge(b,a),b,0)
    return mkmax
# defining the signed distance function inside and outside of the square given for domain = 2
def distins(a,b):
    dist = makemax(abs(a)-0.5,abs(b)-0.5)
    return dist
def distout(a,b):
    dist = makemin(abs(a)-0.5,abs(b)-0.5)
    return dist
# defining signed distance function of the square given for domain =  2
def distsq(a,b):
    distance = makemax(0,distout(a,b))+makemin(0,distins(a,b))
    return distance
# defining signed distance function for the oval given for domain = 6
def ovaldist(a,b):
    distance =a**2/(4.0*(1.0-alph*b))+(b**2)/4.0-1.0
    return distance
# defining signed distance function for the unit circle given for domain = 0
def unitcircledist(u,v):
    distance = u**2+v**2-1.0
    return distance

#Calculating values for benchmarks
eye = as_matrix([[1,0],[0,1]])
gradu = as_vector([du0, du1])
Hessu = as_matrix([[d2udxx, d2udxy], [d2udxy, d2udyy]])
def g(a,b):
    gee = (1.0+a**2+b**2)**2
    return gee
#Calculating f(x,y) for benchmarking purposes.
if benchmark == True:
    f = determinant(Hessu)/(dot(gradu,gradu)+1)**2
else:
    f = 1


#Defining our initial guess u_0
u0 = u
d2udxx0 = d2udxx
d2udxy0 = d2udxy
d2udyy0 = d2udyy
C0 = C

#Beginning Newton iteration loop:
i = 0
while (i < itermax):
    if domain==0:
        if experiment ==True:
            mesh = mesh
        else:
            dom = Circle(Point(0, 0), 1)
            mesh = generate_mesh(dom, 2**(i+1))
            parameters['allow_extrapolation'] = True
            print(mesh.hmax())
    elif domain==1:
        if experiment == True:
            mesh = mesh
        else:
            mesh = UnitSquareMesh(2**(i+2),2**(i+2))
    elif domain==2:
        if experiment ==True:
            mesh = mesh
        else:
            mesh = RectangleMesh(-0.5,-0.5,0.5,0.5,20*2**(i+2),20*2**(i+2),'left')#RectangleMesh(-0.5,-0.5,0.5,0.5,4**(i+1),4**(i+1),'left')
            parameters['allow_extrapolation'] = True
    elif domain==3:
        if experiment == True:
            mesh = mesh
        else:
            mesh = RectangleMesh(0.5,0.5,1.0,1.0,2**(i+2),2**(i+2),'left')
            parameters['allow_extrapolation'] = True
    elif domain==4:
        mesh = mesh
        parameters['allow_extrapolation'] = True
    elif domain==5:
        mesh = mesh
        parameters['allow_extrapolation'] = True
    elif domain==6:
        mesh=mesh
        parameters['allow_extrapolation'] = True
    else:
        print("No domain with that associated value, fool!")



    #Redefining finite element spaces etc on refined mesh, as well as our solution U=[u,d2udxx,d2udxy,d2udxy,G[du0],G[du1],const] and our test function V=[v,v11,v12,v22,v1,v2,const].
    hmin = mesh.hmin()
    FES = FunctionSpace(mesh, solution_space,deg)
    FESH = FunctionSpace(mesh, Hessian_solution_space, deg)

#Including a constant test space, so we can include the zero integral of the solution. As well as the spaces for gradient recovery.
    Pk = FiniteElement("CG", mesh.ufl_cell(), deg)
    Con = FiniteElement("R", mesh.ufl_cell(), 0)
    S = FunctionSpace(mesh, Pk * Pk * Pk * Pk * Con)
    x = project(x0,FES)
    y = project(x1,FES)
    U = Function(S)
    V = TestFunction(S)
    
    #Splitting the solution and test function into it's separate components for defining our nonlinear problem.
    (uh, H00, H11, H01,c) = U
    (vh, phi00, phi11, phi01,d) = V
    
    #Defining unit normal for defining our nonlinear problem
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    
    #Defining our Hessian to make our nonlinear form more compact
    Hessuh = as_matrix([[H00, H01], [H01, H11]])
    
    #defining "toggleable" penalty parameter
    if PENALTY == True:
        penalty = h**2
    else:
        penalty = 1
    #defining nonlinear variational problem
    F = ( H00 * phi00 + uh.dx(0) * phi00.dx(0) )*dx(mesh) \
        + ( H11 * phi11 + uh.dx(1) * phi11.dx(1) )*dx(mesh) \
        + ( H01 * phi01 + uh.dx(0) * phi01.dx(1) )*dx(mesh) \
        - (phi00 * n[0]) * (uh.dx(0)) * ds(mesh) \
        - (phi11 * n[1]) * (uh.dx(1)) * ds(mesh) \
        - (phi01 * n[1]) * (uh.dx(0)) * ds(mesh) \
        + (-determinant(Hessuh)+f*g(uh.dx(0),uh.dx(1))) * vh * dx(mesh)\
        + (c*vh+uh*d)*dx(mesh)\
        + (penalty)*(unitcircledist(uh.dx(0),uh.dx(1))*vh)*ds(mesh)\
    
    #Defining our updated "initial" guesses, i.e. u_1,u_2,... etc.
    if i == 0:
        uh0 = 0.5*(x**2+y**2)
        H000 = Constant(1.0)
        H110 = Constant(1.0)
        H010 = Constant(0.0)
        C0 = C
    else:
        uh0 = project(uhold, FES)
        H000 = project(H00old, FESH)
        H110 = project(H11old, FESH)
        H010 = project(H01old, FESH)
        C0 = C
    
    #Defining our updated "initial" guess as one expression
    class InitConditions(Expression):
        def __init__(self,u0, H000,H110,H010C0):
            self.u0 = u0
            self.H000 = H000
            self.H110 = H110
            self.H010 = H010
            self.C0 = C0
        def eval(self,value,x):
            value[0] = self.u0(x)
            value[1] = self.H000(x)
            value[2] = self.H110(x)
            value[3] = self.H010(x)
            value[4] = self.C0(x)
        def value_shape(self):
            return(5,)
    
    #Setting our updated "initial" guess for use.
    init_guess = as_vector([uh0,H000,H110,H010,C0])
    
    #Projecting our "initial" guess on the finite element space
    U0 = project(init_guess, S)
    
    #Defining our solution (which needs to be preset with an initial guess, which is then replaced when we apply the solver)
    U.assign(U0)
    
    #set log level, changes the amount of computation information given in the terminal, it does not affect the computation itself.
    PROGRESS = 16
    set_log_level(PROGRESS)
    
    #Setting some solver parameters.
    solver_parameters = NonlinearVariationalSolver.default_parameters()
    solver_parameters['newton_solver']['maximum_iterations'] = 20
    solver_parameters['newton_solver']['relative_tolerance'] = tol
    solver_parameters['newton_solver']['absolute_tolerance'] = tol
    solver_parameters['newton_solver']['relaxation_parameter'] = 1.0
    solver_parameters['newton_solver']['linear_solver'] = 'lu'
    solver_parameters['newton_solver']['preconditioner'] = 'default'
    
    #solving nonlinear problem
    solve(F == 0, U , solver_parameters=solver_parameters)
    
    #splitting up solution for use in updating "initial" guesses.
    (uh, H00, H11, H01,C) = U
    
    #Generating more visual results (for use in paraview)
    #sol_file << project(uh,fes)
    #errorfile << project(u-uh,fes)
    #det_file << project(f,fesW)
    
    #Defining error norms, when we put assemble(...) that integrates the input, that must be written as function*dx. (or *ds or *dS)
    erroru = (uh - u)**2*dx(mesh)
    errorp = (pow(grad(uh)[0] - du0,2) + pow(grad(uh)[1] - du1,2))*dx(mesh)
    error3 = (pow(H00 - d2udxx,2) + 2*pow(H01 - d2udxy,2) + pow(H11 - d2udyy,2))*dx(mesh)
    
    
    # Calculate the sqrt of the integral of the differential forms above
    eu = sqrt(assemble(erroru))
    ep = sqrt(assemble(errorp)) # gradient error
    e3 = sqrt(assemble(error3))
    
    # Store them in a vector of size itermax
    e_L2.append(eu)
    e_H1.append(ep)
    e_D2.append(e3)
    
    # Compute the number of degrees of freedom on the current mesh. For our mixed method it's dim(V) + d^2*dim(W)
#ndof.append(U.vector().array().size/4)
#one.append(10*U.vector().array().size**(-0.5)/4)
    
    # Restart the time counter
    t = time()
    
    #Defining updated "initial" guesses
    uhold = uh
    H00old = H00
    H11old = H11
    H01old = H01
    hm[i] = mesh.hmax()
    i = i+1

EOCL2 = []
EOCH1 = []
EOCH2 = []
EOCD2 = []

EOCL2.append(0)
EOCH1.append(0)
EOCH2.append(0)
EOCD2.append(0)

#Calcuating error orders of convergence.
for k in range(1,len(e_L2)):
    EOCL2.append(ln(e_L2[k-1]/(e_L2[k]))/ln(hm[k-1]/hm[k]))
    EOCH1.append(ln(e_H1[k-1]/(e_H1[k]))/ln(hm[k-1]/hm[k]))
    EOCD2.append(ln(e_D2[k-1]/(e_D2[k]))/ln(hm[k-1]/hm[k]))

k = 0
e = zeros([k,2])
for k in range(1,len(e_L2)):
    #    print("Number of DOFs = ", ndof[k])
    print("||u - u_h||_0 = ", e_L2[k], "   EOC = ", EOCL2[k])
    print("| u - u_h |_1 = ", e_H1[k], "   EOC = ", EOCH1[k])
    print("||D^2u - H_h[u_h]||_0=", e_D2[k], "   EOC = ", EOCD2[k])

