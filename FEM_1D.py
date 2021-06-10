
"""
Python Code to Solve 1-D Elastic Bar using FEM

Developed by Philip Luke K, as part of MIN-552 Coursework submission at IIT-Roorkee, October 2020

THIS IS A GENERALIZED FEM CODE TO SOLVE A 1-D LINEAR-ELASTIC BAR SUBJECTED TO AXIAL LOADS. Key Features of the code include :

	1. Discretization into any number of elements.

	2. Incorporates Linear and Quadratic elements.

	3. Ability to handle any type of loading, boundary conditions,
		c/s area and elasticity along the length.

	4. Graphical representation of the primary mesh (structure and 		elements).

	5. Graphical representation of the variation of c/s area.

	6. Support conditions can be any combination of 'Free' and 			'Fixed'.

	7. Displacements and Reactions at all nodes in the domain can 		be obtained.

	8. Numerical value of displacement, strain and stress at any 		point in the domain can be found out.

	9. Graphical representation of the variation of Shape and Deformation Functions for the Element specified.

User Input :

	(A) Main Function
			1. Length of the structure
			2. No:of elements in the mesh
			3. Value of traction (if any)
			4. Type of Structure
			5. Type of element in the mesh
            6. Type of Loading

	(B)	Class 'System'
			1. C/S Area along length
			2. Elasticity along length

	(C) Class 'BoundaryConditions'
			1. Displacement Boundary Conditions
			2. Traction Boundary Conditions

	(D) Class 'Loading'
			1. Value(s) of external load along the length


For further detailed info, the user is requested to view the documentation of each class/method.
To view the Documentation of any class/method, user may execute the command <class_name.__doc__> and/or <class_name.method_name.__doc__>

Comments has been provided wherever necessary.

"""


# IMPORTING ESSENTIAL LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import math

# GLOBAL VARIABLES

n = 100 # No:of points for integration
N = n + 1

# CLASS DEFINITIONS GO FROM HERE

class Vertex :
    """Class used to define the X Co-ordinates of the nodes of element/mesh/structure"""

    def __init__(self, xcoord) :
        """Takes the position of element/structure and generate it as Co-ordinate for the Edge class to calculate the element/structure length"""

        self.xcoord = xcoord

    def get_coordinate(self) :
        """Prints the Co-ordinate of the node.
        Optional Function"""

        print("X Cordinate of this point is :",self.xcoord)
        return self.xcoord

class Edge :
    """Class used to define each element of the mesh"""

    def __init__(self, v1, v2) :
        """Takes in two objects of Class Vertex and assigns to First and Last coordinate of the element"""

        self.coord1 = v1.xcoord # Coordinate of the first node
        self.coord2 = v2.xcoord # Coordinate of the end node

    def ele_length(self) :
        """Calculates the length of each element by subtracting first coordinate from the end coordinate
   		Returns the element length"""

        self.length = self.coord2 - self.coord1 # Length of each element
        #print("Length of the element is : ",self.length)
        return self.length

class Mesh :
    """Class used to generate the Finite Element Mesh for the structure """

    def __init__(self, origin_vertex, end_vertex, struct, no_ele) :
        """Uses Vertex objects and Edge object to calculate the length of the structure based on the coordinates of the origin and end. Takes in the no:of elements defined by user, to generate mesh"""

        self.origin = origin_vertex.xcoord
        self.end = end_vertex.xcoord
        self.no_ele = no_ele
        self.len = struct.ele_length()
        print("Length of the Structure is : ", self.len, "units")

    def mesh_generate(self) :
        """Generates FE Mesh based on the no:of elements and structure length. """

        self.ele_len = self.len/self.no_ele # Calculates Element Length
        print("Length of each element of the mesh is : ", self.ele_len, "units")

        self.xcoord_incr = self.origin
        self.mesh = []
        while(self.xcoord_incr < self.end) :
            self.mesh.append(self.xcoord_incr)
            self.xcoord_incr = self.xcoord_incr + self.ele_len
        self.mesh.append(self.xcoord_incr)

        print("Meshed points for the 1-D structure is: ", self.mesh) # Prints the FEMesh Coordinates
        plt.axis('off')
        plt.suptitle("1-D structure and Mesh")
        plt.plot(self.mesh, np.ones(len(self.mesh)), marker='o') # Graphical Representation of the FEMesh

        return self.mesh, self.ele_len

class System :
    """Class used to Define System Parameters, Area and Elasticity over the mesh"""

    def __init__(self, mesh, no_ele) :
        """Initializes the class with Mesh details"""

        self.mesh = mesh
        self.no_ele = no_ele

    def area(self) :
        """Defines and returns a vector having Area corresponding to each element in the mesh.
        Can be of 3 types

            1. Constant
                    C/S area is constant throughout the length. Value is hard-coded inside the function

            2. Linear
                    C/S area varies linearly throughout the length. User needs to specify the value at the first and last point

            3. User Defined
                    C/S area at the end nodes of each element specified by the user.

            The method also give a graphical representation of the variation of area over the length.

            """
        self.area_type = "Constant" # Constant, Linear, User-Defined

        if (self.area_type == "Constant") :
            self.area_value = 5.0 # Value of constant area of c/s
            self.mesh_area = np.ones(len(mesh))*self.area_value

        elif (self.area_type == "Linear") :
            self.area_value1 = 1.0 # Value of area at origin
            self.area_value2 = 5.0 # value of area at end
            self.mesh_area = np.linspace(self.area_value1, self.area_value2, len(self.mesh))

        elif (self.area_type == "User-Defined") :
            print("Enter c/s area at end nodes of each element. Use same value at connected nodes")
            self.mesh_area = np.ones(len(mesh))
            for i in range(len(self.mesh)-1) :
                print("Element",i+1)
                self.mesh_area[i] = input("First Node")
                self.mesh_area[i+1] = input("Second Node")

        self.ele_area = np.zeros(self.no_ele)
        for i in range(self.no_ele) :
            self.ele_area[i] = (self.mesh_area[i] + self.mesh_area[i+1])/2 # averaging the area at end nodes to obtain element-wise are

        return self.mesh_area, self.ele_area # area of each element in the mesh

    def plot_area(self, mesh_area) :
        """Plots the variation of area of c/s along the length of the structure.

        Optional Function"""

        plt.figure()
        plt.axis('off')
        plt.suptitle("c/s Area of the structure : ")
        plt.plot(self.mesh_area,'k')
        plt.plot(-self.mesh_area,'k')


    def elasticity(self) :
        """Defines and returns a vector having Elastic Modulus corresponding to each element in the mesh.
        Can be of 2 types

            1. Constant
                    Elasticity is constant throughout the length. Value is hardcoded inside the function


            2. User Defined
                    Elasticity for each element specified by the user.

            """

        self.elast_type = "Constant" # Constant, User-Defined
        self.ele_elast = np.ones(self.no_ele)

        if (self.elast_type == "Constant") :
            self.elast_value = 200000000000.0 # Value of constant Elasticity
            self.ele_elast *= self.elast_value

        elif (self.elast_type == "User-Defined") :
            print("Enter values of Elastic Modulus for each element.")
            for i in range(len(self.ele_elast)) :
                print("\nElement ",)
                self.ele_elast[i] = input(i+1)

        return self.ele_elast # elasticity of each element in the mesh

class BoundaryCondition :
    """Class used to specify the complete Boundary Condition for the particular FE Problem

    The user has the major control over B.C's."""

    def __init__(self, ele_type, no_ele, struct_type, traction) :
        """Initializes the class with the problem background. The type of element, no: of elements. elemental DOF, type of structure and the value of traction (if any)

        Calculates the Global DOF"""

        self.no_ele = no_ele
        self.ele_type = ele_type
        if self.ele_type == "Linear" :
            self.ele_dof = 2
            self.gl_dof = self.no_ele + 1
        elif self.ele_type == "Quadratic" :
            self.ele_dof = 3
            self.gl_dof = (self.no_ele*self.ele_dof) - (self.no_ele-1)
        else :
            quit()

        self.disp_bc = np.ones(self.gl_dof)
        self.traction_bc = np.zeros(self.gl_dof)
        self.struct_type = struct_type
        self.traction_value = traction
        self.residue = np.zeros(self.gl_dof)

    def get_bc(self) :
        """Returns the Displacement, Traction Boundary Conditions and the Residue Vector based on structure type.
        Residue vector - non-zero where displacement BC is specified

        Structure Type may be of 5 :
            1. Fixed-Free : Left end fixed, right end free
                Displacement at node 1 is 0.
                Traction (if any) is specified at the very last node
                Residue will be non-zero (here assigned with 1) at node 1

            2. Free-Fixed : Left end free, right end fixed
                Displacement at last node is 0.
                Traction (if any) is specified at node 0
                Residue non-zero at last node

            3. Fixed-Fixed : Both ends fixed
                Displacement at first and last node is 0
                Residue is non-zero at first and last node
                No traction B.C valid

            4. Free-Free : Both ends free (unstable)
                Displacement B.C is not valid
                Residue 0 at every node
                Traction may be at both first and last node

            5. User-Defined :
                User has to type in the values of displacement and traction at every node in the mesh
                Should comply with theory of Mechanics to obtain logical solution.
                Traction and Displacement B.C shouldn't be specified together at any node.
                In case of fixity, enter 0 for Displacement B.C. Residue vector will be self calculated
                based on that
        """

        if self.struct_type=="Fixed-Free" :

            self.disp_bc[0] = 0
            self.traction_bc[-1] = self.traction_value
            self.residue[0] = 1

        elif self.struct_type=="Free-Fixed" :

            self.disp_bc[-1] = 0
            self.traction_bc[0] = self.traction_value
            self.residue[-1] = 1

        elif self.struct_type=="Fixed-Fixed" :

            self.disp_bc[0] = 0
            self.disp_bc[-1] = 0
            self.residue[0] = 1
            self.residue[-1] = 1

        elif self.struct_type=="Free-Free" :

            self.traction_bc[0] = self.traction_value
            self.traction_bc[-1] = self.traction_value

        elif self.struct_type=="User-Defined" :

            print("Enter Displacement Boundary Condition at Each Co-ordinate. ")
            self.inp = np.arange(1,(self.gl_dof+1),1)

            for i in range(len(self.inp)) :
                print(":\n")
                self.disp_bc[i] = input(self.inp[i]) # Prompts the user to input displacement at each node
            print("Co-ordinates and corresponding Displacement Boundary Condition of the mesh is : \n", self.inp, "\n", self.disp_bc)
            self.flag = np.where(self.disp_bc==0)
            self.residue[self.flag] = 1 # residue value = 1, where displacement BC = 0

            print("Enter Traction Boundary Condition at Each Co-ordinate. Avoid the nodes where displacement Boundary Condition is already given")

            for i in range(len(self.inp)) :
                print(":\n")
                self.traction_bc[i] = input(self.inp[i]) # Prompts the user to input traction at each node
            print("Co-ordinates and corresponding Displacement Boundary Condition of the mesh is : \n", self.inp, "\n", self.traction_bc)

        return self.ele_dof, self.gl_dof, self.disp_bc, self.traction_bc, self.residue

class ShapeFunction :
    """Class used to generate the Shape Function (N), Deformation Function (B) based on the type of element and display the graphical variation of N and B over an element"""

    def __init__(self, ele_type, ele_len) :

        """Initializes the class by attributing the element type assigned, length of the element and degree of freedom for the particular element"""

        self.ele_type = ele_type
        self.ele_len = ele_len
        self.ele_dof = 2 if self.ele_type=="Linear" else 3 if self.ele_type=="Quadratic" else 0
        #self.B = np.zeros(self.ele_dof)
        #self.N = np.zeros(self.ele_dof)

    def get_shape_function(self, x) :
        """Generates and returns the shape function vector (N) based on the type of element. Takes in the position (x) and calculation using the already known component functions.
        Displays an error message if neither Linear/Quadratic element and exits."""

        self.ele_pos = x # position on which calculation is done. varies from 0 - element length

        if self.ele_type == "Linear" :

            self.N1 = 1 - (self.ele_pos/self.ele_len)
            self.N2 = self.ele_pos/(self.ele_len)
            self.N = np.array([[self.N1], [self.N2]])

        elif self.ele_type == "Quadratic" :

            self.N1 = 1 - (3*self.ele_pos/self.ele_len) + (2*(self.ele_pos**2)/(self.ele_len**2))
            self.N2 = (4*self.ele_pos/self.ele_len) - (4*(self.ele_pos**2)/(self.ele_len**2))
            self.N3 = (-self.ele_pos/self.ele_len) + (2*(self.ele_pos**2)/(self.ele_len**2))
            self.N = np.array([[self.N1], [self.N2], [self.N3]])

        else :
            print("Undefined Element Type. Program Terminating...")
            exit()

        return self.N

    def get_deformation_function(self, x) :
        """Generates and returns the Deformation function vector (B) based on the type of element. Takes in the position (x) and calculation using the already known component functions
        Displays an error message if neither Linear/Quadratic element and exits."""

        self.ele_pos = x # position on which calculation is done. varies from 0 - element length

        if self.ele_type == "Linear" :

            self.B1 = -1/(self.ele_len)
            self.B2 = 1/(self.ele_len)
            #self.B = np.append(self.B1, self.B2)
            self.B = np.array([[self.B1], [self.B2]])

        elif self.ele_type == "Quadratic" :

            self.B1 = (-3/self.ele_len) + (4*self.ele_pos/(self.ele_len**2))
            self.B2 = (4/self.ele_len) + (-8*self.ele_pos/(self.ele_len**2))
            self.B3 = (-1/self.ele_len) + (4*self.ele_pos/(self.ele_len**2))
            self.B = np.array([[self.B1], [self.B2], [self.B3]])

        else :
            print("Undefined Element Type. Program Terminating...")
            exit()

        return self.B

    def plot_shape_function(self) :
        """Plots the variation of Shape and Deformation function over an element graphically
        Optional Function"""

        self.x = np.linspace(0, self.ele_len, 100)
        self.N_value = np.zeros([self.ele_dof,1, len(self.x)])
        self.B_value = np.zeros([self.ele_dof,1, len(self.x)])
        for i in range(len(self.x)) :

            self.N_value[:,:,i] = self.get_shape_function(self.x[i])
            self.B_value[:,:,i] = self.get_deformation_function(self.x[i])

        self.x = self.x.reshape(1,100)
        fig_N,axs_N = plt.subplots()
        fig_N.suptitle('Variation of Shape Function across each element')
        axs_N.axis([0, self.ele_len, 0, 1.0])
        fig_B,axs_B = plt.subplots()
        fig_B.suptitle('Variation of Deformation Function')
        #axs_B.axis([0, self.ele_len, 0, 1.0])

        for i in range(len(self.N_value)) :
            axs_N.plot(self.x.transpose(), self.N_value[i].transpose())
            axs_B.plot(self.x.transpose(), self.B_value[i].transpose())

        return None

class Loading :
    """Class used to define external load in the structure"""

    def __init__(self, load_type, mesh) :
        """Initializes the class with load_type specified by the user and the mesh generated"""

        self.load_type = load_type

        if self.load_type == "Discrete" :
            # Values and position of discrete forces in the domain. To be entered by the user
            self.load_pos = np.array([0.33, 0.66]) # position (s) of the load in the domian
            self.load_value = np.array([3.0, -1.0]) # value of load (s) corresponding to the positions given above
            self.flag = 0

    def get_point_load(self, x, ele_len, ele_flag, dx) :
        """Returns the value of body force and integration length at any given point in the domain. Used for integrating over an element to obtain elemental load vector. Four types of loading cases are acceptable :

            1. Uniform ( b(x) = k ) - magnitude of load is constant throughout the domain. User need to enter the value of k manually.

            2. Linear ( b(x) = ax + c ) - magnitude of load varies linearly throughout the domain. User need to enter the values for a and c.

            3. Quadratic ( b(x) =  ax^2 + cx + d) - magnitude of load varies quadratically. User needs to enter values for a, c and d.

            4. Discrete - If the structure consists of discrete load only. The position and values, entered in the initial method will be taken up.

        For any other type of complex/irregular loading, seperate function need be written.
        """

        self.pos = (ele_flag-1)*ele_len + x # position of the point in the domain

        if self.load_type == "Uniform" :

            self.const_load = 10.0 # Value of constant load
            return self.const_load, dx

        elif self.load_type == "Linear" :

            self.b_a = 5.0 # Value of a in ax + c
            self.b_c = 0.0 # Value of c in ax + c
            self.linear_load = self.b_a*self.pos+ self.b_c
            return self.linear_load, dx

        elif self.load_type == "Quadratic" :

            self.b_a = 10.0 # Value of a in ax^2 + cx + d
            self.b_c = 0.0 # Value of c in ax^2 + cx + d
            self.b_d = 0.0 # Value of d in ax^2 + cx + d
            self.quad_load = self.b_a*self.pos*self.pos + self.b_c*self.pos + self.b_d
            return self.quad_load, dx

        elif self.load_type == "Discrete" :

            for i in range(len(self.load_pos)) :
                # Loop makes sure that the discrete body load is returned only once for the smooth integration over the element in ElementLoadMatrix class.

                if self.pos-dx < self.load_pos[i] < self.pos+dx :
                    if self.flag == i :
                        self.b = self.load_value[i]
                        self.flag += 1
                        break
                    else :
                        self.b = 0

                else:
                    self.b = 0

            return self.b, 1

        else :
            return 0

class ElementStiffnessMatrix :
    """Class used to generate and return the stiffness matrix of any element in the mesh"""

    def __init__(self, ele_sf, area, elast, ele_dof) :
        """Initializes the class with the element geometric and material properties"""

        self.ele_area = area # Area of the element
        self.ele_elast = elast # Elastic Modulus of the element
        self.ele_len = ele_sf.ele_len # Length of the element
        self.x = np.linspace(0, self.ele_len, N) # Divides the element into 100 points for integration
        self.dx = self.ele_len/N # length of one division
        self.ele_dof = ele_dof # Elemental DOF

        self.k = np.zeros([self.ele_dof,self.ele_dof])

    def get_ele_stiff(self) :
        """Generates and returns the element stiffness matrix by integrating (B)^T.A.E.B over the element

        At each division of the element, ShapeFunction is called to get B. Values at all points are then added together. Thereby integration is achieved"""

        for i in range(len(self.x)) :
            self.B = ele_sf.get_deformation_function(self.x[i])
            self.Bt = self.B.transpose()
            self.k += (self.B@self.Bt)*self.ele_area*self.ele_elast*self.dx

        return (self.k)

class ElementLoadMatrix :
    """Class used to generate and return the force matrix of any element in the mesh"""

    def __init__(self, ele_sf, struct_load, area, ele_dof) :
        """Initializes the class with the element geometric and material properties

        Linear Distribution of Load over an element is assumed. For other kinds of distribution, separate functions/classes need to be written"""

        self.ele_len = ele_sf.ele_len # Length of the element
        self.ele_area = area # Area of the element
        self.x = np.linspace(0, self.ele_len, N) # Divides the element into 100 points for integration
        self.dx = self.ele_len/N # length of one division
        self.ele_dof = ele_dof # Elemental DOF

        self.f = np.zeros([self.ele_dof,1])

    def get_ele_load(self, ele_traction_bc, ele_flag) :
        """Generates and returns the element load matrix by integrating (N)^T.b over the element

        At each division of the element, ShapeFunction and Loading are called to get N and b at that point. Values at all points are then added together. Thereby integration is achieved

        If at either the origin/end element, traction is specified, the effect is deduced from the load at that node. It is assumed that the user provides traction B.C sensible to Mechanics theory."""

        for i in range(len(self.x)) :
            self.N = ele_sf.get_shape_function(self.x[i])
            self.b, self.dx1 = struct_load.get_point_load(self.x[i], self.ele_len, ele_flag, self.dx)
            self.f += (self.N)*self.b*self.dx1

        if any(ele_traction_bc) :
            self.tra_pos = np.where(ele_traction_bc != 0)
            self.tra_val = ele_traction_bc[self.tra_pos[0]]
            self.f -= self.N*self.ele_area*self.tra_val

        return (self.f)

class Solver :
    """Class used to solve the Finite Element formulation and return displacement and reaction vectors"""

    def __init__(self, ele_type, no_ele, ele_dof, gl_dof, k, f) :
        """Initializes the class with stiffness matrices and force vectors for all elements, element type and no:of elements for the given problem
        Calculates the global DOF"""

        self.ele_type = ele_type
        self.no_ele = no_ele
        self.ele_dof = ele_dof
        self.gl_dof = gl_dof

        self.ele_k = k # multi-dimensional array having all element stiffness matrices
        self.ele_f = f # multi-dimensional array having all element force matrices

    def get_global_stiffness(self) :
        """Generates and returns the Global stiffness matrix (K) for the given system"""

        self.gl_stiff = np.zeros([self.gl_dof, self.gl_dof])

        for i in range(self.no_ele) :
            # Loop goes over the stiffness matrix of each element and joins the values of connected nodes as well.

            stiff = self.ele_k[:,:,i]

            k = np.zeros(self.ele_dof)
            #k = []
            for j in range(self.ele_dof) :

                k[j] = (self.ele_dof-1)*i + j
                #temp = (self.ele_dof-1)*i + j
                #k.append(temp)

            r,s = 0, 0
            for p in k :
                for q in k :
                    p, q = int(p), int(q)
                    self.gl_stiff[p, q] += stiff[r, s]
                    s += 1
                r += 1
                s=0

        return self.gl_stiff

    def get_global_force(self) :
        """Generates and returns the Global Force vector (F) for the given system"""


        self.gl_force = np.zeros([self.gl_dof,1])

        for i in range(self.no_ele) :

            force = self.ele_f[:,:,i]

            k = np.zeros(self.ele_dof)
            for j in range(self.ele_dof) :
            # Loop goes over the stiffness matrix of each element and joins the values of connected nodes as well.
                k[j] = (self.ele_dof-1)*i + j

            r = 0
            for p in k :
                p = int(p)
                self.gl_force[p] += force[r]
                r += 1

        return self.gl_force

    def get_nodal_displacement(self, disp_bc, residue) :
        """
        Calculates and returns the Displacement vector based on the equation 'KU-F = R'

               Input : Displacement Boundary Condition, Residue Vector
               Output : Displacement vector

        """
        self.K = self.get_global_stiffness() # Global stiffness matrix
        #print("Global Stiffness Matrix : \n", self.K)
        self.F = self.get_global_force() # Global Force Vector
        #print("Global Force Vector : \n", self.F)

        self.U = np.zeros(self.gl_dof)

        self.disp_bc = disp_bc
        self.residue = residue

        self.fixeddof = np.where(self.residue == 1) # Generates the vector of nodes where displacement is bounded
        self.fixeddof = self.fixeddof[0]
        self.freedof = np.where(self.residue != 1) # Generates the vector of nodes where displacement is unbounded
        self.freedof = self.freedof[0]

        self.residue1 = np.delete(self.residue, self.fixeddof) # Residue vector for unknown nodal displacements
        self.F1 = np.delete(self.F, self.fixeddof) # Global force vector corresponding to unknown nodal displacements

        self.K1 = self.K
        for i in range(len(self.fixeddof)) :
            # The loop removes the rows and columns corresponding to restricted displacements
            # from the global Stiffness Matrix
            j = self.fixeddof[i]
            #self.K1 = self.K[j+1:,j+1:]
            self.K1 = np.delete(self.K1,j-i,0)
            self.K1 = np.delete(self.K1,j-i,1)

        self.U1 = np.linalg.inv(self.K1)@(self.residue1 + self.F1) # Solving K.U - F = R to get unknown displacement


        # Defines the global displacement vector and assigns calculated and known displacement
        self.U = np.zeros(self.gl_dof)
        self.U[self.freedof] = self.U1
        self.U[self.fixeddof] = self.disp_bc[self.fixeddof]

        # Defines the global residue(reaction) vector, calculates and assigns the reaction at fixed and free nodes
        self.R = np.zeros(self.gl_dof).reshape(self.gl_dof,1)
        self.R1 = self.K@self.U.reshape(self.gl_dof,1) - self.F # R = KU-F
        self.R[self.fixeddof] = self.R1[self.fixeddof]

        return self.U, self.R # returns the global displacement and reaction vector

class Results :
    """Class used to generate and return results

        1. Displacement at any point in the domain
        2. Strain at any point in the domain
        3. Stress at any point in the domain
    """

    def __init__(self, ele_len, ele_type, ele_dof, U, elast) :
        """Initialize the class with element and material properties and solved displacement vector"""

        self.ele_len = ele_len
        self.ele_type = ele_type
        self.ele_dof = ele_dof
        self.U = U
        self.elast = elast

    def get_ele_disp(self, x) :
        """Calculates and returns the element-displacement vector for any point in the domain

        Input : Position of the point in the domain
        Output : Displacement vector corresponding to the nodes of the element of the input point

        """

        self.ele_pos = x # position of the point from origin
        self.ele = math.ceil(x/self.ele_len) # element corresponding to the point
        self.ele_disp = np.zeros([self.ele_dof, 1])

        if self.ele_type == "Linear" :
            self.node1 = (self.ele*self.ele_dof) - (self.ele+1) # first node of the corresponding element
            self.node2 = (self.ele*self.ele_dof) - (self.ele) # end node of the corresponding element
            self.ele_disp = np.array([self.U[self.node1], self.U[self.node2]]) #displacement vector of the element

        elif self.ele_type == "Quadratic" :
            self.node1 = (self.ele*self.ele_dof) - (self.ele+2) # first node of the corresponding element
            self.node2 = (self.ele*self.ele_dof) - (self.ele) # end node of the corresponding element
            self.node3 = (self.ele*self.ele_dof) - (self.ele+1)    # second node of the corresponding element
            self.ele_disp = np.array([self.U[self.node1], self.U[self.node3], self.U[self.node2]]) #element displacement vector

        return self.ele_disp

    def get_displacement(self, x) :
        """Calculates and returns displacement at any point in the domain

        Input : Point in domain where displacement needs to be calculated
        Output : Displacement corresponding to the input point"""

        self.ele_pos = x
        self.ele_sf = ShapeFunction(self.ele_type, self.ele_len)
        self.ele_disp = self.get_ele_disp(self.ele_pos) # obtains element displacement vector
        self.pos = self.ele_pos - (self.ele - 1)*self.ele_len # position of point from the first node
        self.N_pos = self.ele_sf.get_shape_function(self.pos)
        self.pos_disp = 0
        for i in range(len(self.ele_disp)):
            self.pos_disp += self.ele_disp[i]*self.N_pos[i]
        #self.pos_disp = self.ele_disp[0] + (self.ele_disp[len(self.ele_disp)-1] - self.ele_disp[0])*self.pos/self.ele_len # interpolation

        return self.pos_disp

    def get_strain(self, x) :
        """Calculates and returns strain at any point in the domain

        Input : Point in domain where strain needs to be calculated
        Output : Strain corresponding to the input point"""

        self.ele_pos = x
        self.ele_disp = self.get_ele_disp(self.ele_pos) # obtains element displacement vector (d)

        self.pos = self.ele_pos - (self.ele - 1)*self.ele_len # position of point from the first node
        self.ele_sf = ShapeFunction(ele_type, ele_len) # obtaining deformation function (B)
        self.pos_strain = self.ele_sf.get_deformation_function(self.pos).transpose()@self.ele_disp # strain,e = B*d

        return self.pos_strain

    def get_stress(self, x) :
        """Calculates and returns strain at any point in the domain

        Input : Point in domain where strain needs to be calculated
        Output : Strain corresponding to the input point"""

        self.ele_pos = x
        self.pos_strain = self.get_strain(self.ele_pos) # obtains strain (e) at the point
        self.ele_elast = self.elast[self.ele-1] # obtains elastic modulus (E) at that point
        self.pos_stress = self.pos_strain*self.ele_elast # stress, s = E*e

        return self.pos_stress

"""

MAIN FUNCTION

Execution of the FE-Method go here. This is achieved by creating objects for the classes defined above.

For user-defined parameters that has multiple options, the options are mentioned as comments nearby the code.

"""

# System Properties

length = 10.0

struct_type = "Fixed-Fixed" # Fixed-Free, Free-Fixed, Fixed-Fixed

load_type = "Discrete" # Uniform, Discrete, Linear, Quadratic

traction = 0.0


no_ele = 5 # Number of elements
ele_type = "Linear" # Linear, Quadratic

bc = BoundaryCondition(ele_type, no_ele, struct_type, traction)
[ele_dof, gl_dof, disp_bc, traction_bc, residue] = bc.get_bc() # Obtains the boundary conditions for the problem

# ele_dof = 2 or 3
# disp_bc = np.array([])
# traction_bc = np.array([])
# residue = np.array([])
# User may define  completely arbitrary Boundary Conditions above, but compatible with the no:of elements and element type

origin = Vertex(0)
end = Vertex(length)
struct = Edge(origin, end)
struct_mesh = Mesh(origin,end,struct,no_ele)
mesh, ele_len = struct_mesh.mesh_generate()  # generates the FE Mesh for the problem.

# mesh = np.array([])
# User may define a completely arbitrary mesh above, but compatible with the no:of elements

sys = System(mesh, no_ele)
msh_area, area = sys.area()
elast = sys.elasticity()

# area = np.array([])
# elast = np.array([])
# User may define a completely arbitrary area/elasticity array above, but compatible with the no:of elements

sys.plot_area(msh_area) # Plots the variation of area along the length

struct_load = Loading(load_type, mesh)

ele_k = np.zeros([ele_dof, ele_dof, no_ele]) # multi-dimensional array to store all element-stiffness matrices
ele_f = np.zeros([ele_dof,1,no_ele])# multi-dimensional array to store all element-force vectors

for i in range(no_ele) :
    # The loop travels through each element in the mesh and generates element stiffness matrix and element force vector which is stored in multi-dimensional arrays k and f respectively.
    ele_flag = i +1 # element number in this loop
    x0 = Vertex(mesh[i])  # First coordinate of the element in this loop
    xl = Vertex(mesh[i+1]) # End coordinate of the element in this loop
    ele = Edge(x0, xl)
    ele_len = ele.ele_length() # Length of the element

    ele_sf = ShapeFunction(ele_type, ele_len)

    ele_stiff = ElementStiffnessMatrix(ele_sf, area[i], elast[i], ele_dof)
    ele_k[:,:,i] = ele_stiff.get_ele_stiff()
    print("\nStiffness matrix of element", i+1, "\n", ele_k[:,:,i])

    ele_force = ElementLoadMatrix(ele_sf, struct_load, area[i], ele_dof)
    ele_f[:,:,i] = ele_force.get_ele_load(traction_bc[i*(ele_dof-1): i*(ele_dof-1)+ele_dof], ele_flag)
    print("\nForce vector of element", i+1, "\n", ele_f[:,:,i])

# Outside the loop.

ele_sf.plot_shape_function() # Plots the variation of shape and deformation functions over the element

solve = Solver(ele_type, no_ele, ele_dof, gl_dof, ele_k, ele_f) # Solver class is called to generate global K and global F.

U, R = solve.get_nodal_displacement(disp_bc, residue) # Displacement vector and residue/reaction vector

print("Displacement Vector is : ",U)
print("Reaction Vector is : ", R)

res = Results(ele_len, ele_type, ele_dof, U, elast) # To generate results

x = np.linspace(0,length,201) # set of points in the domain where results need to be found out
x1 = 0.5 # point in the domain where results need to be found out
stress =np.zeros(len(x))
strain = np.zeros(len(x))
disp = np.zeros(len(x))

print(res.get_displacement(x1)) # Prints displacement value at x1
print(res.get_strain(x1)) # Prints strain value at x1
print(res.get_stress(x1)) # Prints stress value at x1


for i in range(len(x)) :
    # Collects results at all the points in x array
    disp[i] = res.get_displacement(x[i])

plt.figure()
plt.xlabel("length of the domain")
plt.ylabel("displacement (m)")
plt.suptitle("2 linear element ")
plt.plot(x, disp)



#####################################################################
############################ END OF CODE ############################
#####################################################################