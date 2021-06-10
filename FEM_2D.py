"""
Python Code to Solve 2-D Linear-Elastic Domain using FEM

Developed by Philip Luke K, as part of MIN-552 Coursework submission at IIT-Roorkee, December 2020

THIS IS A GENERALIZED FEM CODE TO SOLVE A 2-D LINEAR-ELASTIC DOMAIN SUBJECTED TO EXTERNAL LOADS. Key Features of the code include :

	1. Discretization into any number of elements in x and y directions.

	2. Incorporates Rectangular, Quadrilateral and Triangular elements in linear and quadratic form

	3. Ability to handle any type of loading, traction and boundary conditions on supports as well as domain edges

	4. Automated mesh generation for Quadrilateral domains

	6. Support and/or edge restraints can be user defined.

	7. Displacements and Reactions at all nodes in the domain can be obtained.

	8. Numerical value of displacement, strain and stress at any point in the domain can be found out.

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
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from sympy import solve
from sympy import Symbol



# CLASS DEFINITIONS GO FROM HERE

class Vertex : 
    """
    Class used to define the X and Y Co-ordinates of the nodes of element/mesh/structure
    """

    def __init__(self, xcoord, ycoord) :
        """
        Takes the positions of element/structure and generate it as Co-ordinates
        """
        self.xcoord = xcoord
        self.ycoord = ycoord

    def get_coordinate(self) :
        """
        Prints the Co-ordinate of the node.
        Optional Function
        """

        print("X Cordinate of this point is :",self.xcoord)
        print("Y Cordinate of this point is :",self.ycoord)
        return self.xcoord, self.ycoord


class Edge() :
    """
    Class used to define geometric parameters of domain/element of the mesh
    """

    def __init__(self, v1, v2, v3, v4) :
        """
        Takes in four objects of Class Vertex 
        Calculates edge lengths of the domain/element
        """
    
        self.edge1 = np.sqrt((v2.xcoord-v1.xcoord)**2 + (v2.ycoord - v1.ycoord)**2)
        self.edge2 = np.sqrt((v3.xcoord-v2.xcoord)**2 + (v3.ycoord - v2.ycoord)**2)
        self.edge3 = np.sqrt((v4.xcoord-v3.xcoord)**2 + (v4.ycoord - v3.ycoord)**2)
        self.edge4 = np.sqrt((v1.xcoord-v4.xcoord)**2 + (v1.ycoord - v4.ycoord)**2)
        
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.v4 = v4

    def ele_area(self) :
        """
        Calculates the area of domain/element
   		Returns the element area
        Area of Polygon = |(x1y2-y1x2) + (x2y3-y3x3) + ... + (xny1-ynx1)|/2 
        """
        self.area = abs(((self.v1.xcoord*self.v2.ycoord - self.v1.ycoord*self.v2.xcoord) + 
                         (self.v2.xcoord*self.v3.ycoord - self.v2.ycoord*self.v3.xcoord) + 
                         (self.v3.xcoord*self.v4.ycoord - self.v3.ycoord*self.v4.xcoord) + 
                         (self.v4.xcoord*self.v1.ycoord - self.v4.ycoord*self.v1.xcoord))/2)
        
        #print("Area of the element is : ",self.area)
        return self.area


class QuadrilateralMesh :
    """
    Class used to automate quadrilateral-mesh generation and node numbering of the domain 
    
    Only Quadrilateral domain is completely discretized into quadrilateral elements.
    
    Error obtained when triangular domain is input.
    """

    def __init__(self, nel_x, nel_y, ele_node) :
        """ 
        Initializes the class with mesh properties. 
        """
        
        self.nel_x = nel_x
        self.nel_y = nel_y
        self.nel = nel_x*nel_y
        self.ele_node = ele_node
        
        self.gl_nodes = (nel_x+1)*(nel_y+1) # no of corner nodes 
        
        self.mesh_coord = np.zeros([self.gl_nodes, 2]) # coordinates of global cornder nodes
        self.ele_end_coord = np.zeros([4, 2, self.nel]) # coordinates of outer nodes of each element
        self.node_number = np.zeros([self.ele_node,self.nel]) # node number of each node of each element

    def mesh_generate(self, v1, v2, v3, v4) :
        """Generates the quadrilateral-mesh of the domain
        
        Returns 1) array with coordinates of global corner nodes
                2) array with coordinates of corner-nodes of each element
                3) array with the node number of each node of each element in the mesh
        
        Element coordinates and numbering : from left bottom in anticlockwise, 1-2-4-3 order obtained. 
        """
        
        if v1.xcoord==v4.xcoord and v1.ycoord==v4.ycoord :
            print("Triangular Domain cannot be discretized purely into ", self.ele_type," elements. Try again.")
            exit()
            
        self.x1 = v1.xcoord
        self.y1 = v1.ycoord
        
        self.x2 = v2.xcoord
        self.y2 = v2.ycoord
        i = 0
        
        self.x = v1.xcoord
        self.y = v1.ycoord

        while(self.x<=v3.xcoord and self.y<=v3.ycoord):
            # Loop travels around the domain and assigns the coordinates of global corner nodes to mesh_coord
            
            self.mesh_coord[i, 0] = self.x
            self.mesh_coord[i, 1] = self.y
            
            self.dx = (self.x2 - self.x1)/self.nel_x
            self.dy = (self.y2 - self.y1)/self.nel_x

            
            self.x = self.x + self.dx
            self.y = self.y + self.dy
            
            if self.x > self.x2 :
                
                self.x1 = self.x1 + (v4.xcoord-v1.xcoord)/self.nel_y
                self.y1 = self.y1 + (v4.ycoord-v1.ycoord)/self.nel_y
            
                self.x2 = self.x2 + (v3.xcoord-v2.xcoord)/self.nel_y
                self.y2 = self.y2 + (v3.ycoord-v2.ycoord)/self.nel_y
                
                self.x = self.x1
                self.y = self.y1

            i = i+1

        k=0
        
        for i in range(self.nel_y) :
            # Loop travels around the domain and assigns the coordinates and node number of each node of each element

           
            for j in range(self.nel_x) :
                self.n1 = i*(self.nel_x+1) + j+1
                self.n2 = self.n1+1
                self.n3 = self.n1 + (self.nel_x+1)
                self.n4 = self.n3 +1
                
                self.node_number[0,k] = self.n1
                self.node_number[1,k] = self.n2
                self.node_number[2,k] = self.n4
                self.node_number[3,k] = self.n3
                 
                self.ele_end_coord[0,:,k] = self.mesh_coord[self.n1-1,:]
                self.ele_end_coord[1,:,k] = self.mesh_coord[self.n2-1,:]
                self.ele_end_coord[2,:,k] = self.mesh_coord[self.n3-1,:]
                self.ele_end_coord[3,:,k] = self.mesh_coord[self.n4-1,:]
                
                if self.ele_node == 8 :
                    # if 8 noded quadrilateral element is called for
                    
                    self.n1 = i*(self.nel_x*3+2) + 2*j + 1
                    self.n2 = self.n1+2
                    self.n4 = self.n1 + (2*self.nel_x+1) + (self.nel_x+1)
                    self.n3 = self.n4 + 2
                    self.n5 = self.n1 + 1
                    self.n7 = self.n4 + 1
                    self.n8 = self.n1 + 2*(self.nel_x-j) + 1 + j
                    self.n6 = self.n8+1
                    self.node_number[0,k] = self.n1
                    self.node_number[1,k] = self.n2
                    self.node_number[2,k] = self.n3
                    self.node_number[3,k] = self.n4
                    self.node_number[4,k] = self.n5
                    self.node_number[5,k] = self.n6
                    self.node_number[6,k] = self.n7
                    self.node_number[7,k] = self.n8

                k = k+1
                
        return self.mesh_coord, self.ele_end_coord, self.node_number


class TriangularMesh :
    """
    Class used to automate triangular-mesh generation and node numbering of the domain 
    
    Only Quadrilateral domain is completely automated. For triangular domain, user need to enter the corresponding coordinates and node numbers
    
    """
    
    def __init__(self, nel_x, nel_y, ele_node) :
        """ 
        Initializes the class with mesh properties. 
        """

        self.nel_x = nel_x
        self.nel_y = nel_y
        self.nel = nel_x*nel_y
        self.ele_node = ele_node
        self.gl_nodes = (int(nel_x/2)+1)*(nel_y+1) # no:of corner nodes 
   
        self.mesh_coord = np.zeros([self.gl_nodes, 2]) # coordinates of global cornder nodes
        self.ele_end_coord = np.zeros([3, 2, self.nel]) # coordinates of outer nodes of each element
        self.node_number = np.zeros([self.ele_node,self.nel]) # node number of each node of each element

 
    def mesh_generate(self, v1, v2, v3, v4) :
        """Generates the triangular-mesh of the domain
        
        Returns 1) array with coordinates of global corner nodes
                2) array with coordinates of corner-nodes of each element
                3) array with the node number of each node of each element in the mesh
        
        Element coordinates and numbering : from left bottom in anticlockwise, 1-2-3 order obtained. 
        """        
        self.x1 = v1.xcoord
        self.y1 = v1.ycoord
        
        if v1.xcoord==v4.xcoord and v1.ycoord==v4.ycoord :
            
            # triangulae domain discretized into triangular mesh 
            
            self.mesh_coord = np.zeros([3, 2,1])
            self.mesh_coord[0,0,0] = v1.xcoord
            self.mesh_coord[0,1,0] = v1.ycoord
            self.mesh_coord[1,0,0] = v2.xcoord
            self.mesh_coord[1,1,0] = v2.ycoord
            self.mesh_coord[2,0,0] = v3.xcoord
            self.mesh_coord[2,1,0] = v3.ycoord
                
            self.ele_end_coord = self.mesh_coord
            self.node_number = np.array([[1],[2],[3]])
            if self.ele_node == 6 :
                self.node_number = np.array([[1],[2],[3],[4],[5],[6]])
                
        else :
            
            # quadrilateral domain automatically discretized into triangular elements 
            
            if self.nel_x%2 :
                # for the specific meshing algorithm, X direction should have even no:of elements
                print("No of elements in x should be even")
                quit()

            self.nel_x = int(self.nel_x/2) # To divide quadrilateral domain into triangular elements 
            self.x2 = v2.xcoord
            self.y2 = v2.ycoord
            i = 0

            self.x = v1.xcoord
            self.y = v1.ycoord

            while(self.x<=v3.xcoord and self.y<=v3.ycoord):
                # Loop travels around the domain and assigns the coordinates of global corner nodes to mesh_coord

                self.mesh_coord[i, 0] = self.x
                self.mesh_coord[i, 1] = self.y

                self.dx = (self.x2 - self.x1)/self.nel_x
                self.dy = (self.y2 - self.y1)/self.nel_x

                self.x = self.x + self.dx
                self.y = self.y + self.dy

                if self.x > self.x2 :

                    self.x1 = self.x1 + (v4.xcoord-v1.xcoord)/self.nel_y
                    self.y1 = self.y1 + (v4.ycoord-v1.ycoord)/self.nel_y

                    self.x2 = self.x2 + (v3.xcoord-v2.xcoord)/self.nel_y
                    self.y2 = self.y2 + (v3.ycoord-v2.ycoord)/self.nel_y

                    self.x = self.x1
                    self.y = self.y1

                i = i+1

            k=0
            m=0
            self.node_number = np.zeros([self.ele_node,self.nel])

            for i in range(self.nel_y) :
                # Loop travels around the domain and assigns the coordinates and node number of each node of each element

                for j in range(self.nel_x) :
                    
                    self.n1 = i*(self.nel_x+1) + j+1
                    self.n2 = self.n1+1
                    self.n3 = self.n1 + (self.nel_x+1)
                    self.n4 = self.n3 +1
                    
                    self.ele_end_coord[0,:,k] = self.mesh_coord[self.n1-1,:]
                    self.ele_end_coord[1,:,k] = self.mesh_coord[self.n2-1,:]
                    self.ele_end_coord[2,:,k] = self.mesh_coord[self.n3-1,:]

                    self.ele_end_coord[0,:,k+1] = self.mesh_coord[self.n2-1,:]
                    self.ele_end_coord[1,:,k+1] = self.mesh_coord[self.n4-1,:]
                    self.ele_end_coord[2,:,k+1] = self.mesh_coord[self.n3-1,:]
                    
                    self.node_number[0,k] = self.n1
                    self.node_number[1,k] = self.n2
                    self.node_number[2,k] = self.n3
                    self.node_number[0,k+1] = self.n2
                    self.node_number[1,k+1] = self.n4
                    self.node_number[2,k+1] = self.n3
                    
                    if self.ele_node == 6 :
                        # for 6 noded triangular elements
                        
                        m += 1
                        self.n1 = i*(self.nel_x*2+5) + 2*j + 1
                        self.n2 = self.n1+2
                        self.n4 = self.n1 + (2*self.nel_x+1) + (self.nel_x+1)
                        self.n3 = self.n4 + 2
                        self.n5 = self.n1 + 1
                        self.n7 = self.n4 + 1
                        self.n8 = self.n1 + 2*self.nel_x + 1 - j
                        self.n6 = self.n8+1
                        self.n9 = (2*self.nel_x+1)*(self.nel_y+1) + self.nel_y*(self.nel_x+1) + m
                        
                        self.node_number[0,k] = self.n1
                        self.node_number[1,k] = self.n2
                        self.node_number[2,k] = self.n4
                        self.node_number[3,k] = self.n5
                        self.node_number[4,k] = self.n9
                        self.node_number[5,k] = self.n8
                        self.node_number[0,k+1] = self.n2
                        self.node_number[1,k+1] = self.n3
                        self.node_number[2,k+1] = self.n4
                        self.node_number[3,k+1] = self.n6
                        self.node_number[4,k+1] = self.n7
                        self.node_number[5,k+1] = self.n9

                    k = k+2

        return self.mesh_coord, self.ele_end_coord,self.node_number


class CompleteMesh :
    """
    class used to generate the complete mesh coordinates including quadratic element nodes
    """
    
    def __init__(self, ele_type, ele_node, nel, mesh_coord, ele_end_coord) :
        """
        Initializes the class with mesh properties and corner node coordinates
        """
        
        self.ele_type = ele_type
        self.ele_node = ele_node
        self.nel = nel
        self.mesh_coord = mesh_coord
        self.ele_end_coord = ele_end_coord
        
        self.comp_mesh = np.zeros([self.ele_node, 2, self.nel])
        
    def get_mesh(self) :
        """
        Returns the complete mesh of the domain. 
        
        Generates the coordinates of intermediate nodes in quadratic elements
        """
        
        if self.ele_type == "Triangular" and self.ele_node == 3 :
            # 3 noded triangular element
            
            self.comp_mesh = self.ele_end_coord
        
        elif self.ele_type == "Triangular" and self.ele_node == 6 :
             # 6 noded triangular element
             
            for i in range(self.nel) :
                k =0
                for j in [0,1,2] :
                    self.comp_mesh[j,:,i] = self.ele_end_coord[k,:,i]
                    m=k+1
                    if k == 2: m = 0
                    self.comp_mesh[j+3,:,i] = (self.ele_end_coord[k,:,i] + self.ele_end_coord[m,:,i])/2
                    k = k+1
        
        elif self.ele_type in ["Quadrilateral", "Rectangular"] and self.ele_node == 4 :
             # 4 noded quadrilateral/rectangular element
            
            self.ele_end_coord[[2,3],:,:] = self.ele_end_coord[[3,2],:,:]
            self.comp_mesh = self.ele_end_coord
            
        elif self.ele_type == "Quadrilateral" and self.ele_node == 8 :
            # 8 noded quadrilateral element
            
            self.ele_end_coord[[2,3],:,:] = self.ele_end_coord[[3,2],:,:]
            
            for i in range(self.nel) :
                
                k =0
                for j in [0,1,2,3] :
                    self.comp_mesh[j,:,i] = self.ele_end_coord[k,:,i]
                    m=k+1
                    if k == 3: m = 0
                    self.comp_mesh[j+4,:,i] = (self.ele_end_coord[k,:,i] + self.ele_end_coord[m,:,i])/2
                    k = k+1
        else : 
            
            print("Unknown element type/degree. Program Terminating...")
            exit()
      
        return self.comp_mesh


class System : 
    """
    Class to input the system parameters and generate the Constitutive Matrix
    """
    
    def __init__(self, elast, mu, plane) :
        """
        Initializes the class with system parameters
        """
        self.elast = elast
        self.mu = mu
        self.plane = plane
        
    def get_system(self) :
        """
        Generates and returns the constitutive matrix, D
        """
        
        if self.plane == "plane-stress" :
            self.D = (self.elast/(1-(self.mu**2)))* np.array([[1, self.mu, 0], 
                                                              [self.mu, 1, 0], 
                                                              [0, 0, (1-self.mu)/2]])
                          
        elif self.plane == "plane-strain" :
            self.D = (self.elast/(1+self.mu)/(1-2*self.mu))*np.array([[1-self.mu, self.mu, 0], 
                                                                      [self.mu, 1-self.mu, 0], 
                                                                      [0, 0, 0.5-self.mu]])
        
        return self.D


class BoundaryCondition :
    """
    Class used to intake the domain boundary conditions and returns the boundary condition for each element of the mesh
    """
    
    def __init__(self, v1, v2, v3, v4, ele_type, ele_node, comp_mesh, nel) :
        """
        Initializes the class with domain and mesh properties
        """
        
        self.nel = nel
        self.ele_type = ele_type
        self.ele_node = ele_node

        self.comp_mesh = comp_mesh
 
        self.v1, self.v2, self.v3, self.v4 = v1, v2, v3, v4 # Corner Coordinates
        
        self.ele_bc = np.ones([self.ele_node*2,self.nel])
    
    def get_disp_bc(self, res1, res2, res3, res4, eres1, eres2, eres3, eres4) :
        """
        Intakes the boundary  condition for the domain.
        
        Returns a matrix of size (ele_dof*no_ele) with components 0 and 1. 0 denotes restriction and 1 denotes
        freedom of movement
        
        res1, res2, res3, res4 - BC for end coordinates of domain
        eres1, eres2, eres3, eres4 - BC for edges of the domain
        """
        
        for i in range(self.nel) :
            # Loop travels through each element, check for possible boundary conditions and generate bc matrix for the element.
            
            self.ele_x = self.comp_mesh[:,0,i] # x coordinate of nodes of the element
            self.ele_y = self.comp_mesh[:,1,i] # y coordinate of nodes of the element

            if not all(eres1) :
                # If there is a restrain in edge 1 
               
                self.edge1_m = (self.v2.ycoord-self.v1.ycoord)/(self.v2.xcoord-self.v1.xcoord)
                
                if eres1[0]==0 :
                    k = 0
                    for j in range(len(self.ele_x)) :
                        
                        self.y = (self.edge1_m*(self.ele_x[j]-self.v1.xcoord))+self.v1.ycoord
                        
                        if self.y == self.ele_y[j] :
                            self.ele_bc[k,i] = 0
                        k += 2
            
                if eres1[1]==0 :
                    
                    k = 1
                    for j in range(len(self.ele_x)) :
                        
                        self.y = (self.edge1_m*(self.ele_x[j]-self.v1.xcoord))+self.v1.ycoord
                        if self.y == self.ele_y[j] :
                            self.ele_bc[k,i] = 0
                        k += 2
       
            if not all(eres3) :
                # If there is a restrain in edge 3
                
                self.edge3_m = (self.v4.ycoord-self.v3.ycoord)/(self.v4.xcoord-self.v3.xcoord)
               
                if eres3[0]==0 :
                    k = int(self.ele_node/2 +2)
                    k = 0
                    for j in range(len(self.ele_x)) :
                        
                        self.y = (self.edge3_m*(self.ele_x[j]-self.v3.xcoord))+self.v3.ycoord
                        if self.y == self.ele_y[j] :
                            self.ele_bc[k,i] = 0
                        k += 2
                
                if eres3[1]==0 :
                    k = int(self.ele_node/2)
                    k=1
                    for j in range(len(self.ele_x)) :
                        
                        self.y = (self.edge3_m*(self.ele_x[j]-self.v3.xcoord))+self.v3.ycoord
                        if self.y == self.ele_y[j] :
                            self.ele_bc[k,i] = 0
                        k += 2

            if not all(eres2) :
                # If there is a restrain in edge 2
                
                if self.v3.xcoord == self.v2.xcoord :
                    if eres2[0] == 0 :
                        k = 0
                        for j in range(len(self.ele_x)) :
                            if self.ele_x[j] == self.v2.xcoord :
                                self.ele_bc[k,i] = 0
                            k += 2
                    if eres2[1] == 0 :
                        k = 1
                        for j in range(len(self.ele_x)) :
                            if self.ele_x[j] == self.v2.xcoord :
                                self.ele_bc[k,i] = 0
                            k += 2
                else :
                    self.edge2_m = (self.v3.ycoord-self.v2.ycoord)/(self.v3.xcoord-self.v2.xcoord)
                    
                    if eres2[0]==0 :
                        k = 0
                        for j in range(len(self.ele_x)) :

                            self.y = (self.edge2_m*(self.ele_x[j]-self.v2.xcoord))+self.v2.ycoord
                            if self.y == self.ele_y[j] :
                                self.ele_bc[k,i] = 0
                            k += 2

                    if eres2[1]==0 :
                        k=1
                        for j in range(len(self.ele_x)) :

                            self.y = (self.edge2_m*(self.ele_x[j]-self.v2.xcoord))+self.v2.ycoord
                            if self.y == self.ele_y[j] :
                                self.ele_bc[k,i] = 0
                            k += 2
                            
            if not all(eres4) :
                # If there is a restrain in edge 4
                                
                if self.v4.xcoord == self.v1.xcoord :
                    if eres4[0] == 0 :
                        k = 0
                        for j in range(len(self.ele_x)) :
                            if self.ele_x[j] == self.v4.xcoord :
                                self.ele_bc[k,i] = 0
                            k += 2
                    if eres4[1] == 0 :
                        k = 1
                        for j in range(len(self.ele_x)) :
                            if self.ele_x[j] == self.v4.xcoord :
                                self.ele_bc[k,i] = 0
                            k += 2
                else :
                    self.edge4_m = (self.v4.ycoord-self.v1.ycoord)/(self.v4.xcoord-self.v1.xcoord)
                    
                    if eres4[0]==0 :
                        k = 0
                        for j in range(len(self.ele_x)) :

                            self.y = (self.edge4_m*(self.ele_x[j]-self.v4.xcoord))+self.v4.ycoord
                            if self.y == self.ele_y[j] :
                                self.ele_bc[k,i] = 0
                            k += 2

                    if eres4[1]==0 :
                        k=1
                        for j in range(len(self.ele_x)) :

                            self.y = (self.edge4_m*(self.ele_x[j]-self.v4.xcoord))+self.v4.ycoord
                            if self.y == self.ele_y[j] :
                                self.ele_bc[k,i] = 0
                            k += 2
           
            if not all(res1) :
                # If there is a restrain in coordinate 1
                
                if res1[0]==0 :
                    k = 0
                    for j in range(len(self.ele_x)) :
                        
                        if self.v1.xcoord == self.ele_x[j] and self.v1.ycoord == self.ele_y[j] :
                            self.ele_bc[k,i] = 0
                        k += 2
                if res1[1]==0 :
                    k = 1
                    for j in range(len(self.ele_x)) :
                        
                        if self.v1.xcoord == self.ele_x[j] and self.v1.ycoord == self.ele_y[j] :
                            self.ele_bc[k,i] = 0
                        k += 2
                        
            if not all(res2) :
                # If there is a restrain in coordinate 2
                
                if res2[0]==0 :
                    k = 0
                    for j in range(len(self.ele_x)) :
                        
                        if self.v2.xcoord == self.ele_x[j] and self.v2.ycoord == self.ele_y[j] :
                            self.ele_bc[k,i] = 0
                        k += 2
                if res2[1]==0 :
                    k = 1
                    for j in range(len(self.ele_x)) :
                        
                        if self.v2.xcoord == self.ele_x[j] and self.v2.ycoord == self.ele_y[j] :
                            self.ele_bc[k,i] = 0
                        k += 2
                        
            if not all(res3) :
                # If there is a restrain in coordinate 3
                
                if res3[0]==0 :
                    k = 0
                    for j in range(len(self.ele_x)) :
                        
                        if self.v3.xcoord == self.ele_x[j] and self.v3.ycoord == self.ele_y[j] :
                            self.ele_bc[k,i] = 0
                        k += 2
                if res3[1]==0 :
                    k = 1
                    for j in range(len(self.ele_x)) :
                        
                        if self.v3.xcoord == self.ele_x[j] and self.v3.ycoord == self.ele_y[j] :
                            self.ele_bc[k,i] = 0
                        k += 2
                        
            if not all(res4) :
                # If there is a restrain in coordinate 4
                
                if res4[0]==0 :
                    k = 0
                    for j in range(len(self.ele_x)) :
                        
                        if self.v4.xcoord == self.ele_x[j] and self.v4.ycoord == self.ele_y[j] :
                            self.ele_bc[k,i] = 0
                        k += 2
                if res4[1]==0 :
                    k = 1
                    for j in range(len(self.ele_x)) :
                        
                        if self.v4.xcoord == self.ele_x[j] and self.v4.ycoord == self.ele_y[j] :
                            self.ele_bc[k,i] = 0
                        k += 2
            
        return self.ele_bc


class Loading :
    """
    Class used to intake the domain loading conditions and returns the load value for each element of the mesh
    """
    
    def __init__(self, v1, v2, v3, v4, ele_type, ele_node, comp_mesh, nel_x, nel_y) :
        """
        Initializes the class with domain and mesh properties
        """
        
        self.ele_type = ele_type
        self.ele_node = ele_node
        self.nel = nel_x*nel_y
        self.nel_x = nel_x
        self.nel_y = nel_y

        self.comp_mesh = comp_mesh
 
        self.v1, self.v2, self.v3, self.v4 = v1, v2, v3, v4 # Corner Coordinates
        self.domain = Edge(v1,v2,v3,v4)
        
        self.ele_load = np.zeros([self.ele_node*2,self.nel])
        
    def get_element_load(self, eload1, eload2, eload3, eload4) :
        """
        Intakes the loading condition for the domain.
        
        Returns a matrix of size (ele_dof*no_ele) with values of load for each dof of each element
        
        eload1, eload2, eload3, eload4 - values of loading for edges of the domain
        """
        
        if self.ele_node == 4 :
            self.node_x = self.nel_x + 1
            self.node_y = self.nel_y + 1
        elif self.ele_node == 3 :
            self.node_x = self.nel_x/2 + 1
            self.node_y = self.nel_y + 1 
        elif self.ele_node == 8 :
            self.node_x = self.nel_x*2 + 1
            self.node_y = self.nel_y*2 + 1
        elif self.ele_node == 6 :
            self.node_x = self.nel_x + 1
            self.node_y = self.nel_y*2 + 1
                    
        for i in range(self.nel) :
            
            self.ele_x = self.comp_mesh[:,0,i] # x coordinate of nodes of the element
            self.ele_y = self.comp_mesh[:,1,i] # y coordinate of nodes of the element

            if any(eload1) :
                # If there is a load in edge 1

                self.edge1_m = (self.v2.ycoord-self.v1.ycoord)/(self.v2.xcoord-self.v1.xcoord)

                if eload1[0] :
                    k = 0
                    self.tx = eload1[0]*self.domain.edge1/self.node_x
                    for j in range(len(self.ele_x)) :

                        self.y = (self.edge1_m*(self.ele_x[j]-self.v1.xcoord))+self.v1.ycoord

                        if self.y == self.ele_y[j] :
                            self.ele_load[k,i] = self.t
                        k += 2

                if eload1[1] :

                    k = 1
                    self.ty = eload1[1]*self.domain.edge1/self.node_x
                    for j in range(len(self.ele_x)) :

                        self.y = (self.edge1_m*(self.ele_x[j]-self.v1.xcoord))+self.v1.ycoord
                        if self.y == self.ele_y[j] :
                            self.ele_load[k,i] = self.ty
                        k += 2

            if any(eload3) :
                # If there is a load in edge 3

                self.edge3_m = (self.v4.ycoord-self.v3.ycoord)/(self.v4.xcoord-self.v3.xcoord)

                if eload3[0] :
                    k = 0
                    self.tx = eload3[0]*self.domain.edge3/self.node_x
                    for j in range(len(self.ele_x)) :

                        self.y = (self.edge3_m*(self.ele_x[j]-self.v3.xcoord))+self.v3.ycoord

                        if self.y == self.ele_y[j] :
                            self.ele_load[k,i] = self.tx
                        k += 2

                if eload3[1] :

                    k = 1
                    self.ty = eload3[1]*self.domain.edge3/self.node_x
                    for j in range(len(self.ele_x)) :

                        self.y = (self.edge3_m*(self.ele_x[j]-self.v3.xcoord))+self.v3.ycoord
                        if self.y == self.ele_y[j] :
                            self.ele_load[k,i] = self.ty
                        k += 2

            if any(eload2) :
                # If there is a load in edge 2

                if self.v3.xcoord == self.v2.xcoord :
                    if eload2[0] :
                        k = 0
                        self.tx = eload2[0]*self.domain.edge2/self.node_y
                        for j in range(len(self.ele_x)) :
                            if self.ele_x[j] == self.v2.xcoord :
                                self.ele_load[k,i] = self.tx
                            k += 2
                    if eload2[1]:
                        k = 1
                        self.ty = eload2[1]*self.domain.edge2/self.node_y
                        for j in range(len(self.ele_x)) :
                            if self.ele_x[j] == self.v2.xcoord :
                                self.ele_load[k,i] = self.ty
                            k += 2
                else :
                    self.edge2_m = (self.v3.ycoord-self.v2.ycoord)/(self.v3.xcoord-self.v2.xcoord)

                    if eload2[0]:
                        k = 0
                        self.tx = eload2[0]*self.domain.edge2/self.node_y
                        for j in range(len(self.ele_x)) :

                            self.y = (self.edge2_m*(self.ele_x[j]-self.v2.xcoord))+self.v2.ycoord
                            if self.y == self.ele_y[j] :
                                self.ele_load[k,i] = self.tx
                            k += 2

                    if eload2[1] :
                        k=1
                        self.ty = eload2[0]*self.domain.edge2/self.node_y
                        for j in range(len(self.ele_x)) :

                            self.y = (self.edge2_m*(self.ele_x[j]-self.v2.xcoord))+self.v2.ycoord
                            if self.y == self.ele_y[j] :
                                self.ele_load[k,i] = self.ty
                            k += 2

            if any(eload4) :
                # If there is a load in edge 4

                if self.v4.xcoord == self.v1.xcoord :
                    if eload4[0] :
                        k = 0
                        self.tx = eload4[0]*self.domain.edge4/self.node_y
                        for j in range(len(self.ele_x)) :
                            if self.ele_x[j] == self.v4.xcoord :
                                self.ele_load[k,i] = self.tx
                            k += 2
                    if eload4[1]:
                        k = 1
                        self.ty = eload4[1]*self.domain.edge4/self.node_y
                        for j in range(len(self.ele_x)) :
                            if self.ele_x[j] == self.v4.xcoord :
                                self.ele_load[k,i] = self.ty
                            k += 2
                else :
                    self.edge4_m = (self.v4.ycoord-self.v1.ycoord)/(self.v4.xcoord-self.v1.xcoord)

                    if eload4[0]:
                        k = 0
                        self.tx = eload4[0]*self.domain.edge4/self.node_y
                        for j in range(len(self.ele_x)) :

                            self.y = (self.edge4_m*(self.ele_x[j]-self.v4.xcoord))+self.v4.ycoord
                            if self.y == self.ele_y[j] :
                                self.ele_load[k,i] = self.tx
                            k += 2

                    if eload4[1] :
                        k=1
                        self.ty = eload4[0]*self.domain.edge4/self.node_y
                        for j in range(len(self.ele_x)) :

                            self.y = (self.edge4_m*(self.ele_x[j]-self.v4.xcoord))+self.v4.ycoord
                            if self.y == self.ele_y[j] :
                                self.ele_load[k,i] = self.ty
                            k += 2
                            
        return self.ele_load


class ShapeFunction :
    """
    class used to define shape functions (N) and deformation functions (B) of each element in the mesh
    
    Cartesian Coordinates used for 3 noded triangular elements
    Natural Coordinates used for 6 noded triangular, 4 and 8 noded quadrilatel/rectangular elements
    """
    def __init__(self,ele_type, ele_node) :
        """
        Initializes the class with element properties
        """
        
        self.ele_type = ele_type
        self.ele_node = ele_node  
    
    def get_shape_function(self, v1, v2, v3, v4, x, y) :
        """
        Returns the Jacobian, shape function, deformation function and shape function components of an element\
            
        v1, v2, v3, v4 - objects of Vertex class for end coordinates of an element
        
        Returns 
            1) ele_N - shape function of the 2D element in matrix form
            2) ele_Ne - components of the shape function as an array
            3) ele_B - deformation function of the 2D element in matrix form
            4) det_J - determinant of the jacobian, if natural coordinates are used
        
        """
        
        self.det_J = 0 # Jacobian 
        
        if self.ele_type == "Triangular" and self.ele_node == 3 :
            # For 3 noded trianguar elements. 
            # Cartesian system is used 
            
            self.x_pos = x
            self.y_pos = y
            
            self.a1 = v2.xcoord*v3.ycoord - v3.xcoord*v2.ycoord
            self.a2 = v3.xcoord*v1.ycoord - v1.xcoord*v3.ycoord
            self.a3 = v1.xcoord*v2.ycoord - v2.xcoord*v1.ycoord
            
            self.b1 = v2.ycoord - v3.ycoord
            self.b2 = v3.ycoord - v1.ycoord
            self.b3 = v1.ycoord - v2.ycoord
            
            self.g1 = v3.xcoord - v2.xcoord
            self.g2 = v1.xcoord - v3.xcoord
            self.g3 = v2.xcoord - v1.xcoord
            
            self.ele_area = 0.5*(v1.xcoord*(v2.ycoord-v3.ycoord) + 
                                 v2.xcoord*(v3.ycoord-v1.ycoord) + 
                                 v3.xcoord*(v1.ycoord-v2.ycoord))
            
            self.N1 = 1/(2*self.ele_area)*(self.a1 + self.b1*self.x_pos + self.g1*self.y_pos)
            self.N2 = 1/(2*self.ele_area)*(self.a2 + self.b2*self.x_pos + self.g2*self.y_pos)
            self.N3 = 1/(2*self.ele_area)*(self.a3 + self.b3*self.x_pos + self.g3*self.y_pos)
            
            self.N = np.array([[self.N1, 0, self.N2, 0, self.N3, 0], 
                               [0, self.N1, 0, self.N2, 0, self.N3]])
            
            self.Ne = np.array([[self.N1], [self.N2], [self.N3]])
            
            self.B = 1/(2*self.ele_area)*np.array([[self.b1, 0, self.b2, 0, self.b3, 0],
                                                   [0, self.g1, 0, self.g2, 0, self.g3],
                                                   [self.g1, self.b1, self.g2, self.b2, self.g3, self.b3]])
            
        
        elif self.ele_type == "Triangular" and self.ele_node == 6 :
            # 6 noded triangular element. 
            # Natural coordinate system is used
            # intermediate nodes found out using end nodes of the element
            
            v4 = Vertex((v1.xcoord+v2.xcoord)/2, (v1.ycoord+v2.ycoord)/2)
            v5 = Vertex((v2.xcoord+v3.xcoord)/2, (v2.ycoord+v3.ycoord)/2)
            v6 = Vertex((v3.xcoord+v1.xcoord)/2, (v3.ycoord+v1.ycoord)/2)
            
            self.L1 = x
            self.L2 = y
            self.L3 = 1 - self.L1 - self.L2 
   
            # components of shape function
            
            self.N1 = self.L1*(2*self.L1 - 1)
            self.N2 = self.L2*(2*self.L2 - 1)
            self.N3 = self.L3*(2*self.L3 - 1)
            self.N4 = 4*self.L1*self.L2
            self.N5 = 4*self.L2*self.L3
            self.N6 = 4*self.L3*self.L1
            
            # shape function matrix 
            
            self.N = np.array([[self.N1, 0, self.N2, 0, self.N3, 0, self.N4, 0, self.N5, 0, self.N6, 0], 
                               [0, self.N1, 0, self.N2, 0, self.N3, 0, self.N4, 0, self.N5, 0, self.N6]])
            
            self.Ne = np.array([[self.N1], [self.N2], [self.N3], [self.N4], [self.N5], [self.N6]])
       
            # derivative of shape function components
            
            self.dN1dL1 = 4*self.L1 - 1
            self.dN1dL2 = 0
            self.dN2dL1 = 0
            self.dN2dL2 = 4*self.L2 -1 
            self.dN3dL1 = -3 + 4*self.L1 + 4*self.L2
            self.dN3dL2 = -3 + 4*self.L1 + 4*self.L2
            self.dN4dL1 = 4*self.L2
            self.dN4dL2 = 4*self.L1
            self.dN5dL1 = -4*self.L2
            self.dN5dL2 = 4 - 4*self.L1 - 8*self.L2
            self.dN6dL1 = 4 - 4*self.L2 - 8*self.L1
            self.dN6dL2 = -4*self.L1
            
            # calculating jacobian and properties 
            
            self.J11 = self.dN1dL1*v1.xcoord + self.dN2dL1*v2.xcoord + self.dN3dL1*v3.xcoord + self.dN4dL1*v4.xcoord + self.dN5dL1*v5.xcoord + self.dN6dL1*v6.xcoord 
            self.J12 = self.dN1dL1*v1.ycoord + self.dN2dL1*v2.ycoord + self.dN3dL1*v3.ycoord + self.dN4dL1*v4.ycoord + self.dN5dL1*v5.ycoord + self.dN6dL1*v6.ycoord
            self.J21 = self.dN1dL2*v1.xcoord + self.dN2dL2*v2.xcoord + self.dN3dL2*v3.xcoord + self.dN4dL2*v4.xcoord + self.dN5dL2*v5.xcoord + self.dN6dL2*v6.xcoord
            self.J22 = self.dN1dL2*v1.ycoord + self.dN2dL2*v2.ycoord + self.dN3dL2*v3.ycoord + self.dN4dL2*v4.ycoord + self.dN5dL2*v5.ycoord + self.dN6dL2*v6.ycoord

            self.J = np.array([[self.J11, self.J12], [self.J21, self.J22]]) # Jacobian matrix 

            self.det_J = np.linalg.det(self.J)
            self.inv_J = np.linalg.inv(self.J)
                    
            self.Ji11, self.Ji12, self.Ji21, self.Ji22 = self.inv_J[0,0], self.inv_J[0,1], self.inv_J[1,0], self.inv_J[1,1]
            
            # determining deformation function 
            
            self.B1 = np.array([[self.Ji11, self.Ji12, 0, 0],
                                        [0, 0, self.Ji21, self.Ji22],
                                        [self.Ji21, self.Ji22, self.Ji11, self.Ji12]])
            
            self.B2 = np.array([[self.dN1dL1, 0, self.dN2dL1, 0, self.dN3dL1, 0, self.dN4dL1, 0, self.dN5dL1, 0, self.dN6dL1, 0],
                                  [self.dN1dL2, 0, self.dN2dL2, 0, self.dN3dL2, 0, self.dN4dL2, 0, self.dN5dL2, 0, self.dN6dL2, 0],
                                  [0, self.dN1dL1, 0, self.dN2dL1, 0, self.dN3dL1, 0, self.dN4dL1, 0, self.dN5dL1, 0, self.dN6dL1],
                                  [0, self.dN1dL2, 0, self.dN2dL2, 0, self.dN3dL2, 0, self.dN4dL2, 0, self.dN5dL2, 0, self.dN6dL2]])

            self.B = self.B1@self.B2 # deformation function

        elif self.ele_type in ["Rectangular", "Quadrilateral"] and self.ele_node == 4 :
            # 4 noded quadrilateral/rectangular element. 
            # natural coordinate system used
            
            self.si = x
            self.eta = y
            
            # components of shape function
            
            self.N1 = 1/4*(1-self.si)*(1-self.eta)
            self.N2 = 1/4*(1+self.si)*(1-self.eta)
            self.N3 = 1/4*(1+self.si)*(1+self.eta)
            self.N4 = 1/4*(1-self.si)*(1+self.eta)
            
            # shape function matrix
            
            self.N = np.array([[self.N1, 0, self.N2, 0, self.N3, 0, self.N4, 0], 
                               [0, self.N1, 0, self.N2, 0, self.N3, 0, self.N4]])
            
            self.Ne = np.array([[self.N1], [self.N2], [self.N3], [self.N4]])
            
            # derivatives of shape function components
            
            self.dN1dsi = -1*(1-self.eta)/4
            self.dN1deta = -1*(1-self.si)/4
            self.dN2dsi = 1*(1-self.eta)/4
            self.dN2deta = -1*(1+self.si)/4
            self.dN3dsi = 1*(1+self.eta)/4
            self.dN3deta = 1*(1+self.si)/4
            self.dN4dsi = -1*(1+self.eta)/4
            self.dN4deta = 1*(1-self.si)/4
            
            # jacobian matrix and it's properties
            
            self.J11 = 0.25*(1-self.eta)*(v2.xcoord-v1.xcoord) + 0.25*(1+self.eta)*(v3.xcoord-v4.xcoord)
            self.J12 = 0.25*(1-self.eta)*(v2.ycoord-v1.ycoord) + 0.25*(1+self.eta)*(v3.ycoord-v4.ycoord)
            self.J21 = 0.25*(1-self.si)*(v4.xcoord-v1.xcoord) + 0.25*(1+self.si)*(v3.xcoord-v2.xcoord)
            self.J22 = 0.25*(1-self.si)*(v4.ycoord-v1.ycoord) + 0.25*(1+self.si)*(v3.ycoord-v2.ycoord)

            self.J = np.array([[self.J11, self.J12], [self.J21, self.J22]])

            self.det_J = np.linalg.det(self.J)
            self.inv_J = np.linalg.inv(self.J)
                    
            self.Ji11, self.Ji12, self.Ji21, self.Ji22 = self.inv_J[0,0], self.inv_J[0,1], self.inv_J[1,0], self.inv_J[1,1]
            
            # determining deformation function
            
            self.B1 = np.array([[self.Ji11, self.Ji12, 0, 0],
                                        [0, 0, self.Ji21, self.Ji22],
                                        [self.Ji21, self.Ji22, self.Ji11, self.Ji12]])
                    
            self.B2 = np.array([[self.dN1dsi, 0, self.dN2dsi, 0, self.dN3dsi, 0, self.dN4dsi, 0],
                                  [self.dN1deta, 0, self.dN2deta, 0, self.dN3deta, 0, self.dN4deta, 0],
                                  [0, self.dN1dsi, 0, self.dN2dsi, 0, self.dN3dsi, 0, self.dN4dsi],
                                  [0, self.dN1deta, 0, self.dN2deta, 0, self.dN3deta, 0, self.dN4deta]])

            self.B = self.B1@self.B2 # deformation function
            
        elif self.ele_type == "Quadrilateral" and self.ele_node == 8 :
            # 8  noded quadrilateral element
            # natural coordinates used
            # intermediate nodes found out using end coordinates of the element
            
            self.si = x
            self.eta = y
            
            v5 = Vertex((v1.xcoord+v2.xcoord)/2, (v1.ycoord+v2.ycoord)/2)
            v6 = Vertex((v2.xcoord+v3.xcoord)/2, (v2.ycoord+v3.ycoord)/2)
            v7 = Vertex((v3.xcoord+v4.xcoord)/2, (v3.ycoord+v4.ycoord)/2)
            v8 = Vertex((v4.xcoord+v1.xcoord)/2, (v4.ycoord+v1.ycoord)/2)
            
            # components of shape function 
            
            self.N1 = 1/4*(1-self.si)*(1-self.eta)*(-1-self.si-self.eta)
            self.N2 = 1/4*(1+self.si)*(1-self.eta)*(-1+self.si-self.eta)
            self.N3 = 1/4*(1+self.si)*(1+self.eta)*(-1+self.si+self.eta)
            self.N4 = 1/4*(1-self.si)*(1+self.eta)*(-1-self.si+self.eta)
            self.N5 = 1/2*(1-self.si**2)*(1-self.eta)
            self.N6 = 1/2*(1+self.si)*(1-self.eta**2)
            self.N7 = 1/2*(1-self.si**2)*(1+self.eta)
            self.N8 = 1/2*(1-self.si)*(1-self.eta**2)
            
            # shape function matrix
    
            self.N = np.array([[self.N1, 0, self.N2, 0, self.N3, 0, self.N4, 0, self.N5, 0, self.N6, 0, self.N7, 0, self.N8, 0], 
                               [0, self.N1, 0, self.N2, 0, self.N3, 0, self.N4, 0, self.N5, 0, self.N6, 0, self.N7, 0, self.N8]])
            
            self.Ne = np.array([[self.N1], [self.N2], [self.N3], [self.N4], [self.N5], [self.N6], [self.N7], [self.N8]])
            
            # derivatives of shape function components
            
            self.dN1dsi = -0.25*(1-self.eta)*(-1-self.si-self.eta) - 0.25*(1-self.si)*(1-self.eta)
            self.dN1deta = -0.25*(1-self.si)*(-1-self.si-self.eta) - 0.25*(1-self.si)*(1-self.eta)
            self.dN2dsi = 0.25*(1-self.eta)*(-1+self.si-self.eta) + 0.25*(1+self.si)*(1-self.eta)
            self.dN2deta = -0.25*(1+self.si)*(-1+self.si-self.eta) - 0.25*(1+self.si)*(1-self.eta)
            self.dN3dsi = 0.25*(1+self.eta)*(-1+self.si+self.eta) + 0.25*(1+self.si)*(1+self.eta)
            self.dN3deta = 0.25*(1+self.si)*(-1+self.si+self.eta) + 0.25*(1+self.si)*(1+self.eta)
            self.dN4dsi = -0.25*(1+self.eta)*(-1-self.si+self.eta) - 0.25*(1-self.si)*(1+self.eta)
            self.dN4deta = 0.25*(1-self.si)*(-1-self.si+self.eta) + 0.25*(1-self.si)*(1+self.eta)                                      
                    
            self.dN5dsi = -self.si*(1-self.eta)
            self.dN5deta = -0.5*(1-self.si**2)
            self.dN6dsi = 0.5*(1-self.eta**2)
            self.dN6deta = -self.eta*(1+self.si)
            self.dN7dsi = -self.si*(1+self.eta)
            self.dN7deta = 0.5*(1-self.si**2)
            self.dN8dsi = -0.5*(1-self.eta**2)
            self.dN8deta = -self.eta*(1-self.si)
            
            # jacobian matrix and it's properties
                                                          
            self.J11 = self.dN1dsi*v1.xcoord + self.dN2dsi*v3.xcoord + self.dN3dsi*v3.xcoord + self.dN4dsi*v4.xcoord + self.dN5dsi*v5.xcoord + self.dN6dsi*v6.xcoord + self.dN7dsi*v7.xcoord + self.dN8dsi*v8.xcoord
            self.J12 = self.dN1dsi*v1.ycoord + self.dN2dsi*v3.ycoord + self.dN3dsi*v3.ycoord + self.dN4dsi*v4.ycoord + self.dN5dsi*v5.ycoord + self.dN6dsi*v6.ycoord + self.dN7dsi*v7.ycoord + self.dN8dsi*v8.ycoord
            self.J21 = self.dN1deta*v1.xcoord + self.dN2deta*v2.xcoord + self.dN3deta*v3.xcoord + self.dN4deta*v4.xcoord + self.dN5deta*v5.xcoord + self.dN6deta*v6.xcoord + self.dN7deta*v7.xcoord + self.dN8deta*v8.xcoord
            self.J22 = self.dN1deta*v1.ycoord + self.dN2deta*v2.ycoord + self.dN3deta*v3.ycoord + self.dN4deta*v4.ycoord + self.dN5deta*v5.ycoord + self.dN6deta*v6.ycoord + self.dN7deta*v7.ycoord + self.dN8deta*v8.ycoord
                    
            self.J = np.array([[self.J11, self.J12], [self.J21, self.J22]])

            self.det_J = np.linalg.det(self.J)
            self.inv_J = np.linalg.inv(self.J)

            self.Ji11, self.Ji12, self.Ji21, self.Ji22 = self.inv_J[0,0], self.inv_J[0,1], self.inv_J[1,0], self.inv_J[1,1]
                        
            # determining deformation function 
            
            self.B1 = np.array([[self.Ji11, self.Ji12, 0, 0],
                                        [0, 0, self.Ji21, self.Ji22],
                                        [self.Ji21, self.Ji22, self.Ji11, self.Ji12]])
                    
            self.B2 = np.array([[self.dN1dsi, 0, self.dN2dsi, 0, self.dN3dsi, 0, self.dN4dsi, 0, self.dN5dsi, 0, self.dN6dsi, 0, self.dN7dsi, 0, self.dN8dsi, 0],
                                  [self.dN1deta, 0, self.dN2deta, 0, self.dN3deta, 0, self.dN4deta, 0, self.dN5deta, 0, self.dN6deta, 0, self.dN7deta, 0, self.dN8deta, 0],
                                  [0, self.dN1dsi, 0, self.dN2dsi, 0, self.dN3dsi, 0, self.dN4dsi, 0, self.dN5dsi, 0, self.dN6dsi, 0, self.dN7dsi, 0, self.dN8dsi],
                                  [0, self.dN1deta, 0, self.dN2deta, 0, self.dN3deta, 0, self.dN4deta, 0, self.dN5deta, 0, self.dN6deta, 0, self.dN7deta, 0, self.dN8deta]])

            self.B = self.B1@self.B2 # deformation matrix 
   
        return self.N, self.Ne, self.B, self.det_J
    
    def get_si_eta(self, x, y, v1, v2, v3, v4) :
        """
        Returns the parent/natural coordinates of given cartesian coordinates
            
            x, y - coordinates of a point in the domain 
            v1, v2, v3, v4 - Vertex Objects for end points of the element containing (x,y)
            
            si_val, eta_val - natural coordinates corresponding to x, y
        
        """
        self.si = Symbol('si')
        self.eta = Symbol('eta')
        
        if self.ele_type == "Triangular" and self.ele_node == 3 :
            # 3 noded triangular element
            # no change because cartesian coordinates are used
            
            self.si_val = x
            self.eta_val = y
            
            return self.si_val, self.eta_val
                
        elif self.ele_type == "Triangular" and self.ele_node == 6 :
            # 6 noded triangular elements
            # generates and solves equation in 'L1' and 'L2' and returns the value
            
            self.L1 = Symbol('L1')
            self.L2 = Symbol('L2')
            
            v4 = Vertex((v1.xcoord+v2.xcoord)/2, (v1.ycoord+v2.ycoord)/2)
            v5 = Vertex((v2.xcoord+v3.xcoord)/2, (v2.ycoord+v3.ycoord)/2)
            v6 = Vertex((v3.xcoord+v1.xcoord)/2, (v3.ycoord+v1.ycoord)/2)
    
            
            self.N1 = self.L1*(2*self.L1 - 1)
            self.N2 = self.L2*(2*self.L2 - 1)
            self.N3 = (1-self.L1-self.L2)*(2*(1-self.L1-self.L2) - 1)
            self.N4 = 4*self.L1*self.L2
            self.N5 = 4*self.L2*(1-self.L1-self.L2)
            self.N6 = 4*(1-self.L1-self.L2)*self.L1
            
            self.eq1 = self.N1*v1.xcoord + self.N2*v2.xcoord + self.N3*v3.xcoord + self.N4*v4.xcoord + self.N5*v5.xcoord + self.N6*v6.xcoord - x
            self.eq2 = self.N1*v1.ycoord + self.N2*v2.ycoord + self.N3*v3.ycoord + self.N4*v4.ycoord + self.N5*v5.ycoord + self.N6*v6.ycoord - y
  
            self.eqs = (self.eq2, self.eq1)
        
            self.ans = solve(self.eqs, self.L1, self.L2)
        
        elif self.ele_type in ["Rectangular", "Quadrilateral"] and self.ele_node == 4 :
            # 4 noded rectangular/quadrilateral elements
            # generates and solves equation in 'si' and 'eta' and returns the value
        
            self.eq1 = (1-self.si)*(1-self.eta)/4*v1.xcoord + (1+self.si)*(1-self.eta)/4*v2.xcoord + (1+self.si)*(1+self.eta)/4*v3.xcoord + (1-self.si)*(1+self.eta)/4*v4.xcoord - x
            self.eq2 = (1-self.si)*(1-self.eta)/4*v1.ycoord + (1+self.si)*(1-self.eta)/4*v2.ycoord + (1+self.si)*(1+self.eta)/4*v3.ycoord + (1-self.si)*(1+self.eta)/4*v4.ycoord - y
        
            self.eqs = (self.eq2, self.eq1)
        
            self.ans = solve(self.eqs, self.si, self.eta)
        
        elif self.ele_type == "Quadrilateral" and self.ele_node == 8 :
            # 8 noded quadrilateral elements
            # generates and solves equation in 'si' and 'eta' and returns the value
            
            v5 = Vertex((v1.xcoord+v2.xcoord)/2, (v1.ycoord+v2.ycoord)/2)
            v6 = Vertex((v2.xcoord+v3.xcoord)/2, (v2.ycoord+v3.ycoord)/2)
            v7 = Vertex((v3.xcoord+v4.xcoord)/2, (v3.ycoord+v4.ycoord)/2)
            v8 = Vertex((v4.xcoord+v1.xcoord)/2, (v4.ycoord+v1.ycoord)/2)
            
            self.N1 = 1/4*(1-self.si)*(1-self.eta)*(-1-self.si-self.eta)
            self.N2 = 1/4*(1+self.si)*(1-self.eta)*(-1+self.si-self.eta)
            self.N3 = 1/4*(1+self.si)*(1+self.eta)*(-1+self.si+self.eta)
            self.N4 = 1/4*(1-self.si)*(1+self.eta)*(-1-self.si+self.eta)
            self.N5 = 1/2*(1-self.si**2)*(1-self.eta)
            self.N6 = 1/2*(1+self.si)*(1-self.eta**2)
            self.N7 = 1/2*(1-self.si**2)*(1+self.eta)
            self.N8 = 1/2*(1-self.si)*(1-self.eta**2)
            
            self.eq1 = self.N1*v1.xcoord + self.N2*v2.xcoord + self.N3*v3.xcoord + self.N4*v4.xcoord + self.N5*v5.xcoord + self.N6*v6.xcoord + self.N7*v7.xcoord + self.N8*v8.xcoord - x
            self.eq2 = self.N1*v1.ycoord + self.N2*v2.ycoord + self.N3*v3.ycoord + self.N4*v4.ycoord + self.N5*v5.ycoord + self.N6*v6.ycoord + self.N7*v7.ycoord + self.N8*v8.ycoord - y
  
            self.eqs = (self.eq1, self.eq2)
   
            self.ans = solve(self.eqs, self.si, self.eta)
            
        if (type(self.ans) == dict):

            self.ans = list(self.ans.values())
                    
            self.si_val = float(self.ans[0])
            self.eta_val = float(self.ans[1])
            
        else : 
                
            if -1.0 <= float(self.ans[0][0]) <= 1.0 :
                
                self.si_val = float(self.ans[0][0])
                self.eta_val = float(self.ans[0][1])
            else : 
                self.si_val = float(self.ans[1][0])
                self.eta_val = float(self.ans[1][1])
                
                
        return self.si_val, self.eta_val
            

class ElementStiffnessMatrix :
    """
    class used to generate the stiffness matrix for each element in the mesh
    """
    
    def __init__(self, ele_type, ele_node, tk, D) :
        """
        initializes the class with element and system properties 
        """
        self.ele_type = ele_type
        self.ele_node = ele_node
        self.ele_tk = tk
        self.Dmat = D 
        
        self.ele_dof = self.ele_node*2
        self.ele_k = np.zeros([self.ele_dof, self.ele_dof])
    
    def get_stiffness_matrix(self, v1, v2, v3, v4) :
        """
        Returns the element stiffness matrix 
        
        v1, v2, v3, v4 - coordinates of the corner points of the element
        
        stiffness, k = B^T*D*B*A*t
        """
        
        if self.ele_type == "Triangular" and self.ele_node == 3 :
            # 3 node triangular element
                      
            self.e = Edge(v1, v2, v3, v4)
            self.ele_area = self.e.ele_area()
            self.sf = ShapeFunction(self.ele_type, self.ele_node)
            self.ele_N, self.Ne, self.ele_B, self.det_J = self.sf.get_shape_function(v1, v2, v3, v4, 0, 0)
            self.ele_k = (self.ele_B.transpose()@self.Dmat@self.ele_B)*self.ele_area*self.ele_tk
       
        elif self.ele_type == "Triangular" and self.ele_node == 6 :
            # 6 node triangular element
            # 3 point gauss quadrature used for integration
            # done using natural coordinates
            
            self.g_L1 = np.array([1/2, 1/2, 0]) # gauss point 1
            self.g_L2 = np.array([1/2, 0, 1/2]) # gauss point 2
            self.g_L3 = np.array([0, 1/2, 1/2]) # gauss point 3
            self.g_w = np.array([1/3, 1/3, 1/3]) # weight for the gauss points
            
            for i in range(len(self.g_L1)) :
                # loop goes through each each gauss point, calculates and sums the stiffness
                
                self.sf = ShapeFunction(self.ele_type, self.ele_node)
                self.ele_N, self.Ne, self.ele_B, self.det_J = self.sf.get_shape_function(v1, v2, v3, v4, self.g_L1[i], self.g_L2[i])
                self.ele_Bt = self.ele_B.transpose()
                self.ele_k += self.ele_tk*((self.ele_Bt@self.Dmat@self.ele_B)*self.det_J)*self.g_w[i]           
        
        elif self.ele_type in ["Rectangular", "Quadrilateral"] and self.ele_node == 4 :
            # 4 node quadrilateral element
            # 2 point gauss quadrature used for integration
            # done using natural coordinates

            self.g_si = np.array([-1/np.sqrt(3), 1/np.sqrt(3)]) # gauss points for si axis
            self.g_eta = np.array([-1/np.sqrt(3), 1/np.sqrt(3)]) # gauss points for eta axis
            self.g_w = np.array([1, 1]) # weights of the gauss points
            
            for i in range(len(self.g_si)) :
                # loop goes through each each gauss point, calculates and sums the stiffness
    
                for j in range(len(self.g_eta)) :    
                                        
                    self.sf = ShapeFunction(self.ele_type, self.ele_node)
                    self.ele_N, self.Ne, self.ele_B, self.det_J = self.sf.get_shape_function(v1, v2, v3, v4, self.g_si[i], self.g_eta[j])
                    self.ele_Bt = self.ele_B.transpose()
                    self.ele_k += self.ele_tk*((self.ele_Bt@self.Dmat@self.ele_B)*self.det_J)*self.g_w[j]*self.g_w[i]
        
        elif self.ele_type == "Quadrilateral" and self.ele_node == 8 :
            # 8 node quadrilateral element
            # 3 point gauss quadrature used for integration
            # done using natural coordinates
            
            self.g_si = np.array([-np.sqrt(0.6), 0, np.sqrt(0.6)]) # gauss points for si axis
            self.g_eta = np.array([-np.sqrt(0.6), 0, np.sqrt(0.6)]) # gauss points for eta axis
            self.g_w = np.array([5/9, 8/9, 5/9]) # weights of the gauss points
            
            for i in range(len(self.g_si)) :
                # loop goes through each each gauss point, calculates and sums the stiffness
    
                for j in range(len(self.g_eta)) :
                   
                    self.sf = ShapeFunction(self.ele_type, self.ele_node)
                    self.ele_N, self.Ne, self.ele_B, self.det_J = self.sf.get_shape_function(v1, v2, v3, v4, self.g_si[i], self.g_eta[j])
                    self.ele_Bt = self.ele_B.transpose()
                    self.ele_k += self.ele_tk*((self.ele_Bt@self.Dmat@self.ele_B)*self.det_J)*self.g_w[j]*self.g_w[i]
                  
        return self.ele_k
    

class Solver : 
    """
    class used to assemble the global matrices and solve for displacement and reaction at the global nodes 
    """
    
    def __init__(self, ele_type, ele_node, nel, k, f, mesh, node_number, disp_bc) :
        """
        Initializes the class with element and mesh properties 
        """
        
        self.ele_type = ele_type
        self.ele_node = ele_node
        self.node_number = node_number
        self.nel = nel
        
        self.ele_k = k
        self.ele_f = f
        self.mesh = mesh
        self.disp_bc = disp_bc
        
        self.ele_dof = self.ele_node*2
        
        # Calculating global nodes in the domain based on the type of element used
        
        if self.ele_node == 4 :
            self.gl_nodes = (nel_x+1)*(nel_y+1)
        elif self.ele_node == 3 :
            self.gl_nodes = (int(nel_x/2)+1)*(nel_y+1)
        elif self.ele_node == 8 :
            self.gl_nodes = (2*nel_x+1)*(nel_y+1) + nel_y*(nel_x+1)
        elif self.ele_node == 6 :
            self.nel_x = int(nel_x/2)
            self.gl_nodes = (2*self.nel_x+1)*(2*nel_y+1)
        
        self.gl_dof = self.gl_nodes*2 # global degrees of freedom
    
    def get_global_stiffness(self) :
        """
        Assembles and returns the global stiffness matrix, gl_stiff, from the elemental matrices 
        """
        self.gl_stiff = np.zeros([self.gl_dof, self.gl_dof])
        
        for i in range(self.nel) :
            # Loop goes over the stiffness matrix of each element and joins the values of connected nodes as well.

            stiff = self.ele_k[:,:,i]
            dof = np.zeros(self.ele_dof)
            k = 0
            
            for j in range(self.ele_node):
                
                n = self.node_number[j,i]
                nx = (n-1)*2 + 1
                ny =  (n-1)*2 + 2
                dof[k] = nx
                dof[k+1] = ny
                k+=2

            for p in range(self.gl_dof) :
                p1 = p+1
                for q in range(self.gl_dof) :
                    q1 = q+1
                    p_pos = np.array(np.where(dof==p1)[0])
                    q_pos = np.array(np.where(dof==q1)[0])

                    if p_pos.size > 0 and q_pos.size > 0 :
                            self.gl_stiff[p, q] += stiff[p_pos[0],q_pos[0]]
 
        return self.gl_stiff
        
    def get_global_force(self) :
        """
        Assembles and returns the global Force vector, gl_force, from the elemental vectors
        """

        self.gl_force = np.zeros([self.gl_dof,1])

        for i in range(self.nel) :
            # Loop goes over the force vector of each element and joins the values of connected nodes as well.

            force = self.ele_f[:,:,i]
            dof = np.zeros(self.ele_dof)
            k = 0
            
            for j in range(self.ele_node):
                
                n = self.node_number[j,i]
                nx = (n-1)*2 + 1
                ny =  (n-1)*2 + 2
                dof[k] = nx
                dof[k+1] = ny
                k+=2
            
            for p in range(self.gl_dof) :
                p1 = p+1
                p_pos = np.array(np.where(dof==p1)[0])
                
                if p_pos.size > 0 :
                            self.gl_force[p,0] += force[p_pos[0]]

        return self.gl_force
    
    def get_global_bc(self) :
        """
        Assembles and returns the global boundary condition vector, from the elemental vectors
        """

        self.gl_bc = np.zeros([self.gl_dof,1])

        for i in range(self.nel) :
            # Loop goes over the force vector of each element and joins the values of connected nodes as well.

            bc = self.disp_bc[:,i]
            dof = np.zeros(self.ele_dof)
            k = 0
            
            for j in range(self.ele_node):
                
                n = self.node_number[j,i]
                nx = (n-1)*2 + 1
                ny =  (n-1)*2 + 2
                dof[k] = nx
                dof[k+1] = ny
                k+=2
            
            for p in range(self.gl_dof) :
                p1 = p+1
                p_pos = np.array(np.where(dof==p1)[0])
                
                if p_pos.size > 0 :
                            self.gl_bc[p,0] = bc[p_pos[0]]
            

        return self.gl_bc
    
    def get_nodal_displacement(self) :
        """
        Solves the linear-elastic system using FEM .
        
        Returns the displacement values at all global dof, U
        
        Returns the reactio values at global dof, R
        """
        
        self.K = self.get_global_stiffness() # Global stiffness matrix

        self.F = self.get_global_force() # Global Force Vector

        self.BC = self.get_global_bc() # Global boundary Condition vector

        self.U = np.zeros([self.gl_dof,1]) # global displacement vector
        self.R = np.zeros([self.gl_dof,1]) # global reaction vector
        
        self.fixeddof = np.where(self.BC == 0)[0] # positions of displacement restraints
        self.freedof = np.where(self.BC != 0)[0] # positions of displacement freedom
        
        self.R1 = np.delete(self.R, self.fixeddof)
        self.F1 = np.delete(self.F, self.fixeddof)
        self.K1 = self.K
  
        for i in range(len(self.fixeddof)) :
            # The loop removes the rows and columns corresponding to restricted displacements from the global Stiffness Matrix
            
            j = self.fixeddof[i]
            self.K1 = np.delete(self.K1,j-i,0)
            self.K1 = np.delete(self.K1,j-i,1)
            
        self.U1 = np.linalg.inv(self.K1)@(self.R1 + self.F1).reshape(len(self.freedof),1) # Solve using KU - F = R
        # Defines the global displacement vector and assigns calculated and known displacement
        self.U[self.freedof] = self.U1
        self.U[self.fixeddof] = 0
        
        # Defines the global residue(reaction) vector, calculates and assigns the reaction at fixed and free nodes
        self.R1 = self.K@self.U - self.F # R = KU-F
        
        self.R[self.fixeddof] = self.R1[self.fixeddof]
        self.R[self.freedof] = 0
        
        return self.U, self.R # returns the global displacement and reaction vector


class Results :
    """
    class used to generate displacement/strain/stress at any point in the domain after FE analysis
    """
    
    def __init__(self, ele_type, ele_node, node_number, ele_end_coord, U, elast) :
        """
        initializes the class with mesh and system properties and FE solution
        """
        
        self.ele_type = ele_type
        self.ele_node = ele_node
        self.nel = nel
        self.ele_end_coord = ele_end_coord
        self.node_number = node_number
        
        self.end_node = 4 if self.ele_node in [4,8] else 3
        
        self.U = U
        self.elast = elast
    
    def get_ele_disp(self, x, y) :
        """
        returns the displacement vector and coordinates of the element containing the point (x, y)
        
        """
        self.ele_no = 0 # provided value for 0th element if the point is outside domain
        
        self.p = Point(x,y) # takes the coordinate point corresponding to (x,y)
        for i in range(self.nel) :
            # loop travels through all the elements to find out the particular element containing (x,y)
            
            ele_coord = self.ele_end_coord[:,:,i]
            
            p1 = Point(ele_coord[0,0], ele_coord[0,1])
            p2 = Point(ele_coord[1,0], ele_coord[1,1])
            p3 = Point(ele_coord[2,0], ele_coord[2,1])
            if self.ele_type == "Triangular" :
                p4 = Point(ele_coord[0,0], ele_coord[0,1])  
            else :
                p4 = Point(ele_coord[3,0], ele_coord[3,1])
            
            ele_coord = [p1,p2,p3,p4]
            ele_poly = Polygon(ele_coord)
          
            if(self.p.intersects(ele_poly)) :
                self.ele_no = i # element containing (x,y)
                break;
        
        self.pos_coord = self.ele_end_coord[:,:,self.ele_no] # coordinates of the element
        self.pos_node = self.node_number[:,self.ele_no] # node numbers for the element
        
        self.ele_disp = np.zeros([self.ele_node*2, 1])

        for i in range(self.ele_node) :
            node = int((self.pos_node[i]-1)*2)
            k = i*2
            self.ele_disp[k,0] = self.U[node]
            self.ele_disp[k+1,0] = self.U[node+1]
        
        return self.ele_disp, self.pos_coord
    
    def get_displacement(self, x, y) :
        """
        returns the displacement at the point (x, y) in the domain 
        
        u = N1u1 + N2u2 +.....
        v = N1v1 + N2v2 +.....
        
        """
        
        self.ele_disp, self.pos_coord = self.get_ele_disp(x, y)

        v1 = Vertex(self.pos_coord[0,0],self.pos_coord[0,1])
        v2 = Vertex(self.pos_coord[1,0],self.pos_coord[1,1])
        v3 = Vertex(self.pos_coord[2,0],self.pos_coord[2,1])
        if self.ele_type == "Triangular" :
            v4 = Vertex(self.pos_coord[0,0],self.pos_coord[0,1]) 
        else :
            v4 = Vertex(self.pos_coord[3,0],self.pos_coord[3,1])
        
        self.sf = ShapeFunction(self.ele_type, self.ele_node)
        
        self.si, self.eta = self.sf.get_si_eta(x, y, v1, v2, v3, v4)
        
        self.N, self.Ne_pos, self.B, self.det_J = self.sf.get_shape_function(v1,v2,v3,v4,self.si,self.eta)
        self.pos_disp = np.zeros([2,1])
        
        for i in range(self.ele_node):
            k = i*2
            
            self.pos_disp[0] += self.ele_disp[k]*self.Ne_pos[i]
            self.pos_disp[1] += self.ele_disp[k+1]*self.Ne_pos[i]
            
        return self.pos_disp
    
    def get_strain(self, x, y) : 
        """
        returns the strain at the point (x, y) in the domain 
        
        strain = B*disp
        
        """
        
        self.ele_disp, self.pos_coord = self.get_ele_disp(x, y)
        
        v1 = Vertex(self.pos_coord[0,0],self.pos_coord[0,1])
        v2 = Vertex(self.pos_coord[1,0],self.pos_coord[1,1])
        v3 = Vertex(self.pos_coord[2,0],self.pos_coord[2,1])
        if self.ele_type == "Triangular" :
            v4 = Vertex(self.pos_coord[0,0],self.pos_coord[0,1]) 
        else :
            v4 = Vertex(self.pos_coord[3,0],self.pos_coord[3,1])
    
    
        self.sf = ShapeFunction(self.ele_type, self.ele_node)
        
        self.si, self.eta = self.sf.get_si_eta(x, y, v1, v2, v3, v4)
        
        self.B, self.Ne_pos, self.B_pos, self.det_J = self.sf.get_shape_function(v1,v2,v3,v4,self.si,self.eta)
   
        self.pos_strain = self.B_pos@self.ele_disp 
        
        return self.pos_strain
    
    def get_stress(self, x, y) :
        """
        returns the stress at the point (x, y) in the domain 
        
        stress = strain*elasticity
        
        """
        
        self.pos_strain = self.get_strain(x, y) # obtains strain (e) at the point
        self.pos_stress = self.pos_strain*self.elast # stress, s = E*e
        
        return self.pos_stress

"""

MAIN FUNCTION

Execution of the FE-Method go here. This is achieved by creating objects for the classes defined above.

For user-defined parameters that has multiple options, the options are mentioned as comments nearby the code.

"""

#################################
#### USER DEFINITION GO HERE ####
#################################

# End Coordinates of Domain - 4 points for quadrilatel, 3 for triangular

x1, y1 = 0, 0 # First Point of the domain, bottom left point
x2, y2 = 2.0, 0 # Second point of the domain, bottom right point
x3, y3 = 2.0, 2.0 # Third point of the domain (top right for quadrilateral)
x4, y4 = 0.0, 2.0 # Fourth point of the domain (same as x1,y1 for triangular domain)

tk = 1.0 # thickness of the domain. Assume uniform thickness 
elast = 210e9 # elasticity
mu = 0.3 # poisson's - ratio
plane = "plane-stress" # plane- condition (plane-stress/plane-strain)

nel_x = 2 # No of elements in X direction
nel_y = 1 # No of elements in Y direction

ele_type = "Quadrilateral" # Type of Element : Triangular, Rectangular, Quadrilateral
ele_node = 8 # No: of nodes in the element : Triangular(3,6), Rectangular(4), Quadrilateral(4,8)

# Restraints on End Coordinates in X and Y respectively. 0 denotes restraint, 1 denotes freedom. 

res_1 = np.array([1, 1])
res_2 = np.array([1, 1])
res_3 = np.array([1, 1])
res_4 = np.array([1, 1])

# Restraints on Edges, anticlockwise from left bottom. 0 denotes restraint, 1 denotes freedom. 

eres_1 = np.array([0, 1])
eres_2 = np.array([1, 1])
eres_3 = np.array([1, 1])
eres_4 = np.array([1, 1])

# External Loading on Edges, anticlockwise from left bottom.

eload_1 = np.array([0, 0])
eload_2 = np.array([0, 0])
eload_3 = np.array([0, 0])
eload_4 = np.array([-3, 0])

#################################
### USER DEFINITIONs END HERE ###
#################################

dom_1 = Vertex(x1, y1)
dom_2 = Vertex(x2, y2)
dom_3 = Vertex(x3, y3)
dom_4 = Vertex(x4, y4) 

nel = nel_x*nel_y # no: of elements in the mesh
ele_dof = ele_node*2

# Calculating no:of nodes in the entire mesh

if ele_node == 4 :
    gl_node = (nel_x+1)*(nel_y+1) 
elif ele_node == 3 :
    gl_node = (int(nel_x/2)+1)*(nel_y+1)
elif ele_node == 8 :
    gl_nodes = (2*nel_x+1)*(nel_y+1) + nel_y*(nel_x+1)
elif ele_node == 6 :
    nel_x1 = int(nel_x/2)
    gl_node = (2*nel_x1+1)*(2*nel_y+1)
    

domain = Edge(dom_1, dom_2, dom_3, dom_4)
dom_area = domain.ele_area()
print("Area of the given domain is :", dom_area, "square units")

# Obtain coordinates of the element and node numbering
if ele_type in ["Quadrilateral", "Rectangular"] :
    
    endmesh = QuadrilateralMesh(nel_x, nel_y, ele_node)
    mesh_coord, ele_end_coord, node_number = endmesh.mesh_generate(dom_1, dom_2, dom_3, dom_4)
    fullmesh = CompleteMesh(ele_type, ele_node, nel, mesh_coord, ele_end_coord)
    mesh = fullmesh.get_mesh()

elif ele_type == "Triangular" :
    
    endmesh = TriangularMesh(nel_x, nel_y, ele_node)
    mesh_coord, ele_end_coord, node_number = endmesh.mesh_generate(dom_1, dom_2, dom_3, dom_4)
    fullmesh = CompleteMesh(ele_type, ele_node, nel, mesh_coord, ele_end_coord)
    mesh = fullmesh.get_mesh()

else : 
    
    print("Undefined Element Type. Program Terminating")
    exit()


# obaining system properties
sys = System(elast, mu, plane)
d_mat  = sys.get_system()

#obtaining boundary condition characteristics
bc = BoundaryCondition(dom_1, dom_2, dom_3, dom_4, ele_type, ele_node, mesh, nel)
mesh_bc = bc.get_disp_bc(res_1, res_2, res_3, res_4, eres_1, eres_2, eres_3, eres_4)

#obtaining loading values and characteristics
load = Loading(dom_1, dom_2, dom_3, dom_4, ele_type, ele_node, mesh, nel_x, nel_y)
mesh_load = load.get_element_load(eload_1, eload_2, eload_3, eload_4)

ele_k = np.zeros([ele_dof, ele_dof, nel]) # multi-dimensional array to store all element-stiffness matrices
ele_f = np.zeros([ele_dof,1, nel])# multi-dimensional array to store all element-force vectors

for i in range(nel) :
    # The loop travels through each element in the mesh and generates element stiffness matrix and element force vector which is stored in multi-dimensional arrays k and f respectively.
    
    ele_flag = i +1 # element number in this loop
    
    ele_coord = mesh[:,:,i]
    n1_coord = ele_coord[0,:]
    n2_coord = ele_coord[1,:]
    n3_coord = ele_coord[2,:]
    if ele_type in ["Quadrilateral","Rectangular"] :
        n4_coord = ele_coord[3,:]
    else :
        n4_coord = n1_coord
    n1 = Vertex(n1_coord[0], n1_coord[1])
    n2 = Vertex(n2_coord[0], n2_coord[1])
    n3 = Vertex(n3_coord[0], n3_coord[1])
    n4 = Vertex(n4_coord[0], n4_coord[1])
        
    ele_stiff = ElementStiffnessMatrix(ele_type, ele_node, tk, d_mat)

    ele_k[:,:,i] = ele_stiff.get_stiffness_matrix(n1, n2, n3, n4) # stiffness matrix of i+1 th element
    #print("Stiffness Matrix of element", ele_flag, "\n", ele_k[:,:,i], "\n", "\n")
    
    ele_f[:,:,i] = mesh_load[:,i].reshape(ele_dof,1) # force vector of i+1 th element
    #print("Force vector of element", ele_flag, "\n", ele_f[:,:,i])

# Outside the loop.

sol = Solver(ele_type, ele_node, nel, ele_k, ele_f, mesh, node_number, mesh_bc) # Solver class is called to generate global K and global F.

U, R = sol.get_nodal_displacement() # Displacement vector and residue/reaction vector

print("Displacement Vector is : ",U)
print("Reaction Vector is : ", R)

res = Results(ele_type, ele_node, node_number, ele_end_coord, U, elast) # To generate results

x, y = 0,0 # Point in the domain where results need to be found out

print("Displacement is ", res.get_displacement(x, y)) # Prints displacement value at (x,y)
print("Strain is", res.get_strain(x, y)) # Prints strain value at (x,y)
print("Stress is", res.get_stress(x, y)) # Prints stress value at (x,y)

# Generating results for the domain

x = np.linspace(0, x2, 101) # set of points in the domain where results need to be found out
y = np.linspace(0, y3, 101)

u = np.zeros([len(x), len(y)])
v = np.zeros([len(x), len(y)])
sx = np.zeros([len(x), len(y)])
sy = np.zeros([len(x), len(y)])
sxy = np.zeros([len(x), len(y)])
ex = np.zeros([len(x), len(y)])
ey = np.zeros([len(x), len(y)])
exy = np.zeros([len(x), len(y)])

for i in range(len(x)) :
    for j in range(len(y)) :
        # Calculates displacement, strain and stress at every combination of (x,y)
        # Returns value at first element if (x,y) not in the domain 
        
        disp = res.get_displacement(x[i], y[j])
        u[i,j] = disp[0]
        v[i,j] = disp[1]
        stress = res.get_stress(x[i],y[j])
        strain = res.get_strain(x[i],y[j])
        sx[i,j] = stress[0]
        sy[i,j] = stress[1]
        sxy[i,j] = stress[2]
        ex[i,j] = strain[0]
        ey[i,j] = strain[1]
        exy[i,j] = strain[2]

# Drawing contour plots for the domain 

Y, X = np.meshgrid(y,x)

fig,ax=plt.subplots(1,1)
mycmap2 = plt.get_cmap('inferno')
cp = ax.contourf(X, Y, u, cmap=mycmap2)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Axial Displacement')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
#plt.show()
plt.savefig('4_2_u_101.png', dpi=300, bbox_inches='tight')

fig,ax=plt.subplots(1,1)
mycmap2 = plt.get_cmap('inferno')
cp = ax.contourf(X, Y, v, cmap=mycmap2)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Vertical Displacement')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
#plt.show()
plt.savefig('4_v_2_101.png', dpi=300, bbox_inches='tight')

fig,ax=plt.subplots(1,1)
mycmap2 = plt.get_cmap('inferno')
cp = ax.contourf(X, Y, sx, cmap=mycmap2)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Axial Stress')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
#plt.show()
plt.savefig('4_sx_2_101.png', dpi=300, bbox_inches='tight')

fig,ax=plt.subplots(1,1)
mycmap2 = plt.get_cmap('inferno')
cp = ax.contourf(X, Y, sy, cmap=mycmap2)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Vertical Stress')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
#plt.show()
plt.savefig('4sy_2_101.png', dpi=300, bbox_inches='tight')

fig,ax=plt.subplots(1,1)
mycmap2 = plt.get_cmap('inferno')
cp = ax.contourf(X, Y, sxy, cmap=mycmap2)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Shear Stress')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
#plt.show()
plt.savefig('4_sxy_2_101.png', dpi=300, bbox_inches='tight')