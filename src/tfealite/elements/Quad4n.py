import numpy as np

class Quad4n:
    def __init__(self, node_coords, material, real):
        self.node_coords = node_coords
        self.E  = material['E']
        self.nu = material['nu']
        self.rho = material['rho'] if ('rho' in material) else 0.0
        self.t = real['t']

        c1 = self.E / (1. - self.nu**2)
        self.C = c1 * np.array([
            [1.,      self.nu,  0.    ],
            [self.nu, 1.,       0.    ],
            [0.,      0.,   (1.-self.nu)/2. ]
        ])

    def cal_element_matrices(self, eval_mass = False):
        gauss_pts = [ -1/np.sqrt(3), 1/np.sqrt(3) ]
        Ke = np.zeros((8,8))
        Me = np.zeros((8,8)) if eval_mass else None
        x_e = self.node_coords
        for xi in gauss_pts:
            for eta in gauss_pts:
                N, dN_dxi = self.shape_functions(xi, eta)
                J = dN_dxi @ x_e
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                B = np.zeros((3,8))
                dN_dxy = invJ @ dN_dxi
                for i in range(4):
                    ix = 2*i
                    iy = 2*i + 1
                    B[0, ix] = dN_dxy[0,i]
                    B[1, iy] = dN_dxy[1,i]
                    B[2, ix] = dN_dxy[1,i]
                    B[2, iy] = dN_dxy[0,i]
                Ke += (B.T @ self.C @ B) * detJ
                if eval_mass:
                    rho_t = self.rho
                    N_2d = np.zeros((2,8))
                    for i in range(4):
                        N_2d[0,2*i  ] = N[i]
                        N_2d[1,2*i+1] = N[i]
                    Me += rho_t * (N_2d.T @ N_2d) * detJ
        if eval_mass:
            return Me, Ke
        else:
            return Ke

    def shape_functions(self, xi, eta):
        N = 0.25 * np.array([
            (1 - xi)*(1 - eta),
            (1 + xi)*(1 - eta),
            (1 + xi)*(1 + eta),
            (1 - xi)*(1 + eta),
        ])
        dN_dxi = 0.25 * np.array([
            [ -(1 - eta),  (1 - eta),  (1 + eta), -(1 + eta)],
            [ -(1 - xi),  -(1 + xi),   (1 + xi),  (1 - xi)]
        ])
        return N, dN_dxi

    def stresses_at_nodes(self, Ue):
        Ue = np.asarray(Ue, dtype=float).ravel()
        nat_coords = [
            (-1.0, -1.0),
            ( 1.0, -1.0),
            ( 1.0,  1.0),
            (-1.0,  1.0),
        ]
        sig = np.zeros((4, 3), dtype=float)
        for a, (xi, eta) in enumerate(nat_coords):
            _, dN_dxi = self.shape_functions(xi, eta)
            J = dN_dxi @ self.node_coords
            invJ = np.linalg.inv(J)
            dN_dxy = invJ @ dN_dxi
            B = np.zeros((3, 8), dtype=float)
            for i in range(4):
                ix = 2*i
                iy = 2*i + 1
                B[0, ix] = dN_dxy[0, i]
                B[1, iy] = dN_dxy[1, i]
                B[2, ix] = dN_dxy[1, i]
                B[2, iy] = dN_dxy[0, i]
            eps = B @ Ue
            sig[a, :] = self.C @ eps
        return sig