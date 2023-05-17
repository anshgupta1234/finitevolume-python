import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import itertools
import imageio
import cv2

"""
Create Your Own Finite Volume Fluid Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the Kelvin Helmholtz Instability
In the compressible Euler equations

"""

frames = []
size = (0, 0)


@dataclass
class GhostPoint:
	pos: tuple
	ip_pos: tuple
	interp_coeffs: list

@dataclass
class BoundaryPoint:
	pos: tuple
	norm: tuple

def getConserved( rho, vx, vy, P, gamma, vol ):
	"""
    Calculate the conserved variable from the primitive
	rho      is matrix of cell densities
	vx       is matrix of cell x-velocity
	vy       is matrix of cell y-velocity
	P        is matrix of cell pressures
	gamma    is ideal gas gamma
	vol      is cell volume
	Mass     is matrix of mass in cells
	Momx     is matrix of x-momentum in cells
	Momy     is matrix of y-momentum in cells
	Energy   is matrix of energy in cells
	"""
	Mass   = rho * vol
	Momx   = rho * vx * vol
	Momy   = rho * vy * vol
	Energy = (P/(gamma-1) + 0.5*rho*(vx**2+vy**2))*vol
	
	return Mass, Momx, Momy, Energy


def getPrimitive( Mass, Momx, Momy, Energy, gamma, vol ):
	"""
    Calculate the primitive variable from the conservative
	Mass     is matrix of mass in cells
	Momx     is matrix of x-momentum in cells
	Momy     is matrix of y-momentum in cells
	Energy   is matrix of energy in cells
	gamma    is ideal gas gamma
	vol      is cell volume
	rho      is matrix of cell densities
	vx       is matrix of cell x-velocity
	vy       is matrix of cell y-velocity
	P        is matrix of cell pressures
	"""
	rho = Mass / vol
	vx  = Momx / rho / vol
	vy  = Momy / rho / vol
	P   = (Energy/vol - 0.5*rho * (vx**2+vy**2)) * (gamma-1)
	
	return rho, vx, vy, P


def getGradient(f, dx, dy):
	"""
    Calculate the gradients of a field
	f        is a matrix of the field
	dx       is the cell size
	f_dx     is a matrix of derivative of f in the x-direction
	f_dy     is a matrix of derivative of f in the y-direction
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_dx = ( np.roll(f,R,axis=0) - np.roll(f,L,axis=0) ) / (2*dx)
	f_dy = ( np.roll(f,R,axis=1) - np.roll(f,L,axis=1) ) / (2*dy)
	
	return f_dx, f_dy


def slopeLimit(f, dx, dy, f_dx, f_dy):
	"""
    Apply slope limiter to slopes
	f        is a matrix of the field
	dx       is the cell size
	f_dx     is a matrix of derivative of f in the x-direction
	f_dy     is a matrix of derivative of f in the y-direction
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_dx = np.maximum(0., np.minimum(1., ( (f-np.roll(f,L,axis=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
	f_dx = np.maximum(0., np.minimum(1., (-(f-np.roll(f,R,axis=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
	f_dy = np.maximum(0., np.minimum(1., ( (f-np.roll(f,L,axis=1))/dy)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
	f_dy = np.maximum(0., np.minimum(1., (-(f-np.roll(f,R,axis=1))/dy)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
	
	return f_dx, f_dy


def extrapolateInSpaceToFace(f, f_dx, f_dy, dx, dy):
	"""
    Calculate the gradients of a field
	f        is a matrix of the field
	f_dx     is a matrix of the field x-derivatives
	f_dy     is a matrix of the field y-derivatives
	dx       is the cell size
	f_XL     is a matrix of spatial-extrapolated values on `left' face along x-axis 
	f_XR     is a matrix of spatial-extrapolated values on `right' face along x-axis 
	f_YL     is a matrix of spatial-extrapolated values on `left' face along y-axis 
	f_YR     is a matrix of spatial-extrapolated values on `right' face along y-axis 
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_XL = f - f_dx * dx/2
	f_XL = np.roll(f_XL,R,axis=0)
	f_XR = f + f_dx * dx/2
	
	f_YL = f - f_dy * dy/2
	f_YL = np.roll(f_YL,R,axis=1)
	f_YR = f + f_dy * dy/2
	
	return f_XL, f_XR, f_YL, f_YR
	

def applyFluxes(F, flux_F_X, flux_F_Y, dx, dy, dt):
	"""
    Apply fluxes to conserved variables
	F        is a matrix of the conserved variable field
	flux_F_X is a matrix of the x-dir fluxes
	flux_F_Y is a matrix of the y-dir fluxes
	dx       is the cell size
	dt       is the timestep
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	# update solution
	F += - dt * dx * flux_F_X
	F +=   dt * dx * np.roll(flux_F_X,L,axis=0)
	F += - dt * dy * flux_F_Y
	F +=   dt * dy* np.roll(flux_F_Y,L,axis=1)
	
	return F


def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
	"""
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule 
	rho_L        is a matrix of left-state  density
	rho_R        is a matrix of right-state density
	vx_L         is a matrix of left-state  x-velocity
	vx_R         is a matrix of right-state x-velocity
	vy_L         is a matrix of left-state  y-velocity
	vy_R         is a matrix of right-state y-velocity
	P_L          is a matrix of left-state  pressure
	P_R          is a matrix of right-state pressure
	gamma        is the ideal gas gamma
	flux_Mass    is the matrix of mass fluxes
	flux_Momx    is the matrix of x-momentum fluxes
	flux_Momy    is the matrix of y-momentum fluxes
	flux_Energy  is the matrix of energy fluxes
	"""
	
	# left and right energies
	en_L = P_L/(gamma-1)+0.5*rho_L * (vx_L**2+vy_L**2)
	en_R = P_R/(gamma-1)+0.5*rho_R * (vx_R**2+vy_R**2)

	# compute star (averaged) states
	rho_star  = 0.5*(rho_L + rho_R)
	momx_star = 0.5*(rho_L * vx_L + rho_R * vx_R)
	momy_star = 0.5*(rho_L * vy_L + rho_R * vy_R)
	en_star   = 0.5*(en_L + en_R)
	
	P_star = (gamma-1)*(en_star-0.5*(momx_star**2+momy_star**2)/rho_star)
	
	# compute fluxes (local Lax-Friedrichs/Rusanov)
	flux_Mass   = momx_star
	flux_Momx   = momx_star**2/rho_star + P_star
	flux_Momy   = momx_star * momy_star/rho_star
	flux_Energy = (en_star+P_star) * momx_star/rho_star
	
	# find wavespeeds
	C_L = np.sqrt(gamma*P_L/rho_L) + np.abs(vx_L)
	C_R = np.sqrt(gamma*P_R/rho_R) + np.abs(vx_R)
	C = np.maximum( C_L, C_R )
	
	# add stabilizing diffusive term
	flux_Mass   -= C * 0.5 * (rho_L - rho_R)
	flux_Momx   -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
	flux_Momy   -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
	flux_Energy -= C * 0.5 * ( en_L - en_R )

	return flux_Mass, flux_Momx, flux_Momy, flux_Energy

def interpolate_image_point(pos, markers):
	dist = np.zeros(shape=(2,2), dtype=float)
	alpha = np.zeros(shape=(2,2), dtype=int)
	interp_coeff = np.zeros(shape=(2,2), dtype=float)

	# Initialize Indices
	i1, j1 = np.floor(pos).astype(int)
	i2 = i1 + 1
	j2 = j1 + 1

	# Calculate distances to each of the corners
	dist[0][0] = np.sqrt(sum((pos - (i1, j1))**2))
	dist[1][0] = np.sqrt(sum((pos - (i2, j1))**2))
	dist[0][1] = np.sqrt(sum((pos - (i1, j2))**2))
	dist[1][1] = np.sqrt(sum((pos - (i2, j2))**2))

	for i, j in itertools.product([0, 1], [0, 1]):
		if dist[i][j] <= 0.05:
			interp_coeff[i][j] = 1
			break
	else:
		eta = 1. / dist**2
		# alpha 0 if point inside ibm, 1 if not
		alpha[0][0] = 0 if markers[i1][j1] == 1 else 1
		alpha[1][0] = 0 if markers[i2][j1] == 1 else 1
		alpha[0][1] = 0 if markers[i1][j2] == 1 else 1
		alpha[1][1] = 0 if markers[i2][j2] == 1 else 1
		buf_mat = alpha * eta
		buf = np.sum(buf_mat)
		if buf > 0:
			interp_coeff = buf_mat / buf
		else:
			buf = np.sum(eta)
			interp_coeff = eta / buf

	return interp_coeff

def calculate_image_point_values(gp, rho, P, vx, vy):
	interp = gp.interp_coeffs
	x, y = np.floor(gp.ip_pos).astype(int)
	rho_IP = interp[0][0] * rho[x][y] + \
		interp[1][0] * rho[x + 1][y] + \
		interp[0][1] * rho[x][y + 1] + \
		interp[1][1] * rho[x + 1][y + 1]

	P_IP = interp[0][0] * P[x][y] + \
		interp[1][0] * P[x + 1][y] + \
		interp[0][1] * P[x][y + 1] + \
		interp[1][1] * P[x + 1][y + 1]
	
	Ux_IP = interp[0][0] * vx[x][y] + \
		interp[1][0] * vx[x + 1][y] + \
		interp[0][1] * vx[x][y + 1] + \
		interp[1][1] * vx[x + 1][y + 1]
	
	Uy_IP = interp[0][0] * vy[x][y] + \
		interp[1][0] * vy[x + 1][y] + \
		interp[0][1] * vy[x][y + 1] + \
		interp[1][1] * vy[x + 1][y + 1]
	
	return rho_IP, P_IP, Ux_IP, Uy_IP

def main():
	""" Finite Volume simulation """
	
	# Simulation parameters
	Nx                     = 250 # resolution
	Ny                     = 100 # resolution
	boxsize                = 1.
	gamma                  = 5/3 # ideal gas gamma
	courant_fac            = 0.4
	t                      = 0
	tEnd                   = 0.4
	tOut                   = 0.02 # draw frequency
	useSlopeLimiting       = False
	u_inlet				   = 0.5
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Mesh
	dx = boxsize / Nx
	dy = boxsize / Ny
	vol = dx*dy
	xlin = np.linspace(0.5*dx, boxsize-0.5*dx, Nx)
	ylin = np.linspace(0.5*dy, boxsize-0.5*dy, Ny)
	Y, X = np.meshgrid( ylin, xlin )

	# Generate Initial Conditions
	rho = np.ones(X.shape)
	vx = 0. + (np.abs(X - 0.5) < 0.2)
	vy = 0
	P = 2.5 * np.ones(X.shape)

	# Generate immersed boundary markers
	R = 15
	ib_pos = (200, 50)
	markers = 0 + (np.sqrt((X*Nx - ib_pos[0])**2 + (Y*Ny - ib_pos[1])**2) < R)
	dist_from_center = np.sqrt((X*Nx - ib_pos[0])**2 + (Y*Ny - ib_pos[1])**2)
	levelset = dist_from_center - R
	levelsetNormal = levelsetNormal = [[(0, 0) for j in range(Ny)] for i in range(Nx)]
	for i in range(Nx):
		for j in range(Ny):
			if levelset[i][j] == 0:
				levelsetNormal[i][j] = (0, 0)
			else:
				levelsetNormal[i][j] = (X[i][j]*Nx - ib_pos[0], Y[i][j]*Ny - ib_pos[1]) / dist_from_center[i][j]

	# Generate boundary points
	bp = []
	num_bps = 100
	for i in range(num_bps):
		theta = 2*np.pi*i/num_bps
		pos = (ib_pos[0] + R*np.cos(theta) - 0.5, ib_pos[1] + R*np.sin(theta) - 0.5)
		norm = (np.cos(theta), np.sin(theta))
		bp.append(BoundaryPoint(pos, norm))

	# Generate Ghost Points
	gp = []
	offset = list(itertools.product(list(range(-2, 3)), list(range(-2, 3))))
	for i in range(Nx):
		for j in range(Ny):
			if markers[i][j] == 1:
				for off_x, off_y in offset:
					if markers[i + off_x][j + off_y] == 0:
						dist = abs(levelset[i][j])
						pos = (i, j)
						ip = pos + 2 * dist * levelsetNormal[i][j]
						interp_coeffs = interpolate_image_point(ip, markers)
						gp.append(GhostPoint(pos, ip, interp_coeffs))
						break

	# Get conserved variables
	Mass, Momx, Momy, Energy = getConserved( rho, vx, vy, P, gamma, vol )
	
	# prep figure
	fig = plt.figure(figsize=(12,12), dpi=80)
	outputCount = 1
	out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 5, (80,80))
	
	# Simulation Main Loop
	while t < tEnd:
		
		# get Primitive variables
		rho, vx, vy, P = getPrimitive( Mass, Momx, Momy, Energy, gamma, vol )

		# Ghost Point Correction
		for p in gp:
			rho_IP, P_IP, Ux_IP, Uy_IP = calculate_image_point_values(p, rho, P, vx, vy)
			x, y = p.pos
			# Set rho and P
			rho[x][y] = rho_IP
			P[x][y] = P_IP
			# No Slip BC
			vx[x][y] = 0
			vy[x][y] = 0
		
		# get time step (CFL) = dx / max signal speed
		dt = courant_fac * np.min( dx / (np.sqrt( gamma*P/rho ) + np.sqrt(vx**2+vy**2)) )
		plotThisTurn = False
		if t + dt > outputCount*tOut:
			dt = outputCount*tOut - t
			plotThisTurn = True
		
		# calculate gradients
		rho_dx, rho_dy = getGradient(rho, dx, dy)
		vx_dx,  vx_dy  = getGradient(vx,  dx, dy)
		vy_dx,  vy_dy  = getGradient(vy,  dx, dy)
		P_dx,   P_dy   = getGradient(P,   dx, dy)
		
		# slope limit gradients
		if useSlopeLimiting:
			rho_dx, rho_dy = slopeLimit(rho, dx, dy, rho_dx, rho_dy)
			vx_dx,  vx_dy  = slopeLimit(vx , dx, dy, vx_dx,  vx_dy )
			vy_dx,  vy_dy  = slopeLimit(vy , dx, dy, vy_dx,  vy_dy )
			P_dx,   P_dy   = slopeLimit(P  , dx, dy, P_dx,   P_dy  )
		
		# extrapolate half-step in time
		rho_prime = rho - 0.5*dt * ( vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
		vx_prime  = vx  - 0.5*dt * ( vx * vx_dx + vy * vx_dy + (1/rho) * P_dx )
		vy_prime  = vy  - 0.5*dt * ( vx * vy_dx + vy * vy_dy + (1/rho) * P_dy )
		P_prime   = P   - 0.5*dt * ( gamma*P * (vx_dx + vy_dy)  + vx * P_dx + vy * P_dy )
		
		# extrapolate in space to face centers
		rho_XL, rho_XR, rho_YL, rho_YR = extrapolateInSpaceToFace(rho_prime, rho_dx, rho_dy, dx, dy)
		vx_XL,  vx_XR,  vx_YL,  vx_YR  = extrapolateInSpaceToFace(vx_prime,  vx_dx,  vx_dy,  dx, dy)
		vy_XL,  vy_XR,  vy_YL,  vy_YR  = extrapolateInSpaceToFace(vy_prime,  vy_dx,  vy_dy,  dx, dy)
		P_XL,   P_XR,   P_YL,   P_YR   = extrapolateInSpaceToFace(P_prime,   P_dx,   P_dy,   dx, dy)
		
		# compute fluxes (local Lax-Friedrichs/Rusanov)
		flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = getFlux(rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, gamma)
		flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = getFlux(rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, gamma)
		
		# update solution
		Mass   = applyFluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dy, dt)
		Momx   = applyFluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dy, dt)
		Momy   = applyFluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dy, dt)
		Energy = applyFluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dy, dt)
		
		# update time
		t += dt
		
		# plot in real time
		if (plotRealTime and plotThisTurn) or (t >= tEnd):
			vmag = np.sqrt(vx**2 + vy**2)
			vmag = vmag[160:240, 10:90]
			plt.cla()
			# plt.imshow(vmag.T)
			plt.quiver(X[175:225], Y[25:75], vx[175:225], vy[25:75])
			plt.clim(0.0, 1.4)
			# bp_x, bp_y = zip(*[p.pos for p in bp])
			# plt.scatter(bp_x, bp_y, s=4, color="black")
			# gp_x, gp_y = zip(*[p.pos for p in gp])
			# plt.scatter(gp_x, gp_y, s=4, color="white")
			# ip_x, ip_y = zip(*[p.ip_pos for p in gp])
			# plt.scatter(ip_x, ip_y, s=4, color="red")
			ax = plt.gca()
			ax.invert_yaxis()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)	
			ax.set_aspect('equal')
			plt.savefig(f'./img/img_{t}.png', dpi=240)
			plt.pause(0.1)
			img = cv2.imread(f'./img/img_{t}.png')
			frames.append(img)
			outputCount += 1
			
	
	# Save figure
	plt.savefig('finitevolume.png',dpi=240)
	plt.show()
	out.release()
	    
	return 0



if __name__== "__main__":
    main()
    height, width, layers = frames[0].shape
    size = (width, height)
    print(size)
    out = cv2.VideoWriter('noslip.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 1.25, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
    # imageio.mimsave('./sim.gif', # output gif
    #     frames,          # array of input frames
    #     duration = 1000)         # optional: frames per second

