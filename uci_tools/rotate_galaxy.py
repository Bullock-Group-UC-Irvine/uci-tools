import numpy as np

# Importing this here in case anything that uses this package expects 
# coord_to_r to be accessible from uci_tools.rotate_galaxy
from .fire_tools import coord_to_r

# ====================== function part ======================
def checklen(x):
    return len(np.array(x,ndmin=1));

def calculate_ang_mom(mass, coord, vel):
    '''
    Calculate specific angular momentum given masses, coordinates and 
    velocities
    '''
    if coord.shape[1]!=3:
        coord = coord.T
        vel = vel.T
    mom = np.zeros((coord.shape[0],3))
    mom[:,0] = mass * ( (coord[:,1]*vel[:,2]) - (coord[:,2]*vel[:,1]) )
    mom[:,1] = mass * ( (coord[:,2]*vel[:,0]) - (coord[:,0]*vel[:,2]) )
    mom[:,2] = mass * ( (coord[:,0]*vel[:,1]) - (coord[:,1]*vel[:,0]) )

    mom1 = np.sum(mom[:,0])
    mom2 = np.sum(mom[:,1])
    mom3 = np.sum(mom[:,2])

    return np.array((mom1,mom2,mom3))/np.sum(mass)

def vrrotvec(a,b):
    ''' Calculate to rotation vector that can rotate vector a to vector b '''

    if a.shape[0]==3 and b.shape[0]==3:
        an = a / np.linalg.norm(a,axis=0)
        bn = b / np.linalg.norm(b,axis=0)
    else:
        raise ValueError('Unexpected shape')

    # the cross product gives a vector perpendicular to both an and bn,
    # which is the vector around which an must be rotated to align with bn.
    axb = np.cross(an,bn) 

    ac = np.arccos(np.dot(an,bn)) #angle between the vectors

    # Now we know the vector/axis around which to rotate and the angle by which
    # we should rotate.
    return np.append(axb,ac)

def vrrotvec2mat(r):
    ''' 
    Convert rotation vector r to rotation matrix

    Parameters:
    r: np.array(list-like, float): r[0] is a 3D vector around which to rotate.
                                   r[1] is the angle by which to rotate
    '''

    print('rotation vector:')
    print(r)
    print('')

    s = np.sin(r[3])
    c = np.cos(r[3])
    t = 1-c

    n = r[0:3] / np.linalg.norm(r[0:3],axis=0)

    x = n[0]
    y = n[1]
    z = n[2]
    m = np.array( ((t*x*x + c, t*x*y - s*z, t*x*z + s*y),\
        (t*x*y + s*z, t*y*y + c, t*y*z - s*x),\
        (t*x*z - s*y, t*y*z + s*x, t*z*z + c)) )
    return m

def rotation_matrix_fr_vecs(a, b):
    '''
    Calculate rotation matrix that can rotate vector a to vector b
    '''

    return vrrotvec2mat(vrrotvec(a,b))

# Alias
cal_rotation_matrix = rotation_matrix_fr_vecs

def rotate(data, r):
    '''
    Rotate data with rotation matrix r
    '''

    if data.shape[1]!=3:
        return np.dot(r,data)
    else:
        return np.dot(r,data.T).T

# Alias
rotate_matrix = rotate

def rotation_matrix_fr_dat(coords_centered, v_vecs, masses, rs):
    '''
    Generate a rotation matrix from a galaxy's data
    '''

    # choose the stars within 10 kpc (or your choice of distance) 
    # from the center to calculate the disk orientation
    print('Masking')
    center_mask = rs<=10.0

    # calculate the average 3D angular momentum of the stars within 10 kpc
    # which is also the rotation axis
    print('Calculating angular momentum')
    disk_ang = calculate_ang_mom(
        masses[center_mask],
        coords_centered[center_mask,:],
        v_vecs[center_mask,:]
    )
    print('angular momentum:')
    print(disk_ang)
    print('')

    # calculate the rotation matrix that can rotate the galaxy so that
    # the rotation axis aligns with the Z axis
    print('Calculating rotation matrix')
    rotation_matrix = rotation_matrix_fr_vecs(
        disk_ang,
        np.array((0.0, 0.0, 1.0))
    )

    return rotation_matrix

def rotate_gal(coords_centered, v_vecs, masses, rs):
    '''
    Parameters
    ----------
    coords_centered: np.ndarray of shape (number_of_parties, 3)
        coordinates of the stars, with galaxy centered at 
        (0,0,0)
    v_vecs: np.ndarray of shape (number_of_particles, 3)
        velocities of the stars, with respective to the galaxy 
        center 
    masses: np.ndarray of shape (number_of_particles,)
        Star masses
    rs: np.ndarray of shape (number_of_particles,)
        3D distances of the stars from the center of the galaxy
    '''

    rotation_matrix = rotation_matrix_fr_dat(coords_centered, v_vecs, 
                                                 masses, rs)

    # get the new coordinates where the stellar disk lies in the XY plane
    print('Rotating')
    coord_rotated = rotate(coords_centered, rotation_matrix)
    v_vecs_rotated = rotate(v_vecs, rotation_matrix)

    return coord_rotated, v_vecs_rotated
