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
    Convert rotation vector r to rotation matrix.

    Parameters
    ----------
    r: np.array(list-like, float)
        r[0:3] is the 3D axis around which to rotate; r[3] is the
        angle in radians.
    '''
    s = np.sin(r[3])
    c = np.cos(r[3])
    t = 1 - c

    n = r[0:3] / np.linalg.norm(r[0:3], axis=0)

    x = n[0]
    y = n[1]
    z = n[2]
    m = np.array(
        ((t*x*x + c,   t*x*y - s*z, t*x*z + s*y),
         (t*x*y + s*z, t*y*y + c,   t*y*z - s*x),
         (t*x*z - s*y, t*y*z + s*x, t*z*z + c))
    )
    return m


def rotation_matrix_fr_vecs(a, b):
    '''
    3x3 rotation matrix R such that R @ a_hat == b_hat, where a_hat
    and b_hat are the normalized versions of a and b. Uses
    Rodrigues' rotation formula. Handles the degenerate case where
    a and b are antiparallel with a 180-degree rotation about a
    perpendicular axis.

    Note: the old implementation (vrrotvec2mat(vrrotvec(a, b)))
    produced NaN for parallel and antiparallel inputs because
    vrrotvec returns a zero-length axis in those cases. This
    implementation handles both correctly.

    Parameters
    ----------
    a: array-like, shape (3,)
        Source direction vector.
    b: array-like, shape (3,)
        Target direction vector.

    Returns
    -------
    R: np.ndarray, shape (3, 3)
        Rotation matrix satisfying R @ (a/||a||) == (b/||b||).
    '''
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-10:
        if c > 0.:
            return np.eye(3)
        # a is antiparallel to b: perform a 180-degree rotation
        # about an axis perpendicular to a.
        perp = (
            np.array([1., 0., 0.])
            if abs(a[0]) < 0.9
            else np.array([0., 1., 0.])
        )
        perp = perp - np.dot(perp, a) * a
        perp /= np.linalg.norm(perp)
        return 2. * np.outer(perp, perp) - np.eye(3)
    k = v / s
    K = np.array(
        [[ 0.,    -k[2],  k[1]],
         [ k[2],   0.,   -k[0]],
         [-k[1],   k[0],  0.  ]]
    )
    return c * np.eye(3) + (1. - c) * np.outer(k, k) + s * K


# rotation_matrix is the canonical name for new code.
# cal_rotation_matrix and rotation_matrix_fr_vecs stay for backward
# compatibility.
rotation_matrix = rotation_matrix_fr_vecs
cal_rotation_matrix = rotation_matrix_fr_vecs


def rotation_matrix_to_z(n):
    '''
    3x3 rotation matrix R such that R @ n_hat == z_hat. Convenience
    wrapper around rotation_matrix_fr_vecs(n, z_hat).

    Parameters
    ----------
    n: array-like, shape (3,)
        Source direction vector to rotate to the z-axis.

    Returns
    -------
    R: np.ndarray, shape (3, 3)
        Rotation matrix satisfying R @ (n/||n||) == [0, 0, 1].
    '''
    return rotation_matrix_fr_vecs(n, np.array([0., 0., 1.]))


# Body-diagonal unit vectors, one per octant. The label encodes the
# sign of each Cartesian component in x-y-z order: p = +1, m = -1.
OCTANT_DIRECTIONS = {
    'ppp': np.array([ 1.,  1.,  1.]) / np.sqrt(3.),
    'ppm': np.array([ 1.,  1., -1.]) / np.sqrt(3.),
    'pmp': np.array([ 1., -1.,  1.]) / np.sqrt(3.),
    'pmm': np.array([ 1., -1., -1.]) / np.sqrt(3.),
    'mpp': np.array([-1.,  1.,  1.]) / np.sqrt(3.),
    'mpm': np.array([-1.,  1., -1.]) / np.sqrt(3.),
    'mmp': np.array([-1., -1.,  1.]) / np.sqrt(3.),
    'mmm': np.array([-1., -1., -1.]) / np.sqrt(3.),
}


def rotate_snapdict(snapdict, R, keys=('Coordinates', 'Velocities')):
    '''
    Return a shallow copy of snapdict with the vector fields listed
    in keys rotated by R. Keys absent from the snapdict are silently
    skipped. Scalar fields such as r and Masses are invariant under
    rotation and do not need updating.

    Parameters
    ----------
    snapdict: dict
        Particle data dictionary, e.g. as returned by
        mockobservation_tools.galaxy_tools.load_sim_General.
    R: np.ndarray, shape (3, 3)
        Rotation matrix. Applies via the row-vector convention:
        rotated_field = field @ R.T.
    keys: tuple of str, default ('Coordinates', 'Velocities')
        Fields to rotate. List only (N, 3) array fields here.

    Returns
    -------
    d: dict
        Shallow copy of snapdict with the specified vector fields
        replaced by their rotated counterparts.
    '''
    d = snapdict.copy()
    for key in keys:
        if key in d:
            d[key] = snapdict[key] @ R.T
    return d

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
    R = rotation_matrix_fr_vecs(
        disk_ang,
        np.array((0.0, 0.0, 1.0))
    )

    return R

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

    R = rotation_matrix_fr_dat(coords_centered, v_vecs, masses, rs)

    # get the new coordinates where the stellar disk lies in the XY plane
    print('Rotating')
    coord_rotated = rotate(coords_centered, R)
    v_vecs_rotated = rotate(v_vecs, R)

    return coord_rotated, v_vecs_rotated
