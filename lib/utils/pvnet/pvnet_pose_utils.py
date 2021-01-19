import numpy as np
import cv2
from scipy.linalg import sqrtm
from scipy.optimize import leastsq

def pnp(points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_EPNP):
    try:
        dist_coeffs = pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method == cv2.SOLVEPNP_EPNP:
        points_3d = np.expand_dims(points_3d, 0)
        points_2d = np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    _, R_exp, t = cv2.solvePnP(points_3d,
                               points_2d,
                               camera_matrix,
                               dist_coeffs,
                               flags=method)
    # , None, None, False, cv2.SOLVEPNP_UPNP)

    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)

    R, _ = cv2.Rodrigues(R_exp)
    # trans_3d=np.matmul(points_3d,R.transpose())+t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    return np.concatenate([R, t], axis=-1)

def uncertainty_pnp(points_3d, points_2d, var, camera_matrix, method=cv2.SOLVEPNP_EPNP):
    # Compute weights
    cov_invs = []
    for vi in range(var.shape[0]):
        if var[vi, 0, 0] < 1e-6 or np.sum(np.isnan(var)[vi]) > 0:
            cov_invs.append(np.zeros([2, 2]).astype(np.float32))
        else:
            cov_inv = np.linalg.inv(sqrtm(var[vi]))
            cov_invs.append(cov_inv)

    cov_invs = np.asarray(cov_invs)  # K,2,2

    # Compute initialization with 4 best points
    weights = cov_invs.reshape([-1, 4])
    weights = weights[:, (0, 1, 3)]
    idxs = np.argsort(weights[:, 0]+weights[:, 1])[-4:]
    _, R_exp, t = cv2.solvePnP(np.expand_dims(points_3d[idxs, :], 0),
                               np.expand_dims(points_2d[idxs, :], 0),
                               camera_matrix, None, None, None, False, flags=cv2.SOLVEPNP_EPNP)
    Rt_vec = np.concatenate([R_exp, t], axis=0)

    # Return if we only have 4 points
    if points_2d.shape[0] == 4:
        R, _ = cv2.Rodrigues(Rt_vec[:3])
        Rt = np.concatenate([R, Rt_vec[3:, None]], axis=-1)
        return Rt

    # Minimize Mahalanobis distance
    Rt_vec, _ = leastsq(mahalanobis, Rt_vec, args=(points_3d, points_2d, cov_invs, camera_matrix))
    R, _ = cv2.Rodrigues(Rt_vec[:3])
    Rt = np.concatenate([R, Rt_vec[3:, None]], axis=-1)
    return Rt

def mahalanobis(Rt_vec, points_3d, points_2d, var, camera_matrix):
    # Rt_vec.shape: (6,)
    # points_3d.shape: (K,3)
    # points_2d.shape: (K,2)
    # var.shape: (K,2,2)
    # camera_matrix.shape: (3,3)
    if np.any(np.iscomplex(var)):
        var = np.real(var)

    R, _ = cv2.Rodrigues(Rt_vec[:3])
    Rt = np.concatenate([R, Rt_vec[3:, None]], axis=-1)

    points_3d_hom = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=-1) # (K,4)
    proj_2d_hom = camera_matrix @ Rt @ points_3d_hom.transpose() # (3, K)
    proj_2d = proj_2d_hom[:2, :] / proj_2d_hom[2:, :] # (2,K)
    err_2d = proj_2d.transpose() - points_2d # (K,2)
    err_2d = np.expand_dims(err_2d, axis=1) # (K,1,2)
    err = err_2d @ var @ err_2d.transpose((0,2,1)) # (K,1,2) x (K,2,2) x (K,2,1) = (K,1,1)
    err = np.sqrt(err.squeeze())
    return err


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def cm_degree_5(pose_pred, pose_targets):
    translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
    rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
    return translation_distance, angular_distance

def transform(verts, trans, convert_to_homogeneous=False):
    """
        verts: [N, 3]
        trans: [3, 4]
        """
    assert len(verts.shape) == 2, "Expected 2 dimensions for verts, got: {}.".format(len(verts.shape))
    assert len(trans.shape) == 2, "Expected 2 dimensions for trans, got: {}.".format(len(trans.shape))
    if convert_to_homogeneous:
        hom_verts = om_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
    else:
        hom_verts = verts

    assert trans.shape[1] == hom_verts.shape[1], \
        "Incompatible shapes: verts.shape: {}, trans.shape: {}".format(verts.shape, trans.shape)

    trans_verts = np.dot(trans, hom_verts.transpose()).transpose()
    return trans_verts