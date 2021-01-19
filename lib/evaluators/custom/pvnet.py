from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_pose_utils, pvnet_data_utils
import os
from lib.utils.linemod import linemod_config
import torch
if cfg.test.icp:
    from lib.utils import icp_utils
from PIL import Image
from lib.utils.img_utils import read_depth
from scipy import spatial
import matplotlib.pyplot as plt

class Evaluator:

    def __init__(self, result_dir):
        self.result_dir = result_dir
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

        data_root = args['data_root']
        model_path = os.path.join(data_root, 'model.ply')
        self.model = pvnet_data_utils.get_ply_model(model_path)
        self.diameter = np.loadtxt(os.path.join(data_root, 'diameter.txt')).item()

        self.proj2d = []
        self.add = []
        self.icp_add = []
        self.cmd5 = []
        self.translational_error = [] # in cm
        self.rotational_error = [] # in degrees
        self.mask_ap = []
        self.data = {}
        self.icp_render = icp_utils.SynRenderer(cfg.cls_type) if cfg.test.icp else None


    def projection_2d(self, pose_pred, pose_targets, K, threshold=5):
        if "obj_verts_2d" not in self.data.keys():
            self.data["obj_verts_2d"] = []

        model_2d_pred = pvnet_pose_utils.project(self.model, K, pose_pred)
        model_2d_pred = np.expand_dims(model_2d_pred, axis=0)  # add batch dimension
        model_2d_targets = pvnet_pose_utils.project(self.model, K, pose_targets)
        model_2d_targets = np.expand_dims(model_2d_targets, axis=0)  # add batch dimension
        proj_diff = np.linalg.norm(model_2d_pred - model_2d_targets, ord=2, axis=-1)
        proj_diff = np.mean(proj_diff, axis=-1)
        self.data["obj_verts_2d"].append(proj_diff)
        self.proj2d.append(proj_diff < threshold)


    def add_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            mean_dist_index = spatial.cKDTree(model_pred)
            mean_dist, _ = mean_dist_index.query(model_targets, k=1)
            mean_dist = np.mean(mean_dist)
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))

        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add.append(mean_dist < diameter)


    def cm_degree_5_metric(self, pose_pred, pose_targets):
        if "obj_translation_3d" not in self.data.keys():
            self.data["obj_translation_3d"] = []
        if "obj_depth_3d" not in self.data.keys():
            self.data["obj_depth_3d"] = []
        if "obj_rotation_3d" not in self.data.keys():
            self.data["obj_rotation_3d"] = []

        # Compute 3d translational error
        translation_distance = np.linalg.norm(pose_pred[:3, 3] - pose_targets[:3, 3])
        # Compute rotational error
        rot_pred_T = pose_pred[:3, :3].T
        rot_target = pose_targets[:3, :3]
        rotation_diff = np.matmul(rot_pred_T, rot_target)
        trace = np.trace(rotation_diff)
        # Check for singularities at 0° or 180°
        is_symmetric = np.all(np.isclose(rot_pred_T, rot_target))
        is_identity = np.all(np.isclose(rot_pred_T, np.eye(3)))
        # Overwrite trace=-1.0 if we're at the singularity at 180°
        if is_symmetric:
            trace = -1.0
        # Overwrite trace=3.0 if we're at the singularity at 0°
        if is_identity:
            trace = 3.0
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))

        if np.isfinite(translation_distance):
            obj_translation_3d = np.expand_dims(translation_distance, axis=0)
            self.data["obj_translation_3d"].append(obj_translation_3d) # in m
            obj_depth_3d = np.expand_dims(np.abs(pose_pred[2, 3] - pose_targets[2, 3]), axis=0)
            self.data["obj_depth_3d"].append(obj_depth_3d) # in m

        if np.isfinite(angular_distance):
            obj_rotation_3d = np.expand_dims(angular_distance, axis=0)
            self.data["obj_rotation_3d"].append(obj_rotation_3d) # in deg
        self.cmd5.append(translation_distance < 0.05 and angular_distance < 5)


    def mask_iou(self, output, batch):
        if 'mask' in batch and batch['mask'] is not None:
            mask_pred = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
            mask_gt = batch['mask'][0].detach().cpu().numpy()
            iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
            self.mask_ap.append(iou > 0.7)


    def icp_refine(self, pose_pred, anno, output, K):
        depth = read_depth(anno['depth_path'])
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        if pose_pred[2, 3] <= 0 or np.sum(mask) < 20:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000
        R_refined, t_refined = icp_utils.icp_refinement(depth, self.icp_render, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(), (depth.shape[1], depth.shape[0]), depth_only=True,            max_mean_dist_factor=5.0)
        R_refined, _ = icp_utils.icp_refinement(depth, self.icp_render, R_refined, t_refined, K.copy(), (depth.shape[1], depth.shape[0]), no_depth=True)
        pose_pred = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))
        return pose_pred


    def obj_verts_3d(self, pose_pred, pose_targets):
        if "obj_verts_3d" not in self.data.keys():
            self.data["obj_verts_3d"] = []
        if "obj_drilltip_trans_3d" not in self.data.keys():
            self.data["obj_drilltip_trans_3d"] = []
        if "obj_drilltip_rot_3d" not in self.data.keys():
            self.data["obj_drilltip_rot_3d"] = []
        if "gt_drill_rot" not in self.data.keys():
            self.data["gt_drill_rot"] = []

        model_3d_pred = pvnet_pose_utils.transform(self.model, pose_pred, convert_to_homogeneous=True)
        model_3d_pred = np.expand_dims(model_3d_pred, axis=0) # add batch dimension
        model_3d_targets = pvnet_pose_utils.transform(self.model, pose_targets, convert_to_homogeneous=True)
        model_3d_targets = np.expand_dims(model_3d_targets, axis=0) # add batch dimension
        vertex_diff = np.linalg.norm(model_3d_pred - model_3d_targets, ord=2, axis=-1)
        vertex_diff = np.mean(vertex_diff, axis=-1)
        self.data["obj_verts_3d"].append(vertex_diff)

        # TODO add fake metrics for evaluating errors w.r.t. drill angle!
        # Compute drill bit position, orientation
        # THIS IS ONLY VALID FOR OUR EXACT DRILL MODEL!
        DRILL_TIP = np.array([0.053554, 0.225361, -0.241646]).reshape((1, 1, 3))
        DRILL_SHANK = np.array([0.057141, 0.220794, -0.121545]).reshape((1, 1, 3))
        pose_pred = np.expand_dims(pose_pred, axis=0)
        pose_targets = np.expand_dims(pose_targets, axis=0)

        #print("pose_pred:\n{}".format(pose_pred))
        #print("pose_targets:\n{}".format(pose_targets))

        pred_drill_tip = _transform(DRILL_TIP, pose_pred)
        #print("pred_drill_tip:\n{}".format(pred_drill_tip))
        pred_drill_shank = _transform(DRILL_SHANK, pose_pred)
        gt_drill_tip = _transform(DRILL_TIP, pose_targets)
        #print("gt_drill_tip:\n{}".format(gt_drill_tip))
        gt_drill_shank = _transform(DRILL_SHANK, pose_targets)
        obj_drilltip_trans_3d = np.linalg.norm(pred_drill_tip - gt_drill_tip, ord=2, axis=-1)
        #print("obj_drilltip_trans_3d:\n{}".format(obj_drilltip_trans_3d))
        obj_drilltip_trans_3d = np.mean(obj_drilltip_trans_3d, axis=-1)
        self.data["obj_drilltip_trans_3d"].append(obj_drilltip_trans_3d)

        pred_drill_vec = pred_drill_tip - pred_drill_shank
        pred_drill_vec = pred_drill_vec / np.expand_dims(np.linalg.norm(pred_drill_vec, axis=2), axis=1)
        gt_drill_vec = gt_drill_tip - gt_drill_shank
        gt_drill_vec = gt_drill_vec / np.expand_dims(np.linalg.norm(gt_drill_vec, axis=2), axis=1)
        dotprod = pred_drill_vec @ gt_drill_vec.transpose((0, 2, 1))
        obj_drilltip_rot_3d = np.rad2deg(np.arccos(dotprod.squeeze(axis=(1, 2))))
        self.data["obj_drilltip_rot_3d"].append(obj_drilltip_rot_3d)
        self.data["gt_drill_rot"].append(np.squeeze(gt_drill_vec, axis=(0,1)))


    def obj_kpt_2d(self, pose_gt, kpt_3d, kpt_2d, K):
        if "obj_keypoints_2d" not in self.data.keys():
            self.data["obj_keypoints_2d"] = []


        gt_kpt_2d = pvnet_pose_utils.project(kpt_3d, K, pose_gt)
        #print("kpt_3d:\n{}".format(kpt_3d))
        #print("K:\n{}".format(K))
        #print("gt_kpt_2d:\n{}".format(gt_kpt_2d))
        #print("kpt_2d:\n{}".format(kpt_2d))
        err_2d = np.linalg.norm(gt_kpt_2d - kpt_2d, ord=2, axis=-1) # (K,1)
        #print("err_2d:\n{}".format(err_2d))
        err_2d = np.mean(err_2d, axis=-1)
        #print("err_2d mean:\n{}".format(err_2d))
        self.data["obj_keypoints_2d"].append(err_2d)


    def evaluate(self, output, batch):
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        var_2d = output['var'][0].detach().cpu().numpy()

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
        #pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        pose_pred = pvnet_pose_utils.uncertainty_pnp(kpt_3d, kpt_2d, var_2d, K)

        if self.icp_render is not None:
            pose_pred_icp = self.icp_refine(pose_pred.copy(), anno, output, K)
            self.add_metric(pose_pred_icp, pose_gt, icp=True)
        self.projection_2d(pose_pred, pose_gt, K)
        self.obj_verts_3d(pose_pred, pose_gt)
        self.obj_kpt_2d(pose_gt, kpt_3d, kpt_2d, K)
        if cfg.cls_type in ['eggbox', 'glue']:
            self.add_metric(pose_pred, pose_gt, syn=True)
        else:
            self.add_metric(pose_pred, pose_gt)
        self.cm_degree_5_metric(pose_pred, pose_gt)
        self.mask_iou(output, batch)


    def summarize(self):
        result = {
            #'add': self.add,
            #'cmd5': self.cmd5,
            #'ap70': self.mask_ap,
        }
        result.update(self.data)

        proj2d = np.mean(self.proj2d)
        add = np.mean(self.add)
        cmd5 = np.mean(self.cmd5)
        ap = np.mean(self.mask_ap)
        print('2d projections metric: {}'.format(proj2d))
        print('ADD metric: {}'.format(add))
        print('5 cm 5 degree metric: {}'.format(cmd5))
        print('mask ap70: {}'.format(ap))
        if self.icp_render is not None:
            print('ADD metric after icp: {}'.format(np.mean(self.icp_add)))

        self.proj2d = []
        self.add = []
        self.cmd5 = []
        self.translational_error = []
        self.rotational_error = []
        self.mask_ap = []
        self.icp_add = []
        self.data = {}
        return result

def _transform(points3d, Rt):
    # points3d: (B,N,3)
    # Rt: (B,3,4)
    hom_points3d = np.concatenate([points3d, np.ones([points3d.shape[0], points3d.shape[1], 1])], axis=2)
    trans_points3d = hom_points3d @ Rt.transpose((0,2,1))
    return trans_points3d