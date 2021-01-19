from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils
from scipy.spatial.transform import Rotation
import trimesh
import os
import torch

mean = pvnet_config.mean
std = pvnet_config.std


class Visualizer:

    def __init__(self):
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)
        self.mesh = trimesh.load(os.path.join(args['data_root'], "model.ply"))

    def visualize(self, output, batch, output_file=None):
        col_nb = 4
        fig = plt.figure(figsize=(12, 3))
        axes = fig.subplots(1, 4, gridspec_kw={"left": 0.0, "right": 1.0, "bottom": 0.0, "top": 1.0})

        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        var_2d = output['var'][0].detach().cpu().numpy()

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
        #pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        pose_pred = pvnet_pose_utils.uncertainty_pnp(kpt_3d, kpt_2d, var_2d, K)

        kpt_2d_gt = pvnet_pose_utils.project(kpt_3d, K, pose_gt)

        verts_2d_gt = pvnet_pose_utils.project(np.array(self.mesh.vertices, copy=True), K, pose_gt)
        verts_2d_pred = pvnet_pose_utils.project(np.array(self.mesh.vertices, copy=True), K, pose_pred)
        rot_verts_3d_gt = np.dot(np.array(self.mesh.vertices, copy=True), pose_gt[:, :3].T) + pose_gt[:, 3:].T
        rot_verts_3d_pred = np.dot(np.array(self.mesh.vertices, copy=True), pose_pred[:, :3].T) + pose_pred[:, 3:].T

        # Column 0
        col_idx = 0
        axes[col_idx].imshow(inp)
        axes[col_idx].axis("off")
        if kpt_2d_gt is not None:
            axes[col_idx].scatter(
                kpt_2d_gt[:, 0], kpt_2d_gt[:, 1], c="b", s=2, marker="X", alpha=0.7
            )
        if kpt_2d is not None:
            axes[col_idx].scatter(
                kpt_2d[:, 0], kpt_2d[:, 1], c="r", s=2, marker="X", alpha=0.7
            )
        if var_2d is not None:
            ells = compute_confidence_ellipses(kpt_2d, var_2d)
            for e in ells:
                axes[col_idx].add_artist(e)
        if kpt_2d_gt is not None and kpt_2d is not None:
            arrow_nb = kpt_2d_gt.shape[0]
            idxs = range(arrow_nb)
            arrows = np.concatenate([kpt_2d_gt[idxs].astype(np.float), kpt_2d[idxs].astype(np.float)], axis=0)
            links = [[i, i + arrow_nb] for i in idxs]
            _visualize_joints_2d(
                axes[col_idx],
                arrows,
                alpha=0.5,
                joint_idxs=False,
                links=links,
                color=["k"] * arrow_nb,
            )

        # Column 1
        col_idx = 1
        axes[col_idx].imshow(inp)
        axes[col_idx].axis("off")
        # Visualize 2D object vertices
        if verts_2d_gt is not None:
            axes[col_idx].scatter(
                verts_2d_gt[:, 0], verts_2d_gt[:, 1], c="b", s=1, alpha=0.05
            )
        if verts_2d_pred is not None:
            axes[col_idx].scatter(
                verts_2d_pred[:, 0], verts_2d_pred[:, 1], c="r", s=1, alpha=0.02
            )

        # Column 2
        # view from the top
        col_idx = 2
        if rot_verts_3d_gt is not None:
            axes[col_idx].scatter(
                rot_verts_3d_gt[:, 2], rot_verts_3d_gt[:, 0], c="b", s=1, alpha=0.02
            )
        if rot_verts_3d_pred is not None:
            axes[col_idx].scatter(
                rot_verts_3d_pred[:, 2], rot_verts_3d_pred[:, 0], c="r", s=1, alpha=0.02
            )
        axes[col_idx].invert_yaxis()

        # Column 3
        # view from the right
        col_idx = 3
        # invert second axis here for more consistent viewpoints
        if rot_verts_3d_gt is not None:
            axes[col_idx].scatter(
                rot_verts_3d_gt[:, 2], -rot_verts_3d_gt[:, 1], c="b", s=1, alpha=0.02
            )
        if rot_verts_3d_pred is not None:
            axes[col_idx].scatter(
                rot_verts_3d_pred[:, 2], -rot_verts_3d_pred[:, 1], c="r", s=1, alpha=0.02
            )

        _squashfig(fig)

        if output_file is None:
            plt.show()
        else:
            fig.savefig(output_file, dpi=300)
            plt.close()


    def visualize_train(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        mask = batch['mask'][0].detach().cpu().numpy()
        vertex = batch['vertex'][0][0].detach().cpu().numpy()
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        fps_2d = np.array(anno['fps_2d'])
        plt.figure(0)
        plt.subplot(221)
        plt.imshow(inp)
        plt.subplot(222)
        plt.imshow(mask)
        plt.plot(fps_2d[:, 0], fps_2d[:, 1])
        plt.subplot(224)
        plt.imshow(vertex)
        plt.savefig('test.jpg')
        plt.close(0)


def compute_confidence_ellipses(mean, cov, n_std=2):
    # Based on https://stackoverflow.com/a/20127387
    # Make sure inputs are batched, add batch dimension otherwise
    if mean.ndim == 1:
        mean = np.expand_dims(mean, axis=0)
    if cov.ndim == 2:
        cov = np.expand_dims(cov, axis=0)
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvals = np.sqrt(eigvals)

    res = []
    for i in range(mean.shape[0]):
        e = Ellipse(xy=tuple(mean[i]),
                    width=eigvals[i, 0] * n_std * 2,
                    height=eigvals[i, 1] * n_std * 2,
                    angle=np.rad2deg(np.arccos(eigvecs[i, 0, 0])),
                    facecolor=(1.0, 0.0, 0.0, 0.2),
                    edgecolor=(1.0, 0.0, 0.0, 0.5))
        res.append(e)
    return res

def _visualize_joints_2d(
        ax,
        joints,
        joint_idxs=True,
        links=None,
        alpha=1,
        scatter=True,
        linewidth=2,
        color=None,
        axis_equal=True,
):
    if links is None:
        links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    if scatter:
        ax.scatter(x, y, 1, "r")

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            ax.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(
        ax, joints, links, alpha=alpha, linewidth=linewidth, color=color
    )
    if axis_equal:
        ax.axis("equal")

def _draw2djoints(ax, annots, links, alpha=1, linewidth=1, color=None):
    colors = ["r", "m", "b", "c", "g", "y", "b"]

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            if color is not None:
                link_color = color[finger_idx]
            else:
                link_color = colors[finger_idx]
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=link_color,
                alpha=alpha,
                linewidth=linewidth,
            )


def _draw2dseg(ax, annot, idx1, idx2, c="r", alpha=1, linewidth=1):
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]],
        [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha,
        linewidth=linewidth,
    )

def _squashfig(fig=None):
    # TomNorway - https://stackoverflow.com/a/53516034
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in fig.axes:
        ax.axis("off")
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
