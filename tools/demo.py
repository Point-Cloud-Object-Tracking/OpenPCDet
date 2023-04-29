import argparse
import os
import glob
import sys
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V

    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V

    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

###################
import matplotlib as mpl

# Since we are running the code on a compute cluster without a graphical display attached
# we do not want to select a graphical backend for matplotlib.
mpl.use("Agg")
import matplotlib.pyplot as plt

classes = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}

# get cmap from matplotlib
cm_viridis = mpl.cm.get_cmap("viridis")
cm_hsv = mpl.cm.get_cmap("hsv")
cm_gp2 = mpl.cm.get_cmap("gnuplot2")
cm_gistrb = mpl.cm.get_cmap("gist_rainbow")
cm_rb = mpl.cm.get_cmap("rainbow")

class_colors = {
    value: color
    for value, color in zip(classes.values(), cm_rb(np.linspace(0, 1, len(classes))))
}
###################


class DemoDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None,
        ext=".bin",
    ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = (
            glob.glob(str(root_path / f"*{self.ext}"))
            if self.root_path.is_dir()
            else [self.root_path]
        )

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == ".bin":
            points = np.fromfile(
                self.sample_file_list[index], dtype=np.float32
            ).reshape(-1, 4)
        elif self.ext == ".npy":
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            "points": points,
            "frame_id": index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="cfgs/kitti_models/second.yaml",
        help="specify the config for demo",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="demo_data",
        help="specify the point cloud data file or directory",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="specify the pretrained model"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".bin",
        help="specify the extension of your point cloud data file",
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("-----------------Quick Demo of OpenPCDet-------------------------")
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        ext=args.ext,
        logger=logger,
    )
    logger.info(f"Total number of samples: \t{len(demo_dataset)}")

    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    all_bboxes = [[] for _ in range(len(demo_dataset))]
    all_types = [[] for _ in range(len(demo_dataset))]
    
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f"Visualized sample index: \t{idx + 1}")
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            # breakpoint()
            # print(pred_dicts.__class__)

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            points = data_dict["points"]
            pred_boxes = pred_dicts[0]["pred_boxes"]
            pred_labels = pred_dicts[0]["pred_labels"]
            pred_scores = pred_dicts[0]["pred_scores"]
            if isinstance(points, torch.Tensor):
                points = points.cpu().numpy()
            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.cpu().numpy()
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.cpu().numpy()
            if isinstance(pred_labels, torch.Tensor):
                pred_labels = pred_labels.cpu().numpy()

            ###################
            # plot birds eye view
            fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(13, 13))
            ax1.scatter3D(
                points[:, 1],
                points[:, 2],
                points[:, 0],
                s=0.2,
                c="black",
                edgecolors="none",
            )

            fig2, ax2 = plt.subplots(figsize=(13, 13))
            ax2.scatter(points[:, 1], points[:, 2], s=0.2, c="black", edgecolors="none")
            
            bboxes = []
            types = []

            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                # print(box, score, label)
                bboxes.append(np.array(np.append(box, score)))
                types.append(np.array(label))
                
                center_x, center_y, center_z, l, w, h, yaw = box

                corners_x = np.array(
                    [[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]]
                ).T
                corners_y = np.array(
                    [[w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]]
                ).T
                corners_z = np.array(
                    [[h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]]
                ).T
                points = np.hstack((corners_x, corners_y, corners_z))

                # rotation matrix
                R = np.array(
                    [
                        [np.cos(-yaw), -np.sin(-yaw), 0],
                        [np.sin(-yaw), np.cos(-yaw), 0],
                        [0, 0, 1],
                    ]
                )
                # rotate the box
                points = points @ R

                # offset the box with center position
                points[:, 0] += center_x
                points[:, 1] += center_y
                points[:, 2] += center_z

                # colors
                line_c = class_colors[label]
                line_a = 1
                line_w = 0.5

                # plot 12 lines of bounding box
                for i in range(4):
                    ax1.plot3D(
                        points[[i, i + 4], 0],
                        points[[i, i + 4], 1],
                        points[[i, i + 4], 2],
                        c=line_c,
                        alpha=line_a,
                        linewidth=line_w,
                    )
                for i in range(4):
                    ax1.plot3D(
                        points[[i, (i + 1) % 4], 0],
                        points[[i, (i + 1) % 4], 1],
                        points[[i, (i + 1) % 4], 2],
                        c=line_c,
                        alpha=line_a,
                        linewidth=line_w,
                    )
                for i in range(4):
                    ax1.plot3D(
                        points[[i + 4, (i + 1) % 4 + 4], 0],
                        points[[i + 4, (i + 1) % 4 + 4], 1],
                        points[[i + 4, (i + 1) % 4 + 4], 2],
                        c=line_c,
                        alpha=line_a,
                        linewidth=line_w,
                    )

                # plot 12 lines of bounding box
                for i in range(4):
                    ax2.plot(
                        points[[i, i + 4], 0],
                        points[[i, i + 4], 1],
                        c=line_c,
                        alpha=line_a,
                        linewidth=line_w,
                    )
                for i in range(4):
                    ax2.plot(
                        points[[i, (i + 1) % 4], 0],
                        points[[i, (i + 1) % 4], 1],
                        c=line_c,
                        alpha=line_a,
                        linewidth=line_w,
                    )
                for i in range(4):
                    ax2.plot(
                        points[[i + 4, (i + 1) % 4 + 4], 0],
                        points[[i + 4, (i + 1) % 4 + 4], 1],
                        c=line_c,
                        alpha=line_a,
                        linewidth=line_w,
                    )


            home = os.path.expanduser("~")
            # plot setup
            ax1.set_aspect("equal", adjustable="box")
            ax2.set_aspect("equal")
            fig1.savefig(
                f"{home}/results/nuscenes/point-pillars/{idx + 1}test3D.png", dpi=300
            )
            fig2.savefig(
                f"{home}/results/nuscenes/point-pillars/{idx + 1}testBEV.png", dpi=300
            )
    
            print("Saving results")
            all_bboxes[idx] = bboxes
            all_types[idx] = types

            ###################

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)
            

    logger.info("Demo done.")
    
    # Save detectionas as 'bboxes' and 'types'
    # bboxes = listof(np.array(x, y, z, l, w, h, yaw, confidence score))
    # types = list(classes) [1: Vehicle, 2: Pedestrian, 4: cyclist]
    np.savez(f"/home/cv08f23/results/nuscenes/point-pillars/test.npz", bboxes=all_bboxes, types=all_types)

if __name__ == "__main__":
    main()
    sys.exit(0)
