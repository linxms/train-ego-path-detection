import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
import albumentations as A  # <-- Add this line

from .common import to_scaled_tensor
from .postprocessing import regression_to_rails


class PathsDataset(Dataset):
    # def __init__(
    #     self,
    #     imgs_path,
    #     annotations_path,
    #     indices,
    #     config,
    #     method,
    #     img_aug=False,
    #     to_tensor=False,
    # ):
    #     """Initializes the dataset for ego-path detection.
    #
    #     Args:
    #         imgs_path (str): Path to the images directory.
    #         annotations_path (str):  Path to the annotations file.
    #         indices (list): List of indices to use in the dataset.
    #         config (dict): Data generation configuration.
    #         method (str): Method to use for ground truth generation ("classification", "regression" or "segmentation").
    #         img_aug (bool, optional): Whether to use stochastic image adjustment (brightness, contrast, saturation and hue). Defaults to False.
    #         to_tensor (bool, optional): Whether to return a ready to infer tensor (scaled and possibly resized). Defaults to False.
    #     """
    #     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    #     self.imgs_path = imgs_path
    #     with open(annotations_path) as json_file:
    #         self.annotations = json.load(json_file)
    #     # 下面注释的代码已经确认可以读取到json文件
    #     # for image_name, annotation in self.annotations.items():
    #     #     left_rail = annotation.get("left_rail")
    #     #     if left_rail is not None:
    #     #         print(f"图像 {image_name} 的 left_rail 值: {left_rail}")
    #     #     else:
    #     #         print(f"图像 {image_name} 中缺少 left_rail 字段")
    #     self.imgs = [sorted(self.annotations.keys())[i] for i in indices]
    #     self.config = config
    #     self.method = method
    #
    #     self.img_aug = (
    #         transforms.ColorJitter(
    #             self.config["brightness"],
    #             self.config["contrast"],
    #             self.config["saturation"],
    #             self.config["hue"],
    #         )
    #         if img_aug
    #         else None
    #     )
    #
    #     self.to_tensor = (
    #         transforms.Compose(
    #             [
    #                 to_scaled_tensor,
    #                 transforms.Resize(self.config["input_shape"][1:][::-1]),
    #             ]
    #         )
    #         if to_tensor
    #         else None
    #     )
    def __init__(
            self,
            imgs_path,
            annotations_path,
            indices,
            config,
            method,
            img_aug=False,
            to_tensor=False,
    ):
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        self.imgs_path = imgs_path
        with open(annotations_path, encoding="utf-8") as json_file:
            self.annotations = json.load(json_file)

        # 过滤掉无效的标注数据
        self.imgs = []
        for img_name in sorted(self.annotations.keys()):
            annotation = self.annotations[img_name]
            if annotation.get("left_rail") and annotation.get("right_rail"):
                self.imgs.append(img_name)

        self.config = config
        self.method = method

        self.img_aug = (
            transforms.ColorJitter(
                self.config["brightness"],
                self.config["contrast"],
                self.config["saturation"],
                self.config["hue"],
            )
            if img_aug
            else None
        )

        self.to_tensor = (
            transforms.Compose(
                [
                    to_scaled_tensor,
                    transforms.Resize(self.config["input_shape"][1:][::-1]),
                ]
            )
            if to_tensor
            else None
        )


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = Image.open(os.path.join(self.imgs_path, img_name))
        annotation = self.annotations[img_name]
        rails_mask = self.generate_rails_mask(img.size, annotation)
        img, rails_mask = self.random_crop(img, rails_mask)
        img, rails_mask = self.random_flip_lr(img, rails_mask)
        # 20%概率进行雨雾或遮挡增强
        if np.random.rand() < 0.2:
            img_np = np.array(img)
            mask_np = np.array(rails_mask)
            aug_type = np.random.choice(['rain', 'fog', 'dropout'])
            if aug_type == 'rain':
                img_np = A.RandomRain(p=1.0)(image=img_np)['image']
                # 雨雾只增强图片，不增强mask
            elif aug_type == 'fog':
                img_np = A.RandomFog(p=1.0)(image=img_np)['image']
            else:
                # dropout时同步增强mask
                augmented = A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(16, 32),
                    hole_width_range=(16, 32),
                    p=1.0
                )(image=img_np, masks=[mask_np])
                img_np = augmented['image']
                mask_np = augmented['masks'][0]
            img = Image.fromarray(img_np)
            rails_mask = mask_np
        if self.to_tensor:
            img = self.to_tensor(img)
        if self.img_aug:
            img = self.img_aug(img)
        if self.method == "regression":
            path_gt, ylim_gt = self.generate_target_regression(rails_mask)
            if self.to_tensor:
                path_gt = torch.from_numpy(path_gt)
                ylim_gt = torch.tensor(ylim_gt)
            return img, path_gt, ylim_gt
        elif self.method == "classification":
            path_gt = self.generate_target_classification(rails_mask)
            if self.to_tensor:
                path_gt = torch.from_numpy(path_gt)
            return img, path_gt
        elif self.method == "segmentation":
            segmentation = self.generate_target_segmentation(rails_mask)
            if self.to_tensor:
                segmentation = segmentation.resize(
                    self.config["input_shape"][1:][::-1], Image.NEAREST
                )
                segmentation = to_scaled_tensor(segmentation)
            return img, segmentation


    # def generate_rails_mask(self, shape, annotation):
    #     rails_mask = Image.new("L", shape, 0)
    #     draw = ImageDraw.Draw(rails_mask)
    #     rails = [np.array(annotation["left_rail"]), np.array(annotation["right_rail"])]
    #     for rail in rails:
    #         draw.line([tuple(xy) for xy in rail], fill=1, width=1)
    #     rails_mask = np.array(rails_mask)
    #     rails_mask[: max(rails[0][:, 1].min(), rails[1][:, 1].min()), :] = 0
    #     for row_idx in np.where(np.sum(rails_mask, axis=1) > 2)[0]:
    #         rails_mask[row_idx, np.nonzero(rails_mask[row_idx, :])[0][1:-1]] = 0
    #     return rails_mask

    # def random_crop(self, img, rails_mask):
    #     # extract sides rails coordinates (red rectangle in paper fig. 4)
    #     rails_mask_last_line = rails_mask[-1, :]
    #     rails_mask_last_line_idx = np.nonzero(rails_mask_last_line)[0]
    #     most_left_rail = np.nonzero(np.sum(rails_mask, axis=0))[0][0]
    #     most_right_rail = np.nonzero(np.sum(rails_mask, axis=0))[0][-1]
    #     # center the crop around the rails (orange rectangle in paper fig. 4)
    #     base_margin_left = rails_mask_last_line_idx[0] - most_left_rail
    #     base_margin_right = most_right_rail - rails_mask_last_line_idx[-1]
    #     max_base_margin = max(base_margin_left, base_margin_right, 0)
    #     mean_crop_left = rails_mask_last_line_idx[0] - max_base_margin
    #     mean_crop_right = rails_mask_last_line_idx[-1] + max_base_margin
    #     # add sides margins (green rectangle in paper fig. 4)
    #     base_width = mean_crop_right - mean_crop_left + 1
    #     mean_crop_left -= base_width * self.config["crop_margin_sides"]
    #     mean_crop_right += base_width * self.config["crop_margin_sides"]
    #     # random left crop
    #     largest_margin = max(mean_crop_left, most_left_rail - mean_crop_left)
    #     std_dev = largest_margin * self.config["std_dev_factor_sides"]
    #     random_crop_left = round(np.random.normal(mean_crop_left, std_dev))
    #     if random_crop_left > rails_mask_last_line_idx[0]:
    #         random_crop_left = 2 * rails_mask_last_line_idx[0] - random_crop_left
    #     random_crop_left = max(random_crop_left, 0)
    #     # random right crop
    #     largest_margin = max(
    #         mean_crop_right - most_right_rail, img.width - 1 - mean_crop_right
    #     )
    #     std_dev = largest_margin * self.config["std_dev_factor_sides"]
    #     random_crop_right = round(np.random.normal(mean_crop_right, std_dev))
    #     if random_crop_right < rails_mask_last_line_idx[-1]:
    #         random_crop_right = 2 * rails_mask_last_line_idx[-1] - random_crop_right
    #     random_crop_right = min(random_crop_right, img.width - 1)
    #     # extract top rails coordinates (red rectangle in paper fig. 4)
    #     most_top_rail = np.nonzero(np.sum(rails_mask, axis=1))[0][0]
    #     # add top margin (green rectangle in paper fig. 4)
    #     rail_height = img.height - most_top_rail
    #     mean_crop_top = most_top_rail - rail_height * self.config["crop_margin_top"]
    #     # random top crop
    #     largest_margin = max(mean_crop_top, img.height - 1 - mean_crop_top)
    #     std_dev = largest_margin * self.config["std_dev_factor_top"]
    #     random_crop_top = round(np.random.normal(mean_crop_top, std_dev))
    #     random_crop_top = max(random_crop_top, 0)
    #     random_crop_top = min(random_crop_top, img.height - 2)  # at least 2 rows
    #     # crop image and mask
    #     img = img.crop(
    #         (random_crop_left, random_crop_top, random_crop_right + 1, img.height)
    #     )
    #     rails_mask = rails_mask[
    #         random_crop_top:, random_crop_left : random_crop_right + 1
    #     ]
    #     return img, rails_mask

    # def generate_rails_mask(self, shape, annotation):
    #     rails_mask = Image.new("L", shape, 0)
    #     draw = ImageDraw.Draw(rails_mask)
    #     rails = [np.array(annotation["left_rail"]), np.array(annotation["right_rail"])]
    #     for rail in rails:
    #         draw.line([tuple(xy) for xy in rail], fill=1, width=1)
    #     rails_mask = np.array(rails_mask)
    #     rails_mask[: max(rails[0][:, 1].min(), rails[1][:, 1].min()), :] = 0
    #     for row_idx in np.where(np.sum(rails_mask, axis=1) > 2)[0]:
    #         rails_mask[row_idx, np.nonzero(rails_mask[row_idx, :])[0][1:-1]] = 0
    #     return rails_mask
    #
    # def random_crop(self, img, rails_mask):
    #     # 提取最后一行轨道的坐标
    #     rails_mask_last_line = rails_mask[-1, :]
    #     rails_mask_last_line_idx = np.nonzero(rails_mask_last_line)[0]
    #
    #     # 检查 rails_mask_last_line_idx 是否为空
    #     if len(rails_mask_last_line_idx) == 0:
    #         raise ValueError("rails_mask 的最后一行没有有效的轨道数据")
    #
    #     # 其他逻辑保持不变
    #     most_left_rail = np.nonzero(np.sum(rails_mask, axis=0))[0][0]
    #     most_right_rail = np.nonzero(np.sum(rails_mask, axis=0))[0][-1]
    #
    #     base_margin_left = rails_mask_last_line_idx[0] - most_left_rail
    #     base_margin_right = most_right_rail - rails_mask_last_line_idx[-1]
    #     max_base_margin = max(base_margin_left, base_margin_right, 0)
    #     mean_crop_left = rails_mask_last_line_idx[0] - max_base_margin
    #     mean_crop_right = rails_mask_last_line_idx[-1] + max_base_margin
    #
    #     base_width = mean_crop_right - mean_crop_left + 1
    #     mean_crop_left -= base_width * self.config["crop_margin_sides"]
    #     mean_crop_right += base_width * self.config["crop_margin_sides"]
    #
    #     largest_margin = max(mean_crop_left, most_left_rail - mean_crop_left)
    #     std_dev = largest_margin * self.config["std_dev_factor_sides"]
    #     random_crop_left = round(np.random.normal(mean_crop_left, std_dev))
    #     if random_crop_left > rails_mask_last_line_idx[0]:
    #         random_crop_left = 2 * rails_mask_last_line_idx[0] - random_crop_left
    #     random_crop_left = max(random_crop_left, 0)
    #
    #     largest_margin = max(
    #         mean_crop_right - most_right_rail, img.width - 1 - mean_crop_right
    #     )
    #     std_dev = largest_margin * self.config["std_dev_factor_sides"]
    #     random_crop_right = round(np.random.normal(mean_crop_right, std_dev))
    #     if random_crop_right < rails_mask_last_line_idx[-1]:
    #         random_crop_right = 2 * rails_mask_last_line_idx[-1] - random_crop_right
    #     random_crop_right = min(random_crop_right, img.width - 1)
    #
    #     most_top_rail = np.nonzero(np.sum(rails_mask, axis=1))[0][0]
    #     rail_height = img.height - most_top_rail
    #     mean_crop_top = most_top_rail - rail_height * self.config["crop_margin_top"]
    #
    #     largest_margin = max(mean_crop_top, img.height - 1 - mean_crop_top)
    #     std_dev = largest_margin * self.config["std_dev_factor_top"]
    #     random_crop_top = round(np.random.normal(mean_crop_top, std_dev))
    #     random_crop_top = max(random_crop_top, 0)
    #     random_crop_top = min(random_crop_top, img.height - 2)
    #
    #     img = img.crop(
    #         (random_crop_left, random_crop_top, random_crop_right + 1, img.height)
    #     )
    #     rails_mask = rails_mask[
    #                  random_crop_top:, random_crop_left: random_crop_right + 1
    #                  ]
    #     return img, rails_mask
    def generate_rails_mask(self, shape, annotation):
        rails_mask = Image.new("L", shape, 0)
        draw = ImageDraw.Draw(rails_mask)
        rails = [np.array(annotation["left_rail"]), np.array(annotation["right_rail"])]

        # 判断有效标注

        for key in ["left_rail", "right_rail"]:
            rail_points = annotation[key]
            if isinstance(rail_points, list) and len(rail_points) > 0:
                rail_points = np.array(rail_points)
                if rail_points.ndim == 1:  # 如果是一维数组，转换为二维数组
                    rail_points = rail_points.reshape(-1, 2)
                if rail_points.shape[0] < 2:  # 检查标注点数量是否足够
                    raise ValueError(f"Not enough points in {key}: {rail_points}")
                rails.append(rail_points)
            else:
                raise ValueError(f"Invalid annotation format for {key}: {rail_points}")
        for rail in rails:
            draw.line([tuple(xy) for xy in rail], fill=1, width=1)
        rails_mask = np.array(rails_mask)
        rails_mask[: max(rails[0][:, 1].min(), rails[1][:, 1].min()), :] = 0
        for row_idx in np.where(np.sum(rails_mask, axis=1) > 2)[0]:
            rails_mask[row_idx, np.nonzero(rails_mask[row_idx, :])[0][1:-1]] = 0

        # 检查生成的rails_mask数组，查看最后一行是否全0
        if np.sum(rails_mask[-1, :]) == 0:
            # 如果最后一行的像素值全为 0，强制设置左右铁轨的最后一个点的 x 值对应的像素为 1
            rails_mask[-1, rails[0][-1, 0]] = 1  # 左铁轨最后一个点的 x 坐标
            rails_mask[-1, rails[1][-1, 0]] = 1  # 右铁轨最后一个点的 x 坐标
        return rails_mask

    def random_crop(self, img, rails_mask):
        # 打印 rails_mask 的行数
        # num_rows = rails_mask.shape[0]
        # print(f"rails_mask 的行数: {num_rows}")

        # 提取最后一行轨道的坐标
        rails_mask_last_line = rails_mask[-1, :]
        rails_mask_last_line_idx = np.nonzero(rails_mask_last_line)[0]

        # 检查 rails_mask_last_line_idx 是否为空
        if len(rails_mask_last_line_idx) == 0:
            print("rails_mask 的最后一行没有有效的轨道数据")
            print(f"rails_mask 的形状: {rails_mask.shape}")
            print(f"rails_mask 的最后一行: {rails_mask_last_line}")
            raise ValueError("rails_mask 的最后一行没有有效的轨道数据")

        # 其他逻辑保持不变
        most_left_rail = np.nonzero(np.sum(rails_mask, axis=0))[0][0]
        most_right_rail = np.nonzero(np.sum(rails_mask, axis=0))[0][-1]

        base_margin_left = rails_mask_last_line_idx[0] - most_left_rail
        base_margin_right = most_right_rail - rails_mask_last_line_idx[-1]
        max_base_margin = max(base_margin_left, base_margin_right, 0)
        mean_crop_left = rails_mask_last_line_idx[0] - max_base_margin
        mean_crop_right = rails_mask_last_line_idx[-1] + max_base_margin

        base_width = mean_crop_right - mean_crop_left + 1
        mean_crop_left -= base_width * self.config["crop_margin_sides"]
        mean_crop_right += base_width * self.config["crop_margin_sides"]

        largest_margin = max(mean_crop_left, most_left_rail - mean_crop_left)
        std_dev = largest_margin * self.config["std_dev_factor_sides"]
        random_crop_left = round(np.random.normal(mean_crop_left, std_dev))
        if random_crop_left > rails_mask_last_line_idx[0]:
            random_crop_left = 2 * rails_mask_last_line_idx[0] - random_crop_left
        random_crop_left = max(random_crop_left, 0)

        largest_margin = max(
            mean_crop_right - most_right_rail, img.width - 1 - mean_crop_right
        )
        std_dev = largest_margin * self.config["std_dev_factor_sides"]
        random_crop_right = round(np.random.normal(mean_crop_right, std_dev))
        if random_crop_right < rails_mask_last_line_idx[-1]:
            random_crop_right = 2 * rails_mask_last_line_idx[-1] - random_crop_right
        random_crop_right = min(random_crop_right, img.width - 1)

        most_top_rail = np.nonzero(np.sum(rails_mask, axis=1))[0][0]
        rail_height = img.height - most_top_rail
        mean_crop_top = most_top_rail - rail_height * self.config["crop_margin_top"]

        largest_margin = max(mean_crop_top, img.height - 1 - mean_crop_top)
        std_dev = largest_margin * self.config["std_dev_factor_top"]
        random_crop_top = round(np.random.normal(mean_crop_top, std_dev))
        random_crop_top = max(random_crop_top, 0)
        random_crop_top = min(random_crop_top, img.height - 2)

        img = img.crop(
            (random_crop_left, random_crop_top, random_crop_right + 1, img.height)
        )
        rails_mask = rails_mask[
                     random_crop_top:, random_crop_left: random_crop_right + 1
                     ]
        return img, rails_mask
    def resize_mask(self, mask, shape):
        height_factor = (shape[0] - 1) / (mask.shape[0] - 1)
        width_factor = (shape[1] - 1) / (mask.shape[1] - 1)
        resized_mask = np.zeros(shape, dtype=np.uint8)
        for i in range(resized_mask.shape[0]):
            row_mask = mask[round(i / height_factor), :]
            row_idx = np.nonzero(row_mask)[0]
            if len(row_idx) == 2:
                resized_mask[i, np.round(row_idx * width_factor).astype(int)] = 1
        return resized_mask

    def random_flip_lr(self, img, rails_mask):
        if np.random.rand() < 0.5:
            img = ImageOps.mirror(img)
            rails_mask = np.fliplr(rails_mask)
        return img, rails_mask

    def generate_target_regression(self, rails_mask):
        unvalid_rows = np.where(np.sum(rails_mask, axis=1) != 2)[0]
        ylim_target = (
            float(1 - (unvalid_rows[-1] + 1) / rails_mask.shape[0])
            if len(unvalid_rows) > 0
            else 1.0
        )
        rails_mask = self.resize_mask(
            rails_mask, (self.config["anchors"], rails_mask.shape[1])
        )
        traj_target = np.array(
            [np.zeros(self.config["anchors"]), np.ones(self.config["anchors"])],
            dtype=np.float32,
        )
        for i in range(self.config["anchors"]):  # it's possible to vectorize this loop
            row = rails_mask.shape[0] - 1 - i
            rails_points = np.nonzero(rails_mask[row, :])[0]
            if len(rails_points) != 2:
                break
            rails_points_normalized = rails_points / (rails_mask.shape[1] - 1)
            traj_target[:, i] = rails_points_normalized
        return traj_target, ylim_target

    def generate_target_classification(self, rails_mask):
        rails_mask = self.resize_mask(
            rails_mask, (self.config["anchors"], self.config["classes"])
        )
        target = (
            np.ones((2, self.config["anchors"]), dtype=int) * self.config["classes"]
        )
        for i in range(self.config["anchors"]):
            row = rails_mask.shape[0] - 1 - i
            rails_points = np.nonzero(rails_mask[row, :])[0]
            if len(rails_points) != 2:
                break
            target[:, i] = rails_points
        return target

    def generate_target_segmentation(self, rails_mask):
        target = np.zeros_like(rails_mask, dtype=np.uint8)
        row_indices, col_indices = np.nonzero(rails_mask)
        range_rows = np.arange(row_indices.min(), row_indices.max() + 1)
        for row in reversed(range_rows):
            rails_points = col_indices[row_indices == row]
            if len(rails_points) != 2:
                break
            target[row, rails_points[0] : rails_points[1] + 1] = 255
        return Image.fromarray(target)

    def get_perspective_weight_limit(self, percentile, logger):
        logger.info("\nCalculating perspective weight limit...")
        weights = []
        for i in range(len(self)):
            _, traj, ylim = self[i]
            rails = regression_to_rails(traj.numpy(), ylim.item())
            left_rail, right_rail = rails
            rail_width = right_rail[:, 0] - left_rail[:, 0]
            weight = 1 / rail_width
            weights += weight.tolist()
        limit = np.percentile(sorted(weights), percentile)
        logger.info(f"Perspective weight limit: {limit:.2f}")
        return limit
