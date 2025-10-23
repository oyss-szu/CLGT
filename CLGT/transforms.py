import functools
import time

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import random
import torch
import numpy as np
from torchvision.transforms import Resize
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import functools
import torch
import torch.nn.functional as F
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform


class BEVtransformer(A.ImageOnlyTransform):
    def __init__(self, Ho=384, Wo=384, angel=170, dty=0, dx=0, dy=0, out=None, always_apply=True, p=1.0): # Ho=518, Wo=518
        super(BEVtransformer, self).__init__(always_apply, p)
        self.Ho = Ho
        self.Wo = Wo
        self.angel = angel
        self.dty = dty
        self.dx = dx
        self.dy = dy
        self.out = out
        print(f"BEVtransformer initialized with p={self.p}")  # 打印调试

    def _axis_angle_rotation(self, axis: str, angle):

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)

        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError(f"Invalid axis {axis}. Should be 'X', 'Y', or 'Z'.")

        return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

    def euler_angles_to_matrix(self, euler_angles, convention: str):
        if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
            raise ValueError("Invalid input euler angles.")
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")

        # Use a lambda to ensure self is used correctly
        matrices = map(lambda args: self._axis_angle_rotation(*args), zip(convention, torch.unbind(euler_angles, -1)))
        return functools.reduce(torch.matmul, matrices)

    def apply(self, image, **params):
        device = 'cpu'
        t0 = time.time()
        Hp, Wp = image.shape[0], image.shape[1]  # Panorama image dimensions

        # 检查 image 是否为 Tensor 并转换为 NumPy 数组
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if self.dty != 0 or Wp != 2 * Hp:
            ty = (Wp / 2 - Hp) / 2 + self.dty  # Non-standard panorama image completion
            matrix_K = np.array([[1, 0, 0], [0, 1, ty], [0, 0, 1]])
            img = cv2.warpPerspective(image, matrix_K, (int(Wp), int(Hp + (Wp / 2 - Hp))))
        else:
            img = image  # 如果没有进行 warpPerspective，则直接使用原始图像

        ######################
        t1 = time.time()

        # 确保 frame 为 Tensor 格式
        frame = torch.from_numpy(img.copy()).to(device)
        t2 = time.time()

        ######################
        if self.out is None:
            Fov = self.angel * torch.pi / 180  # Field of View in radians
            center = torch.tensor([Wp / 2 + self.dx, Hp + self.dy]).to(device)  # Overhead view center

            anglex = torch.tensor(self.dx).to(device) * 2 * torch.pi / Wp
            angley = -torch.tensor(self.dy).to(device) * torch.pi / Hp
            anglez = torch.tensor(0).to(device)

            # Euler angles
            euler_angles = (anglex, angley, anglez)
            euler_angles = torch.stack(euler_angles, -1)

            # Calculate the rotation matrix
            R02 = self.euler_angles_to_matrix(euler_angles, "XYZ")
            R20 = torch.inverse(R02)

            f = self.Wo / 2 / torch.tan(torch.tensor(Fov / 2))
            out = torch.zeros((self.Wo, self.Ho, 2)).to(device)
            f0 = torch.zeros((self.Wo, self.Ho, 3)).to(device)
            f0[:, :, 0] = self.Ho / 2 - (
                        torch.ones((self.Ho, self.Wo)).to(device) * (torch.arange(self.Ho)).to(device)).T
            f0[:, :, 1] = self.Wo / 2 - torch.ones((self.Ho, self.Wo)).to(device) * torch.arange(self.Wo).to(device)
            f0[:, :, 2] = -torch.ones((self.Wo, self.Ho)).to(device) * f
            f1 = R20 @ f0.reshape((-1, 3)).T  # x, y, z (3, N)
            # f1 = f0.reshape((-1, 3)).T
            f1_0 = torch.sqrt(torch.sum(f1 ** 2, 0))
            f1_1 = torch.sqrt(torch.sum(f1[:2, :] ** 2, 0))
            theta = torch.atan2(f1[2, :], f1_1) + torch.pi / 2  # [-pi/2, pi/2] => [0, pi]
            phi = torch.atan2(f1[1, :], f1[0, :])  # [-pi, pi]
            phi = phi + torch.pi  # [0, 2pi]

            i_p = 1 - theta / torch.pi  # [0, 1]
            j_p = 1 - phi / (2 * torch.pi)  # [0, 1]
            out[:, :, 0] = j_p.reshape((self.Ho, self.Wo))
            out[:, :, 1] = i_p.reshape((self.Ho, self.Wo))
            out[:, :, 0] = (out[:, :, 0] - 0.5) / 0.5  # [-1, 1]
            out[:, :, 1] = (out[:, :, 1] - 0.5) / 0.5  # [-1, 1]
        # else:
        #     out = out.to(device)
        t3 = time.time()

        BEV = F.grid_sample(frame.permute(2, 0, 1).unsqueeze(0).float(), out.unsqueeze(0), align_corners=True)
        t4 = time.time()

        return np.array(BEV.permute(0, 2, 3, 1).squeeze(0).int()).astype(np.uint8)

    def get_transform_init_args_names(self):
        # print("get_transform_init_args_names 被调用")
        return ("Ho", "Wo", "angel", "dty", "dx", "dy", "out","p")
        # return ("bev",)

class ContentAwareCFEtransformer(A.ImageOnlyTransform):
    def __init__(self, alpha=0.2, beta=1.0, always_apply=True, p=1.0):
        super(ContentAwareCFEtransformer, self).__init__(always_apply, p)
        self.alpha = alpha
        self.beta = beta

    def FreCom(self, img):
        h, w = img.shape[:2]
        img_dct = np.zeros((h, w, 3))
        for i in range(3):
            img_ = np.float32(img[:, :, i])
            img_dct[:, :, i] = cv2.dct(img_)
        return img_dct

    def get_dynamic_thresholds(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
        grad_strength = np.mean(np.abs(sobel)) / 255.0
        grad_strength = np.clip(grad_strength, 0.05, 0.4)

        f1 = 0.1 + grad_strength * 0.3
        f2 = 0.3 + grad_strength * 0.4
        f3 = 0.6 + grad_strength * 0.2
        return f1, f2, f3

    def apply(self, img, **params):
        theta = np.random.uniform(self.alpha, self.beta)
        h, w = img.shape[:2]
        img_dct = self.FreCom(img)

        cx, cy = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        radius = np.sqrt((X - cy) ** 2 + (Y - cx) ** 2)
        norm_radius = radius / radius.max()

        f1, f2, f3 = self.get_dynamic_thresholds(img)

        mask = np.zeros((h, w, 3))
        for i in range(3):
            mask[..., i][norm_radius <= f1] = 0.1
            mask[..., i][(norm_radius > f1) & (norm_radius <= f2)] = 0.5
            mask[..., i][(norm_radius > f2) & (norm_radius <= f3)] = 0.85
            mask[..., i][norm_radius > f3] = 1.0

        n_mask = 1 - mask

        non_img_dct = img_dct * mask
        cal_img_dct = img_dct * n_mask

        ref_dct = np.zeros_like(non_img_dct)
        for i in range(3):
            ref_dct[:, :, i] = non_img_dct[:, :, i] * (1 + np.random.normal(0, 0.5))

        img_fc = ref_dct + cal_img_dct
        img_out = np.zeros_like(img, dtype=np.float32)
        for i in range(3):
            img_out[:, :, i] = cv2.idct(img_fc[:, :, i]).clip(0, 255)

        return img_out.astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("alpha", "beta", "p")




class Cut(ImageOnlyTransform):
    def __init__(self,
                 cutting=None,
                 always_apply=False,
                 p=1.0):
        super(Cut, self).__init__(always_apply, p)
        self.cutting = cutting

    def apply(self, image, **params):
        if self.cutting:
            image = image[self.cutting:-self.cutting, :, :]

        return image

    def get_transform_init_args_names(self):
        return ("size", "cutting")





def get_transforms_val(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0,
                       ):
    satellite_transforms = A.Compose(
        [A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
         A.Normalize(mean, std),
         ToTensorV2(),
         ])
    ground_transforms_bev = A.Compose([
                                   A.Resize(img_size_ground[0], img_size_ground[1],
                                            interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   # BEVtransformer(),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1],
                                            interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    return satellite_transforms, ground_transforms_bev,ground_transforms


def get_transforms_train_ours(image_size_sat,
                                img_size_ground,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                                ground_cutting=0):
    satellite_transforms = A.Compose([
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=25,
                            max_height=int(0.2 * image_size_sat[0]),
                            max_width=int(0.2 * image_size_sat[0]),
                            min_holes=10,
                            min_height=int(0.1 * image_size_sat[0]),
                            min_width=int(0.1 * image_size_sat[0]),
                            p=1.0),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    ground_transforms_bev= A.Compose([
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        # BEVtransformer(),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=25,
                            max_height=int(0.2 * image_size_sat[0]),
                            max_width=int(0.2 * image_size_sat[0]),
                            min_holes=10,
                            min_height=int(0.1 * image_size_sat[0]),
                            min_width=int(0.1 * image_size_sat[0]),
                            p=1.0),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1],
                                            interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15,
                                                 always_apply=False, p=0.5),
                                   A.OneOf([
                                       A.AdvancedBlur(p=1.0),
                                       A.Sharpen(p=1.0),
                                   ], p=0.3),
                                   A.OneOf([
                                       A.GridDropout(ratio=0.5, p=1.0),
                                       A.CoarseDropout(max_holes=25,
                                                       max_height=int(0.2 * img_size_ground[0]),
                                                       max_width=int(0.2 * img_size_ground[0]),
                                                       min_holes=10,
                                                       min_height=int(0.1 * img_size_ground[0]),
                                                       min_width=int(0.1 * img_size_ground[0]),
                                                       p=1.0),
                                   ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    groundA_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1],
                                            interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   ContentAwareCFEtransformer(),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])


    return satellite_transforms, ground_transforms_bev, ground_transforms,groundA_transforms

#%%
if __name__ == '__main__':
    import numpy as np

    mean=(0.5, 0.5, 0.5)
    std=(0.5, 0.5, 0.5)
    print("What")
    ground_transforms_bev = A.Compose([

        BEVtransformer(p=1.0),

        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    # print(ground_transforms_bev)
    print("ground_transforms_bev：")
    for t in ground_transforms_bev.transforms:
        print(t)

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    transformed = ground_transforms_bev(image=image)["image"]