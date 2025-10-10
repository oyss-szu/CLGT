# We appreciate the previous open-source works.
# [Boosting3DoF]([https://github.com/YujiaoShi/Boosting3DoFAccuracy])

import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import ImageOps
from CLGT.torch_geometry import euler_angles_to_matrix, get_perspective_transform
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

def grid_sample(image, optical, jac=None):
    # Interpolation function 

    N, C, IH, IW = image.shape  # Extracting dimensions from the image tensor
    _, H, W, _ = optical.shape  # Extracting dimensions from the optical tensor

    ix = optical[..., 0].view(N, 1, H, W)  
    iy = optical[..., 1].view(N, 1, H, W)  

    with torch.no_grad():
        ix_nw = torch.floor(ix)  
        iy_nw = torch.floor(iy)  
        ix_ne = ix_nw + 1        
        iy_ne = iy_nw            
        ix_sw = ix_nw            
        iy_sw = iy_nw + 1        
        ix_se = ix_nw + 1        
        iy_se = iy_nw + 1       

        # Clamp coordinates to be within valid range
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    # Create masks for valid coordinates
    mask_x = (ix >= 0) & (ix <= IW - 1)
    mask_y = (iy >= 0) & (iy <= IH - 1)
    mask = mask_x * mask_y

    assert torch.sum(mask) > 0  # Ensure that there are valid coordinates

    # Calculate the weights for interpolation
    nw = (ix_se - ix) * (iy_se - iy) * mask
    ne = (ix - ix_sw) * (iy_sw - iy) * mask
    sw = (ix_ne - ix) * (iy - iy_ne) * mask
    se = (ix - ix_nw) * (iy - iy_nw) * mask

    # Flatten the image for easier indexing
    image = image.view(N, C, IH * IW)

    # Gather the values at the four corners
    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)

    # Perform bilinear interpolation
    out_val = (nw_val * nw + ne_val * ne + sw_val * sw + se_val * se)

    if jac is not None:
        # Calculate the gradients with respect to x and y
        dout_dpx = (nw_val * (-(iy_se - iy) * mask) + ne_val * (iy_sw - iy) * mask +
                    sw_val * (-(iy - iy_ne) * mask) + se_val * (iy - iy_nw) * mask)
        dout_dpy = (nw_val * (-(ix_se - ix) * mask) + ne_val * (-(ix - ix_sw) * mask) +
                    sw_val * (ix_ne - ix) * mask + se_val * (ix - ix_nw) * mask)
        dout_dpxy = torch.stack([dout_dpx, dout_dpy], dim=-1)  # [N, C, H, W, 2]

        # Combine with the jacobian if provided
        jac_new = dout_dpxy[None, :, :, :, :, :] * jac[:, :, None, :, :, :]
        jac_new1 = torch.sum(jac_new, dim=-1)

        return out_val, jac_new1  # Return the interpolated values and updated jacobian
    else:
        return out_val, None  # Return only the interpolated values if no jacobian is provided

def get_BEV_projection(img, Ho, Wo, Fov=170, dty=-20, dx=0, dy=0, device = 'cpu'):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = device

    # **检查 img 类型**
    if isinstance(img, torch.Tensor):
        H, W = img.shape[-2:]  # PyTorch Tensor 格式 [C, H, W]
    elif isinstance(img, np.ndarray):
        H, W = img.shape[:2]  # NumPy 格式 [H, W, C]
    elif isinstance(img, Image.Image):
        W, H = img.size  # PIL 格式，注意顺序是 (W, H)
        img = np.array(img)  # 转换为 NumPy
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    Hp, Wp = img.shape[0], img.shape[1]  # Panorama image dimensions
    if dty != 0 or Wp != 2 * Hp:
        ty = (Wp / 2 - Hp) / 2 + dty  # Non-standard panorama image completion
        matrix_K = np.array([[1, 0, 0], [0, 1, ty], [0, 0, 1]])
        img = cv2.warpPerspective(img, matrix_K, (int(Wp), int(Hp + (Wp / 2 - Hp))))

    frame = torch.from_numpy(img.copy()).to(device)
    # H, W = img.shape[0], img.shape[1]  # Panorama image dimensions

    Fov = Fov * torch.pi / 180  # Field of View in radians
    center = torch.tensor([W / 2 + dx, H + dy]).to(device)  # Overhead view center

    anglex = torch.tensor(dx).to(device) * 2 * torch.pi / W
    angley = -torch.tensor(dy).to(device) * torch.pi / H
    anglez = torch.tensor(0).to(device)

    # Euler angles
    euler_angles = (anglex, angley, anglez)
    euler_angles = torch.stack(euler_angles, -1)

    # Calculate the rotation matrix
    R02 = euler_angles_to_matrix(euler_angles, "XYZ")
    R20 = torch.inverse(R02)

    f = Wo / 2 / torch.tan(torch.tensor(Fov / 2))
    out = torch.zeros((Wo, Ho, 2)).to(device)
    f0 = torch.zeros((Wo, Ho, 3)).to(device)
    f0[:, :, 0] = Ho / 2 - (torch.ones((Ho, Wo)).to(device) * (torch.arange(Ho)).to(device)).T
    f0[:, :, 1] = Wo / 2 - torch.ones((Ho, Wo)).to(device) * torch.arange(Wo).to(device)
    f0[:, :, 2] = -torch.ones((Wo, Ho)).to(device) * f
    f1 = R20 @ f0.reshape((-1, 3)).T  # x, y, z (3, N)
    f1_0 = torch.sqrt(torch.sum(f1**2, 0))
    f1_1 = torch.sqrt(torch.sum(f1[:2, :]**2, 0))
    theta = torch.arctan2(f1[2, :], f1_1) + torch.pi / 2  # [-pi/2, pi/2] => [0, pi]
    phi = torch.arctan2(f1[1, :], f1[0, :])  # [-pi, pi]
    phi = phi + torch.pi  # [0, 2pi]

    i_p = 1 - theta / torch.pi  # [0, 1]
    j_p = 1 - phi / (2 * torch.pi)  # [0, 1]
    out[:, :, 0] = j_p.reshape((Ho, Wo))
    out[:, :, 1] = i_p.reshape((Ho, Wo))
    out[:, :, 0] = (out[:, :, 0] - 0.5) / 0.5  # [-1, 1]
    out[:, :, 1] = (out[:, :, 1] - 0.5) / 0.5  # [-1, 1]

    BEV = F.grid_sample(frame.permute(2, 0, 1).unsqueeze(0).float(), out.unsqueeze(0), align_corners=True)
    # BEV = ((BEV.permute(0, 2, 3, 1).squeeze(0)).to(torch.int)).to(torch.uint8)
    # BEV = np.array(BEV.permute(0, 2, 3, 1).squeeze(0).int()).astype(np.uint8)
    BEV = np.array(BEV.permute(0, 2, 3, 1).squeeze(0).int().cpu()).astype(np.uint8)


    return BEV
def extract_edge(image, low_thresh=80, high_thresh=150):
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # 高斯滤波降噪
    blurred = cv2.GaussianBlur(gray, (11, 11), 2.0)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
    enhanced = clahe.apply(blurred)
    # Canny 边缘检测
    # v = np.median(enhanced)
    # low = int(max(0, 0.66 * v))
    # high = int(min(255, 1.33 * v))
    # edges = cv2.Canny(enhanced, low, high)

    blur1 = cv2.GaussianBlur(gray, (3, 3), 1)
    blur2 = cv2.GaussianBlur(gray, (11, 11), 2)
    dog = cv2.subtract(blur1, blur2)
    _, edges = cv2.threshold(dog, 10, 255, cv2.THRESH_BINARY)

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # edges = cv2.Canny(enhanced, threshold1=low_thresh, threshold2=high_thresh)

    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    # min_area = 60  # 你可以试 50~200
    # clean_edges = np.zeros_like(edges)
    # for i in range(1, num_labels):
    #     if stats[i, cv2.CC_STAT_AREA] >= min_area:
    #         clean_edges[labels == i] = 255



    # plt.figure(figsize=(6, 6))
    # plt.imshow(edges, cmap='gray')
    # plt.title('Canny Edge Detection')
    # plt.axis('off')
    # plt.show()
    # input()

    return edges

from PIL import ImageOps

def resize_and_pad_image(image, target_height, target_width):

    # For CVUSA
    # Get the current dimensions of the image
    current_width, current_height = image.size

    # Calculate the padding needed for height
    if current_height == target_height:
        padded_image = image
    else:
        # Calculate the padding needed for the top and bottom
        top_padding = (target_height - current_height) // 2
        bottom_padding = target_height - current_height - top_padding
        # Use Pillow's ImageOps.expand to add padding
        padded_image = ImageOps.expand(image, (0, top_padding, 0, bottom_padding), fill='black')

    # Resize the image to the target dimensions
    resized_image = padded_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return resized_image


def process_images(input_folder, output_folder, H, W, B, resize_and_pad=False):
    os.makedirs(output_folder, exist_ok=True)

    # List all JPG files in the input directory
    files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    # Process images in batches
    for i in tqdm(range(0, len(files), B), desc="Processing images"):
        # Adjust B to the remaining file count if necessary
        current_batch_size = min(B, len(files) - i)
        batch_files = files[i:i + current_batch_size]
        batch_images = []
        batch_output_paths = []

        # Load and transform images
        for file in batch_files:
            file_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)
            image = Image.open(file_path).convert('RGB')
            # Apply resize and pad preprocessing if the flag is set to True
            if resize_and_pad:
                image = resize_and_pad_image(image, H, W)

            # plt.figure(figsize=(6, 6))
            # plt.imshow(image)
            # plt.title('Original Image')
            # plt.axis('off')
            # plt.show()

            # edges = extract_edge(image)

            # plt.figure(figsize=(6, 6))
            # plt.imshow(edges,cmap='gray')
            # plt.title('Original edge Image')
            # plt.axis('off')
            # plt.show()
            # input()
            # edge_only = np.stack([edges] * 3, axis=-1)
            # edge_map = edges[..., None]  # (H, W, 1)
            # img_with_edge = np.concatenate([image, edge_map], axis=-1)  # (H, W, 4)

            # image = transform(image)
            transformed_images = get_BEV_projection(image, H, W, dty=-20) # CVUSA dty=-20
            # transformed_images_edge = get_BEV_projection(edge_only, H, W)

            # bev_edge = transformed_images[..., 3]  # 第4通道是边缘图

            # plt.figure(figsize=(6, 6))
            # plt.imshow(edge_only)
            # plt.title('Original BEV Image')
            # plt.axis('off')
            # plt.show()
            #
            # plt.figure(figsize=(6, 6))
            # plt.imshow(transformed_images_edge)
            # plt.title('Canny Edge Detection')
            # plt.axis('off')
            # plt.show()
            #
            # input()

            batch_images.append(transformed_images)
            batch_output_paths.append(output_path)

        # Stack images into a batch tensor
        # batch_images_tensor = torch.stack(batch_images)

        # Apply grid sampling transformation
        # transformed_images, _ = get_BEV_projection(batch_images_tensor, uv)

        # Save transformed images
        for j, transformed_image in enumerate(batch_images):
            img = Image.fromarray(transformed_image)
            img.save(batch_output_paths[j])




def main():
    # Define input and output directories for processing images
    input_folder = '../dataset/CVUSA-C-ALL'
    output_folder = '../dataset/CVACT/CVUSA-C-ALL_bev'


    # Set parameters for image processing（CVACT）
    B = 32            # Batch size; The default value is usually 1, which needs to be divisible by the number of files.
    S = 512           # Size parameter for the grid (Satellite size)
    H = 384           # Height of the input street image
    W = 384          # Width of the input street image
    Camera_height = -1.5 # Camera height parameter for BEV transformation (Assume the difference between the ground height and the camera height)

    # Create a rotation tensor with all values set to 90 degrees (If North is in the center of Street View)
    rot = torch.tensor([90] * B, dtype=torch.float32)

    # Define the scale of meters per pixel （CVACT）
    meter_per_pixel = 0.06

    # Compute UV coordinates for the satellite to ground transformation
    # uv = BEV_transform(rot, B, S, H, W, meter_per_pixel, Camera_height)

    # Process images using the computed UV coordinates
    process_images(input_folder, output_folder, H, W, B)
    # process_images(input_folder, output_folder, uv, H, W, B)


if __name__ == "__main__":
    main()
