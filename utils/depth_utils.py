"""Utils for monoDepth.
"""
import sys
import re
import numpy as np
import cv2
import torch
import matplotlib


def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)


def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def resize_image(img):
    """Resize image and make it fit for network.

    Args:
        img (array): image

    Returns:
        tensor: data ready for network
    """
    height_orig = img.shape[0]
    width_orig = img.shape[1]

    if width_orig > height_orig:
        scale = width_orig / 384
    else:
        scale = height_orig / 384

    height = (np.ceil(height_orig / scale / 32) * 32).astype(int)
    width = (np.ceil(width_orig / scale / 32) * 32).astype(int)

    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    img_resized = (
        torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).contiguous().float()
    )
    img_resized = img_resized.unsqueeze(0)

    return img_resized


def resize_depth(depth, width, height):
    """Resize depth map and bring to CPU (numpy).

    Args:
        depth (tensor): depth
        width (int): image width
        height (int): image height

    Returns:
        array: processed depth
    """
    depth = torch.squeeze(depth[0, :, :, :]).to("cpu")

    depth_resized = cv2.resize(
        depth.numpy(), (width, height), interpolation=cv2.INTER_CUBIC
    )

    return depth_resized

def write_depth(path, depth, grayscale, bits=1):
    """Write depth map to png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
        grayscale (bool): use a grayscale colormap?
    """
    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return


def render_depth(values, colormap_name="magma_r"):
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return colors


# def calibrate_disparity(disparity_est_orig, depth_map):
#     """
#     Calibrate the estimated depth map to depth map from renderer.
#     Input:
#         depth_est_orig: estimated disparity map from e.g. MiDaS, shape [N, H, W]
#         depth_map: depth map from NeRF, shape [N, H, W]
#     """
#     with torch.no_grad():        
#         # disparity_est_orig = 1 / (depth_est_orig + 1e-8)
#         N, H, W = disparity_est_orig.shape
#         assert depth_map.shape == (N, H, W), f"Shape mismatch: depth_map {depth_map.shape} vs (N, H, W) {(N, H, W)}"

#         disparity_nerf = 1 / (depth_map + 1e-8)

#         # Shift and normalize each sample in the batch
#         min_vals = disparity_est_orig.view(N, -1).min(dim=1)[0]  # [N]
#         max_vals = disparity_est_orig.view(N, -1).max(dim=1)[0]  # [N]
#         # disparity_est = (
#         #     (disparity_est_orig - min_vals.view(N, 1, 1)) 
#         #     / (max_vals.view(N, 1, 1) - min_vals.view(N, 1, 1) + 1e-8) 
#         #     * (1 - 0.01) 
#         #     + 0.01
#         # )
#         disparity_est = (
#             (disparity_est_orig - min_vals.view(N, 1, 1)) 
#             / (max_vals.view(N, 1, 1) - min_vals.view(N, 1, 1) + 1e-8)
#         )

#         # Compute medians for each sample
#         # median_nerf = torch.median(disparity_nerf.view(N, -1), dim=1)[0]  # [N]
#         # median_est = torch.median(disparity_est.view(N, -1), dim=1)[0]    # [N]

#         # Compute scale factors for each sample
#         # scale_nerf = torch.abs(disparity_nerf - median_nerf.view(N, 1, 1)).view(N, -1).mean(dim=1)  # [N]
#         # scale_est = torch.abs(disparity_est - median_est.view(N, 1, 1)).view(N, -1).mean(dim=1)     # [N]
#         min_nerf = disparity_nerf.view(N, -1).min(dim=1)[0].view(N, 1, 1)
#         scale_nerf = (disparity_nerf.view(N, -1).max(dim=1)[0] - disparity_nerf.view(N, -1).min(dim=1)[0]).view(N, 1, 1)

#         # Apply per-sample calibration
#         # scale_ratio = scale_nerf / (scale_est + 1e-8)
#         # disparity_aligned = (
#         #     scale_ratio.view(N, 1, 1) * (disparity_est - median_est.view(N, 1, 1)) 
#         #     + median_nerf.view(N, 1, 1)
#         # )

#         disparity_aligned = scale_nerf * disparity_est + min_nerf

#         depth_aligned = 1 / (disparity_aligned + 1e-8)

#     return depth_aligned.squeeze()

    
def calibrate_disparity(depth_est_orig, depth_map):
    """
    Calibrate the estimated depth map to depth map from renderer.
    Input:
        depth_est_orig: estimated disparity map from e.g. MiDaS, shape [N, H, W]
        depth_map: depth map from NeRF, shape [N, H, W]
    """
    with torch.no_grad():        
        # disparity_est_orig = 1 / (depth_est_orig + 1e-8)
        disparity_est_orig = depth_est_orig.clone()
        N, H, W = depth_est_orig.shape
        assert depth_map.shape == (N, H, W), f"Shape mismatch: depth_map {depth_map.shape} vs (N, H, W) {(N, H, W)}"

        disparity_nerf = 1 / (depth_map + 1e-8)

        # Shift and normalize each sample in the batch
        min_vals = disparity_est_orig.view(N, -1).min(dim=1)[0]  # [N]
        max_vals = disparity_est_orig.view(N, -1).max(dim=1)[0]  # [N]
        disparity_est = (
            (disparity_est_orig - min_vals.view(N, 1, 1)) 
            / (max_vals.view(N, 1, 1) - min_vals.view(N, 1, 1) + 1e-8) 
            * (1 - 0.01) 
            + 0.01
        )

        # Compute medians for each sample
        median_nerf = torch.median(disparity_nerf.view(N, -1), dim=1)[0]  # [N]
        median_est = torch.median(disparity_est.view(N, -1), dim=1)[0]    # [N]

        # Compute scale factors for each sample
        scale_nerf = torch.abs(disparity_nerf - median_nerf.view(N, 1, 1)).view(N, -1).mean(dim=1)  # [N]
        scale_est = torch.abs(disparity_est - median_est.view(N, 1, 1)).view(N, -1).mean(dim=1)     # [N]

        # Apply per-sample calibration
        scale_ratio = scale_nerf / (scale_est + 1e-8)
        disparity_aligned = (
            scale_ratio.view(N, 1, 1) * (disparity_est - median_est.view(N, 1, 1)) 
            + median_nerf.view(N, 1, 1)
        )

        depth_aligned = 1 / (disparity_aligned + 1e-8)

    return depth_aligned.squeeze()
