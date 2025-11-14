import os
import cv2
import numpy as np
import pytz
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Log images
def log_input_image(x, opts):
	return tensor2im(x)


def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


# Visualization utils
def get_colors():
	# currently support up to 19 classes (for the celebs-hq-mask dataset)
	colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
			  [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
			  [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
	return colors


def vis_faces(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(12, 4 * display_count))
	gs = fig.add_gridspec(display_count, 4)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		vis_faces_no_id(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig


# def vis_faces_no_id(hooks_dict, fig, gs, i):
# 	plt.imshow(hooks_dict['input_face'], cmap="gray")
# 	plt.title('Input')
# 	fig.add_subplot(gs[i, 1])
# 	plt.imshow(hooks_dict['y_hat'])
# 	plt.title('Output (TriPlaneNet)')
# 	fig.add_subplot(gs[i, 2])
# 	plt.imshow(hooks_dict['y_hat_psp'])
# 	plt.title('Output (pSp)')
def vis_faces_no_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'], cmap="gray")
	plt.title('Input')

	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['y_hat'])
	plt.title('Reconstruction')
	
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['y_hat_novel'])
	plt.title('Novel View')

	fig.add_subplot(gs[i, 3])
	plt.imshow(hooks_dict['warp'])
	plt.title('Warp')

# def get_time(tight=False):
#     est_tz = pytz.timezone('US/Eastern')
#     est_now = est_tz.localize(datetime.datetime.now())
#     cst_tz = pytz.timezone('Asia/Shanghai')
#     cst_now = est_now.astimezone(cst_tz)
#     now_time = cst_now.strftime("[%Y%m%d-%H%M%S]") if tight else cst_now.strftime("[%Y-%m-%d %H:%M:%S]")
#     return now_time
def get_time(tight=False):
	cst_tz = pytz.timezone('Asia/Shanghai')
	cst_now = cst_tz.localize(datetime.datetime.now())
	now_time = cst_now.strftime("[%Y%m%d-%H%M%S]") if tight else cst_now.strftime("[%Y-%m-%d %H:%M:%S]")
	return now_time

def addtime2path(path):
	path_dir = os.path.dirname(path)
	path_base = os.path.basename(path)
	new_path = os.path.join(path_dir, get_time(tight=True)+'_'+path_base)
	return new_path


def add_grid_lines(big_image, hw, line_width=16):

    _, nh, mw = big_image.shape
    h = w = hw
    assert nh % h == 0 and mw % w == 0

    n = nh // h
    m = mw // w

    new_h = n * (h + line_width) - line_width
    new_w = m * (w + line_width) - line_width

    new_image = torch.ones(3, new_h, new_w, dtype=big_image.dtype, device=big_image.device)

    for i in range(n):
        row_start  = i * h
        row_end    = row_start + h
        new_rstart = i * (h + line_width)
        new_rend   = new_rstart + h

        for j in range(m):
            col_start  = j * w
            col_end    = col_start + w
            new_cstart = j * (w + line_width)
            new_cend   = new_cstart + w

            new_image[:, new_rstart:new_rend, new_cstart:new_cend] = \
                big_image[:, row_start:row_end, col_start:col_end]

    return new_image