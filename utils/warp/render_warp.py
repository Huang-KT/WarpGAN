import numpy as np
import torch

from splatting import splatting_function

# splatting_function:
# https://github.com/CompVis/geometry-free-view-synthesis/blob/master/geofree/modules/warp/midas.py


def render_forward(src_ims, src_dms,
                   sex,  # b, 4, 4
                   K_src_inv,
                   K_dst,
                   alpha=None):
    # R: b,3,3
    # t: b,3
    # K_dst: b,3,3
    # K_src_inv: b,3,3

    R = sex[:,:3,:3]
    t = sex[:,:3,3]

    t = t[...,None]
    src_dms = src_dms.squeeze(1)

    #######

    assert len(src_ims.shape) == 4 # b,c,h,w
    assert len(src_dms.shape) == 3 # b,h,w
    assert src_ims.shape[2:4] == src_dms.shape[1:3], (src_ims.shape,
                                                      src_dms.shape)

    x = np.arange(src_ims.shape[3])
    y = np.arange(src_ims.shape[2])
    coord = np.stack(np.meshgrid(x,y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:,:,[0]]), -1) # z=1
    coord = coord.astype(np.float32)
    coord = torch.as_tensor(coord, dtype=K_dst.dtype, device=K_dst.device)
    coord = coord[None] # b,h,w,3

    D = src_dms[:,:,:,None,None] # b,h,w,1,1

    points = K_dst[:,None,None,...]@(R[:,None,None,...]@(D*K_src_inv[:,None,None,...]@coord[:,:,:,:,None])+t[:,None,None,:,:])
    points = points.squeeze(-1)

    new_z = points[:,:,:,[2]].clone().permute(0,3,1,2) # b,1,h,w
    points = points/torch.clamp(points[:,:,:,[2]], 1e-8, None)

    flow = points - coord
    flow = flow.permute(0,3,1,2)[:,:2,...]

    if alpha is not None:
        # used to be 50 but this is unstable even if we subtract the maximum
        importance = alpha/new_z
        #importance = importance-importance.amin((1,2,3),keepdim=True)
        importance = importance.exp()
    else:
        # use heuristic to rescale import between 0 and 10 to be stable in
        # float32
        importance = 1.0/new_z
        importance_min = importance.amin((1,2,3),keepdim=True)
        importance_max = importance.amax((1,2,3),keepdim=True)
        importance=(importance-importance_min)/(importance_max-importance_min+1e-6)*10-10
        importance = importance.exp()

    input_data = torch.cat([importance*src_ims, importance], 1)
    output_data = splatting_function("summation", input_data, flow)

    num = output_data[:,:-1,:,:]
    nom = output_data[:,-1:,:,:]

    #rendered = num/(nom+1e-7)
    rendered = num/nom.clamp(min=1e-8)
    return rendered