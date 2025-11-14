import torch
import torch.nn.functional as F


def warping(img, src_camera, target_camera, depth):
    
    N = src_camera.shape[0]
    extrinsic = target_camera[:, :16].view(N, 4, 4)
    intrinsic = target_camera[:, 16:].view(N, 3, 3)
    init_ext = src_camera[:, :16].view(N, 4, 4)

    ray_generator = RaySampler()

    _, _, h, w = img.shape
    depth = F.interpolate(depth, size=(h, w), mode='bilinear')
    

    # make loss only from the foreground area
    depth_mean = torch.mean(depth)
    depth_zeros = torch.zeros_like(depth_mean).cuda()
    depth_ones = torch.ones_like(depth_mean).cuda()
    masked_depths = torch.where(depth<depth_mean, depth_ones, depth_zeros)
    
    ray_origins2, ray_dirs2 = ray_generator(extrinsic, intrinsic.reshape(3,3).unsqueeze(0), depth.shape[-1]) #world space
    
    # Calculate the surface points
    cam_xyz1 = ray_generator.calculate_xyz_of_depth(ray_origins2, ray_dirs2, depth) # world space
    cam_xyz = cam_xyz1[:3].permute(1,0)#grad only goes to extrinsic
    init_trans = init_ext[:, :3, 3]
    
    canonical_cam_origin = init_trans.repeat(cam_xyz.shape[0], 1) # Camera origin
    vectors = cam_xyz - canonical_cam_origin # Ray direction
    plane_norm_vector = -canonical_cam_origin # Norm vector orthogonal to image plane
    plane_point = torch.bmm(init_ext.reshape(-1,4,4), torch.Tensor([[0,0,1,1]]).unsqueeze(-1).cuda()).squeeze(-1).repeat(cam_xyz.shape[0], 1)[:, :3] # Select a point on the image plane

    #Calculate intersections 
    intersections = LinePlaneCollision(plane_norm_vector, plane_point, vectors, canonical_cam_origin) # N, 3
    tmp_ones = torch.ones(intersections.shape[0], 1).cuda()
    intersections1 = torch.cat([intersections, tmp_ones], dim=-1).permute(1,0) # N, 4
    
    # Normalize to uv coordinate
    w2c = torch.linalg.inv(init_ext.reshape(4,4)) 
    pred_uv = torch.mm(w2c, intersections1)[:3].permute(1,0)
    pred_uv = pred_uv/pred_uv[:, 2:]
    pred_uv = torch.mm(intrinsic.reshape(3,3), pred_uv.permute(1,0))[:2].permute(1,0)
    pred_uv = (pred_uv-0.5)*2 #128, 128
    
    res = int(pred_uv.shape[0]**(1/2))

    # Sample feature map by pred_uv
    warpped_image = F.grid_sample(img, pred_uv.reshape(1, res,res,-1), mode='bilinear', align_corners=False)
    masked_depths = F.interpolate(masked_depths, size=(h, w), mode='bilinear')

    warpped_image = warpped_image * masked_depths
    
    return warpped_image, masked_depths


def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    '''
    every input should be in (N, 3) shape
    return Psi is also B, N
    reference : https://gist.github.com/TimSC/8c25ca941d614bf48ebba6b473747d72, 
                https://discuss.pytorch.org/t/dot-product-batch-wise/9746
    '''
    ndotu = torch.bmm(planeNormal.unsqueeze(1), rayDirection.unsqueeze(-1)).squeeze(-1) #B, 1
    if abs(torch.min(ndotu)) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w_vec = rayPoint - planePoint
    si = -torch.bmm(planeNormal.unsqueeze(1), w_vec.unsqueeze(-1)).squeeze(-1) / ndotu
    Psi = w_vec + si * rayDirection + planePoint
    return Psi


class RaySampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = None, None, None, None, None


    def forward(self, cam2world_matrix, intrinsics, resolution, need_cam_space=False):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        intrinsics: (N, 3, 3)
        resolution: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 2)
        """
        N, M = cam2world_matrix.shape[0], resolution**2
        cam_locs_world = cam2world_matrix[:, :3, 3]
        
        cam_locs_cam = torch.zeros_like(cam_locs_world).cuda()
        #4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), indexing='ij')) * (1./resolution) + (0.5/resolution)
        uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
        uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

        x_cam = uv[:, :, 0].view(N, -1)
        y_cam = uv[:, :, 1].view(N, -1)
        z_cam = torch.ones((N, M), device=cam2world_matrix.device)

        x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)

        world_rel_points = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

        ray_dirs = world_rel_points - cam_locs_world[:, None, :]
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)
        
        ray_dirs_cam = cam_rel_points[:, :, :3]
        ray_dirs_cam = torch.nn.functional.normalize(ray_dirs_cam, dim=2)
        

        ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

        if need_cam_space:
            return cam_locs_cam, ray_dirs_cam, uv
        else:
            return ray_origins, ray_dirs

    def calculate_xyz_of_depth(self, ray_origin, ray_dirs, depth):
        '''
        this calculates the actual xyz coordinate from depthmap.
        depth = (1, res, res)
        ray_origin = (1, res*res, 3) -> (3, res, res)
        ray_dirs = (1, res*res, 3) -> (3, res, res)

        xyz = (3, res, res)
        '''
        res = depth.shape[-1]
        if ray_origin.shape[0]==1 and ray_origin.shape[1]==res**2:
            ray_origin = ray_origin.squeeze(0).reshape(res,res,3).permute(2,0,1)
        if ray_dirs.shape[0]==1 and ray_dirs.shape[1]==res**2:
            ray_dirs = ray_dirs.squeeze(0).reshape(res,res,3).permute(2,0,1)
        device = ray_origin.device
        xyz = ray_origin + ray_dirs * depth.squeeze(0)
        ones = torch.ones(1, xyz.shape[1], xyz.shape[2]).to(device)
        xyz1 = torch.cat([xyz, ones], dim=0).reshape(4, res*res)
        return xyz1