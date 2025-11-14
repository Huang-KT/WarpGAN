import os
import shutil
import sys
from glob import glob
from tqdm import tqdm

gpu_id = sys.argv[1]
custom_folder = sys.argv[2]
temp_folder = sys.argv[3]
output_folder = sys.argv[4]

name_list = [x for x in sorted(glob("%s/*.png"%(custom_folder))) if 'mask' not in x]
name_list = name_list + [x for x in sorted(glob("%s/*.jpg"%(custom_folder))) if 'mask' not in x]
print("Number of images to process: ", len(name_list))
os.system('CUDA_VISIBLE_DEVICES=%s python DataProcess/Gen_HeadMask.py --img_dir %s'%(gpu_id,custom_folder))
os.system('CUDA_VISIBLE_DEVICES=%s python DataProcess/Gen_Landmark.py --img_dir %s'%(gpu_id,custom_folder))
for name_all in tqdm(name_list, desc="align_roll"):
    name = os.path.basename(name_all)[:-4]
    os.system('python align_roll.py %s %s %s'%(name, custom_folder,temp_folder))

os.system('CUDA_VISIBLE_DEVICES=%s python process_test_images.py --input_dir %s --gpu=%s'%(gpu_id,temp_folder,gpu_id))

# os.system('python check_pose.py %s %s '%(temp_folder,output_folder))

os.makedirs(output_folder, exist_ok=True)
result_dir = os.path.join(temp_folder, "cropped_images")
for filename in os.listdir(result_dir):
    result_file = os.path.join(result_dir, filename)
    if os.path.isfile(result_file):
        shutil.copy2(result_file, output_folder)

# import shutil
# shutil.rmtree(temp_folder)

##example
#python extract_pose.py 0 custom_imgs_folder temp_folder output_folder

print("Pose extraction done!")