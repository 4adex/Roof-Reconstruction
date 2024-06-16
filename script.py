### This is example of the script that will be run in the test environment.
### Some parts of the code are compulsory and you should NOT CHANGE THEM.
### They are between '''---compulsory---''' comments.
### You can change the rest of the code to define and test your solution.
### However, you should not change the signature of the provided function.
### The script would save "submission.parquet" file in the current directory.
### The actual logic of the solution is implemented in the `handcrafted_solution.py` file.
### The `handcrafted_solution.py` file is a placeholder for your solution.
### You should implement the logic of your solution in that file.
### You can use any additional files and subdirectories to organize your code.

'''---compulsory---'''
# import subprocess
# from pathlib import Path         
# def install_package_from_local_file(package_name, folder='packages'):
#     """
#     Installs a package from a local .whl file or a directory containing .whl files using pip.

#     Parameters:
#     path_to_file_or_directory (str): The path to the .whl file or the directory containing .whl files.
#     """
#     try:
#         pth = str(Path(folder) / package_name)
#         subprocess.check_call([subprocess.sys.executable, "-m", "pip", "install", 
#                                "--no-index",  # Do not use package index
#                                "--find-links", pth,  # Look for packages in the specified directory or at the file
#                                package_name])  # Specify the package to install
#         print(f"Package installed successfully from {pth}")
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to install package from {pth}. Error: {e}")
        
# install_package_from_local_file('hoho')

import hoho; hoho.setup() # YOU MUST CALL hoho.setup() BEFORE ANYTHING ELSE
# import subprocess
# import importlib
# from pathlib import Path
# import subprocess


# ### The function below is useful for installing additional python wheels.        
# def install_package_from_local_file(package_name, folder='packages'):
#     """
#     Installs a package from a local .whl file or a directory containing .whl files using pip.

#     Parameters:
#     path_to_file_or_directory (str): The path to the .whl file or the directory containing .whl files.
#     """
#     try:
#         pth = str(Path(folder) / package_name)
#         subprocess.check_call([subprocess.sys.executable, "-m", "pip", "install", 
#                                "--no-index",  # Do not use package index
#                                "--find-links", pth,  # Look for packages in the specified directory or at the file
#                                package_name])  # Specify the package to install
#         print(f"Package installed successfully from {pth}")
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to install package from {pth}. Error: {e}")
        

# pip download webdataset -d packages/webdataset --platform manylinux1_x86_64 --python-version 38 --only-binary=:all:
# install_package_from_local_file('webdataset')
# install_package_from_local_file('tqdm')

### Here you can import any library or module you want.
### The code below is used to read and parse the input dataset.
### Please, do not modify it.

import webdataset as wds
from tqdm import tqdm
from typing import Dict
import pandas as pd
from transformers import AutoTokenizer
import os
import time
import io
from PIL import Image as PImage
import numpy as np

from hoho.read_write_colmap import read_cameras_binary, read_images_binary, read_points3D_binary
from hoho import proc, Sample

def convert_entry_to_human_readable(entry):
    out = {}
    already_good = ['__key__', 'wf_vertices', 'wf_edges', 'edge_semantics', 'mesh_vertices', 'mesh_faces', 'face_semantics', 'K', 'R', 't']
    for k, v in entry.items():
        if k in already_good:
            out[k] = v
            continue
        if k == 'points3d':
            out[k] = read_points3D_binary(fid=io.BytesIO(v))
        if k == 'cameras':
            out[k] = read_cameras_binary(fid=io.BytesIO(v))
        if k == 'images':
            out[k] = read_images_binary(fid=io.BytesIO(v))
        if k in ['ade20k', 'gestalt']:
            out[k] =  [PImage.open(io.BytesIO(x)).convert('RGB') for x in v]
        if k == 'depthcm':
            out[k] = [PImage.open(io.BytesIO(x)) for x in entry['depthcm']]
    return out

'''---end of compulsory---'''

### The part below is used to define and test your solution.

from pathlib import Path
def save_submission(submission, path):
    """
    Saves the submission to a specified path.

    Parameters:
    submission (List[Dict[]]): The submission to save.
    path (str): The path to save the submission to.
    """
    sub = pd.DataFrame(submission, columns=["__key__", "wf_vertices", "wf_edges"])
    sub.to_parquet(path)
    print(f"Submission saved to {path}")

if __name__ == "__main__":
    from baseline import predict
    print ("------------ Loading dataset------------ ")
    params = hoho.get_params()
    dataset = hoho.get_dataset(decode=None, split='all', dataset_type='webdataset')

    print('------------ Now you can do your solution ---------------')
    solution = []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = []
        for i, sample in enumerate(tqdm(dataset)):
            results.append(pool.submit(predict, sample, visualize=False))
        
        for i, result in enumerate(tqdm(results)):
            key, pred_vertices, pred_edges = result.result()
            solution.append({
                            '__key__': key,
                            'wf_vertices': pred_vertices.tolist(),
                            'wf_edges': pred_edges
                        })
            if i % 100 == 0:
                # incrementally save the results in case we run out of time
                print(f"Processed {i} samples")
                # save_submission(solution, Path(params['output_path']) / "submission.parquet")
    print('------------ Saving results ---------------')
    save_submission(solution, Path(params['output_path']) / "submission.parquet")
    print("------------ Done ------------ ")
