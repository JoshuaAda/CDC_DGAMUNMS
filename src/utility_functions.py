# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:54:44 2022

@author: mlukluek
"""
# %% imports
from pathlib import Path
from datetime import date
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
# import itertools
import numpy as np
import json
import tempfile

# %% functions - general

def get_project_paths(file = __file__, level=0):
    # git repo
    #   - src
    #   - data
    #   - runtime
    working_directory_path = Path().parent.resolve()
    if level == 1 :
        source_path = Path(file).parent.parent.resolve()
        assert source_path.stem == "src"
    elif level == 0:        
        source_path = Path(file).parent.resolve()
        assert source_path.stem == "src"
    else:
        print("not implemented; check folder level")
    data_path = source_path.parent.joinpath('data')
    runtime_path = source_path.parent.joinpath('runtime')
    return working_directory_path, source_path, data_path, runtime_path

def get_time_hms(end,start):
    t = end-start
    h = t//3600
    m = (t%3600)//60
    s = t%60
    print(str(int(h))+"h:"+str(int(m))+"m:"+str(int(s))+"s")

def save_data(data,file_name,folder_path=get_project_paths()[2]):
    """Saves data as pickle file to specified folder under specified name

    Args:
        data: The data to be saved.
        file_name: Name of file
        folder_path: Path to folder (default: runtime/data)

    Returns:
        None
    """

    file_path = folder_path.joinpath(file_name+".pkl")
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)
    print("Data saved to {}".format(file_path))

def load_data(file_name, file_path=get_project_paths()[2]):
    """Loads data from specified folder under specified name.

    Args:
        file_name: Name of file
        folder_path: Path to folder (default: runtime/data)

    Returns:
        data: The data loaded from the file.
    """

    file_path = file_path.joinpath(file_name+".pkl")
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    print("Data loaded from {}".format(file_path))
    return data

# %% extionsions after 17.04.2023
def get_date_today() -> str:
    today = date.today()
    today_str = today.strftime("%Y_%m_%d")
    return today_str

# # get current git commit of file
# import git
# def get_git_commit(file=__file__):
#     repo = git.Repo(search_parent_directories=True)
#     sha = repo.head.object.hexsha
#     return sha

def get_versions(packages):
    versions = {}

    for package in packages:
        try:
            versions[package.__name__] = package.__version__
        except:
            versions[str(package)] = None    # print versions
    print(json.dumps(versions, indent=4))

def set_seeds(packages,seed=0):
    for package in packages:
        if package.__name__ == "torch":
            package.manual_seed(seed)
        elif package.__name__ == "numpy":
            package.random.seed(seed)
        else:
            print("Package not supported")
    # torch.manual_seed(seed)
    # np.random.seed(seed)

def create_daily_path(parent_path,given_date: str =None):
    if given_date is None:
        today_str = get_date_today()
        daily_path = parent_path.joinpath(today_str)
    else:
        daily_path = parent_path.joinpath(given_date)
    # create folder in save_path
    daily_path.mkdir(parents=True, exist_ok=True)
    return daily_path

# # %% extionsions after 21.08.2023

# # function to move object to temporary file on disk
# def move_to_temp(obj):
#     temp = tempfile.NamedTemporaryFile(delete=False)
#     pkl.dump(obj, temp)
#     temp.close()
#     return temp.name