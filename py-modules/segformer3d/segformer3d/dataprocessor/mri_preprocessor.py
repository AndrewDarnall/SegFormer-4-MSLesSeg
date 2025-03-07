import os
import sys

import torch
import nibabel
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from monai.data import MetaTensor
from multiprocessing import Process, Pool
from sklearn.preprocessing import MinMaxScaler 
from monai.transforms import (
    Orientation, 
    EnsureType,
    ConvertToMultiChannelBasedOnBratsClasses,
)

from typing import (
    Optional
)

"""

Target Dataset Structure (based on BraTs 2021)

data 
 │
 ├───train
 │      ├──BraTS2021_00000 
 │      │      └──BraTS2021_00000_flair.nii.gz
 │      │      └──BraTS2021_00000_t1.nii.gz
 │      │      └──BraTS2021_00000_t1ce.nii.gz
 │      │      └──BraTS2021_00000_t2.nii.gz
 │      │      └──BraTS2021_00000_seg.nii.gz
 │      ├──BraTS2021_00002
 │      │      └──BraTS2021_00002_flair.nii.gz
 │      ...    └──...
"""


class DatasetPreprocessor:
    def __init__(
        self,
        root_dir: str,
        train_folder_name: str,
        save_dir: str,
    ):
        """
        root_dir: path to the data folder where the raw train folder is
        train_folder_name: name of the folder of the training data
        save_dir: path to directory where each case is going to be saved as a single file containing four modalities
        """
        self.train_folder_dir = os.path.join(root_dir, train_folder_name)
        assert os.path.exists(self.train_folder_dir)
        # walking through the raw training data and list all the folder names, i.e. case name
        self.case_name = next(os.walk(self.train_folder_dir), (None, None, []))[1]
        # MRI mode (not all types of datasets might have this)
        self.MRI_MODE = ["flair", "t1", "t1ce", "t2", "seg"]
        self.save_dir = save_dir


    def get_modality_fp(self, case_name: str, MRI_MODE: str) -> Optional[str]:
        """
        Return the modality file path if the file exists, otherwise return None.
        
        case_name: patient ID
        MRI_MODE: one of ["flair", "t1", "t1ce", "t2", "seg"]
        """
        modality_fp = os.path.join(
            self.train_folder_dir,
            case_name,
            case_name + f"_{MRI_MODE}.nii.gz",
        )
        
        if os.path.exists(modality_fp):
            return modality_fp
        else:
            return None
    


    def load_nifti(self, fp) -> list:
        """
        load a nifti file
        fp: path to the nifti file with (nii or nii.gz) extension
        """
        nifti_data = nibabel.load(fp)
        # get the floating point array
        nifti_scan = nifti_data.get_fdata()
        # get affine matrix
        affine = nifti_data.affine
        return nifti_scan, affine
    


    def normalize(self, x:np.ndarray) -> np.ndarray:
        # Transform features by scaling each feature to a given range.
        scaler = MinMaxScaler(feature_range=(0, 1))
        # (H, W, D) -> (H * W, D)
        normalized_1D_array = scaler.fit_transform(x.reshape(-1, x.shape[-1]))
        normalized_data = normalized_1D_array.reshape(x.shape)
        return normalized_data
    


    def orient(self, x: MetaTensor) -> MetaTensor:
        # Orient the array to be in (Right, Anterior, Superior) scanner coordinate systems
        assert type(x) == MetaTensor
        return Orientation(axcodes="RAS")(x)
    


    def detach_meta(self, x: MetaTensor) -> np.ndarray:
        assert type(x) == MetaTensor
        return EnsureType(data_type="numpy", track_meta=False)(x)
    


    def crop_zero_pixels(self, x: np.ndarray) -> np.ndarray:
        # Get rid of the zero pixels around mri scan and cut it so that the region is useful
        # Crop (1, 240, 240, 155) to (1, 128, 128, 128)
        return x[:, 56:184, 56:184, 13:141]



    def preprocess_modality(self, data_fp: str, is_label: bool = False) -> np.ndarray:
        """
        Apply preprocess stage to the modality
        data_fp: directory to the modality
        """
        data, affine = self.load_nifti(data_fp)
        # Label do not the be normalized 
        if is_label:
            # Binary mask does not need to be float64! For saving storage purposes!
            data = data.astype(np.uint8)
            # categorical -> one-hot-encoded 
            # (240, 240, 155) -> (3, 240, 240, 155)
            data = ConvertToMultiChannelBasedOnBratsClasses()(data)
        else:
            data = self.normalize(x=data)
            # (240, 240, 155) -> (1, 240, 240, 155)
            data = data[np.newaxis, ...]
        
        data = MetaTensor(x=data, affine=affine)
        # For oreinting the coordinate system we need the affine matrix
        data = self.orient(data)
        # Detaching the meta values from the oriented array
        data = self.detach_meta(data)
        # (240, 240, 155) -> (128, 128, 128)
        data = self.crop_zero_pixels(data)
        return data




    def process(self, idx):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # Get the 4D modalities along with the label
        modalities, label, case_name = self.__getitem__(idx)
        # Creating the folder for the current case id
        data_save_path = os.path.join(self.save_dir, case_name)
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        # Saving the preprocessed 4D modalities containing all the modalities to save path
        modalities_fn = data_save_path + f"/{case_name}_modalities.pt"
        torch.save(modalities, modalities_fn)
        # Saving the preprocessed segmentation label to save path
        label_fn = data_save_path + f"/{case_name}_label.pt"
        torch.save(label, label_fn)



    def __call__(self):
        print(" ---- Started Processing the Dataset ---- ")
        with Pool(processes=os.cpu_count()) as multi_p:
            multi_p.map_async(func=self.process, iterable=range(self.__len__()))
            multi_p.close()
            multi_p.join()
        print(" +++++ Finished Processing the Dataset ++++ ")



    def __getitem__(self, idx):

        case_name = self.case_name[idx]
        # e.g: train/BraTS2021_00000/BraTS2021_00000_flair.nii.gz

        modes = {
            "flair_transv" : None,
            "t1_transv" : None,
            "t1ce_transv" : None,
            "t2_transv" : None,
            "label_transv" : None
        }

        label_transv = None
        
        # Preprocess Flair modality
        FLAIR = self.get_modality_fp(case_name, self.MRI_MODE[0])
        if FLAIR is not None:
            flair = self.preprocess_modality(data_fp=FLAIR, is_label=False)
            flair_transv = flair.swapaxes(1, 3) # transverse plane
            modes["flair_transv"] = flair_transv
            
        # Preprocess T1 modality
        T1 = self.get_modality_fp(case_name, self.MRI_MODE[1])
        if T1 is not None:
            t1 = self.preprocess_modality(data_fp=T1, is_label=False)
            t1_transv = t1.swapaxes(1, 3) # transverse plane
            modes["t1_transv"] = t1_transv
        
        # Preprocess T1ce modality
        T1ce = self.get_modality_fp(case_name, self.MRI_MODE[2])
        if T1ce is not None:
            t1ce = self.preprocess_modality(data_fp=T1ce, is_label=False)
            t1ce_transv = t1ce.swapaxes(1, 3) # transverse plane
            modes["t1ce_transv"] = t1ce_transv
        
        # Preprocess T2
        T2 = self.get_modality_fp(case_name, self.MRI_MODE[3])
        if T2 is not None:
            t2 = self.preprocess_modality(data_fp=T2, is_label=False)
            t2_transv = t2.swapaxes(1, 3) # transverse plane
            modes["t2_transv"] = t2_transv
        
        # Preprocess segmentation label
        Label = self.get_modality_fp(case_name, self.MRI_MODE[4])
        if Label is not None:
            label = self.preprocess_modality(data_fp=Label, is_label=True)
            label_transv = label.swapaxes(1, 3) # transverse plane
            modes["label_transv"] = label_transv

        # Stack not-None modalities along the first dimension 
        modalities = np.concatenate(
            [value for value in modes.values() if value is not None],
            axis=0,
        )
        ret_label = label_transv

        if ret_label is None:
            print(" ==== Error in DataPreprocessor - __getitem__() --> The ret_label variable is NONE ")


        return modalities, ret_label, case_name
    

        
    def __len__(self):
        return self.case_name.__len__()
    


# For additional code rarding the visualization and animation, consult the official repo 
# --> https://github.com/OSUPCVLab/SegFormer3D/blob/main/data/brats2021_seg/brats2021_raw_data/brats2021_seg_preprocess.py