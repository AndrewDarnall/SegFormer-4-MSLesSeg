"""

    This source file contains two separate data pre-processor classes, each used for a specific version of BraTs
    not ideal but we are on a tight schedule

"""
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

BraTs 2017 Dataset Structure

data 
 │
 ├───train
 │      ├──imageTr
 │      │      └──BRATS_001_0000.nii.gz
 │      │      └──BRATS_001_0001.nii.gz
 │      │      └──BRATS_001_0002.nii.gz
 │      │      └──BRATS_001_0003.nii.gz
 │      │      └──BRATS_002_0000.nii.gz
 │      │      └──...
 │      ├──labelsTr
 │      │      └──BRATS_001.nii.gz
 │      │      └──BRATS_002.nii.gz
 │      │      └──...
 │      ├──imageTs
 │      │      └──BRATS_485_000.nii.gz
 │      │      └──BRATS_485_001.nii.gz
 │      │      └──BRATS_485_002.nii.gz
 │      │      └──BRATS_485_003.nii.gz
 │      │      └──BRATS_486_000.nii.gz
 │      ...    └──...

"""

# ----------------------------------------------- #

"""

BraTs 2021 Dataset Structure

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

class ConvertToMultiChannelBasedOnBrats2017Classes(object):
    """
    Convert labels to multi channels based on brats17 classes:
    "0": "background", 
    "1": "edema",
    "2": "non-enhancing tumor",
    "3": "enhancing tumour"
    Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2),
    and the necrotic and non-enhancing tumor (NCR/NET — label 1)
    """
    def __call__(self, img):
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img == 2) | (img == 3), (img == 2) | (img == 3) | (img == 1), img == 3]
        # merge labels 1 (tumor non-enh) and 3 (tumor enh) and 1 (large edema) to WT
        # label 3 is ET
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)

class BraTs2017PreProcessor:
    def __init__(
        self,
        root_dir: str,
        train_folder_name: str = "train",
        save_dir: str = "../BraTS2017_Training_Data",
    ):
        """
        root_dir: path to the data folder where the raw train folder is
        roi: spatiotemporal size of the 3D volume to be resized
        train_folder_name: name of the folder of the training data
        save_dir: path to directory where each case is going to be saved as a single file containing four modalities
        """

        self.train_folder_dir = os.path.join(root_dir, train_folder_name)
        label_folder_dir = os.path.join(root_dir, train_folder_name, "labelsTr")
        assert os.path.exists(self.train_folder_dir)
        assert os.path.exists(label_folder_dir)
        
        self.save_dir = save_dir
        # we only care about case names for which we have label! 
        self.case_name = next(os.walk(label_folder_dir), (None, None, []))[2]
        
        # MRI type
        self.MRI_CODE = {"Flair": "0000", "T1w": "0001", "T1gd": "0002", "T2w": "0003", "label": None}


    def __len__(self):
        return self.case_name.__len__()

    def normalize(self, x:np.ndarray)->np.ndarray:
        # Transform features by scaling each feature to a given range.
        scaler = MinMaxScaler(feature_range=(0, 1))
        # (H, W, D) -> (H * W, D)
        normalized_1D_array = scaler.fit_transform(x.reshape(-1, x.shape[-1]))
        normalized_data = normalized_1D_array.reshape(x.shape)
        return normalized_data

    def orient(self, x: MetaTensor) -> MetaTensor:
        # orient the array to be in (Right, Anterior, Superior) scanner coordinate systems
        assert type(x) == MetaTensor
        return Orientation(axcodes="RAS")(x)

    def detach_meta(self, x: MetaTensor) -> np.ndarray:
        assert type(x) == MetaTensor
        return EnsureType(data_type="numpy", track_meta=False)(x)

    def crop_brats2021_zero_pixels(self, x: np.ndarray)->np.ndarray:
        # get rid of the zero pixels around mri scan and cut it so that the region is useful
        # crop (240, 240, 155) to (128, 128, 128)
        return x[:, 56:184, 56:184, 13:141]

    def remove_case_name_artifact(self, case_name: str)->str:
        # BRATS_066.nii.gz -> BRATS_066
        return case_name.rsplit(".")[0]

    def get_modality_fp(self, case_name: str, folder: str, mri_code: str = None):
        """
        return the modality file path
        case_name: patient ID
        folder: either [imagesTr, labelsTr]
        mri_code: code of any of the ["Flair", "T1w", "T1gd", "T2w"]
        """
        if mri_code:
            f_name = f"{case_name}_{mri_code}.nii.gz"
        else:
            f_name = f"{case_name}.nii.gz"

        modality_fp = os.path.join(
            self.train_folder_dir,
            folder,
            f_name,
        )
        return modality_fp

    def load_nifti(self, fp):
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

    def _2metaTensor(self, nifti_data: np.ndarray, affine_mat: np.ndarray):
        """
        convert a nifti data to meta tensor
        nifti_data: floating point array of the raw nifti object
        affine_mat: affine matrix to be appended to the meta tensor for later application such as transformation
        """
        # creating a meta tensor in which affine matrix is stored for later uses(i.e. transformation)
        scan = MetaTensor(x=nifti_data, affine=affine_mat)
        # adding a new axis
        D, H, W = scan.shape
        # adding new axis
        scan = scan.view(1, D, H, W)
        return scan

    def preprocess_brats_modality(self, data_fp: str, is_label: bool = False)->np.ndarray:
        """
        apply preprocess stage to the modality
        data_fp: directory to the modality
        """
        data, affine = self.load_nifti(data_fp)
        # label do not the be normalized 
        if is_label:
            # Binary mask does not need to be float64! For saving storage purposes!
            data = data.astype(np.uint8)
            # categorical -> one-hot-encoded 
            # (240, 240, 155) -> (3, 240, 240, 155)
            data = ConvertToMultiChannelBasedOnBrats2017Classes()(data)
        else:
            data = self.normalize(x=data)
            # (240, 240, 155) -> (1, 240, 240, 155)
            data = data[np.newaxis, ...]
        
        data = MetaTensor(x=data, affine=affine)
        # for oreinting the coordinate system we need the affine matrix
        data = self.orient(data)
        # detaching the meta values from the oriented array
        data = self.detach_meta(data)
        # (240, 240, 155) -> (128, 128, 128)
        data = self.crop_brats2021_zero_pixels(data)
        return data

    def __getitem__(self, idx):
        # BRATS_001_0000.nii.gz
        case_name = self.case_name[idx]
        # BRATS_001_0000
        case_name = self.remove_case_name_artifact(case_name)

        
        # preprocess Flair modality
        code = self.MRI_CODE["Flair"]
        flair = self.get_modality_fp(case_name, "imagesTr", code)
        Flair = self.preprocess_brats_modality(flair, is_label=False)
        flair_transv = Flair.swapaxes(1, 3) # transverse plane

        
        # preprocess T1w modality
        code = self.MRI_CODE["T1w"]
        t1w = self.get_modality_fp(case_name, "imagesTr", code)
        t1w = self.preprocess_brats_modality(t1w, is_label=False)
        t1w_transv = t1w.swapaxes(1, 3) # transverse plane
        
        # preprocess T1gd modality
        code = self.MRI_CODE["T1gd"]
        t1gd = self.get_modality_fp(case_name, "imagesTr", code)
        t1gd = self.preprocess_brats_modality(t1gd, is_label=False)
        t1gd_transv = t1gd.swapaxes(1, 3) # transverse plane

        
        # preprocess T2w
        code = self.MRI_CODE["T2w"]
        t2w = self.get_modality_fp(case_name, "imagesTr", code)
        t2w = self.preprocess_brats_modality(t2w, is_label=False)
        t2w_transv = t2w.swapaxes(1, 3) # transverse plane


        # preprocess segmentation label
        code = self.MRI_CODE["label"]
        label = self.get_modality_fp(case_name, "labelsTr", code)
        label = self.preprocess_brats_modality(label, is_label=True)
        label = label.swapaxes(1, 3) # transverse plane 

        # stack modalities (4, D, H, W)
        modalities = np.concatenate(
            (flair_transv, t1w_transv, t1gd_transv, t2w_transv),
            axis=0,
            dtype=np.float32,
        )
    
        return modalities, label, case_name


    def __call__(self):
        print(" ---- Started Processing on BraTs 2017 Dataset ---- ")
        with Pool(processes=os.cpu_count()) as multi_p:
            multi_p.map_async(func=self.process, iterable=range(self.__len__()))
            multi_p.close()
            multi_p.join()
        print(" ---- Finished Processing on BraTs 2017 Dataset ---- ")

    def process(self, idx):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        modalities, label, case_name = self.__getitem__(idx)
        # creating the folder for the current case id
        data_save_path = os.path.join(self.save_dir, case_name)
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        modalities_fn = data_save_path + f"/{case_name}_modalities.pt"
        label_fn = data_save_path + f"/{case_name}_label.pt"
        torch.save(modalities, modalities_fn)
        torch.save(label, label_fn)


 # ------------------------------------------------------------------------------------------ #       



class BraTs2021PreProcessor:
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
        print(" ---- Started Processing the BraTs 2021 Dataset ---- ")
        with Pool(processes=os.cpu_count()) as multi_p:
            multi_p.map_async(func=self.process, iterable=range(self.__len__()))
            multi_p.close()
            multi_p.join()
        print(" +++++ Finished Processing the BraTs 2021 Dataset ++++ ")



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
            print(" ==== Error in BraTs2021PreProcessor - __getitem__() --> The ret_label variable is NONE ")


        return modalities, ret_label, case_name
    

        
    def __len__(self):
        return self.case_name.__len__()
    


# For additional code rarding the visualization and animation, consult the official repo 
# --> https://github.com/OSUPCVLab/SegFormer3D/blob/main/data/brats2021_seg/brats2021_raw_data/brats2021_seg_preprocess.py
