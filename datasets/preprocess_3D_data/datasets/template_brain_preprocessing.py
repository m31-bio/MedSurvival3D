import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from datasets.preprocess_3D_data.hd_bet_prediction import hdbet_predict, get_hdbet_predictor
from datasets.preprocess_3D_data.crop_to_mask import load_image_np, crop_center_with_padding_np, get_mask_center
from datasets.preprocess_3D_data.default_resampling import resample_data_or_seg_to_spacing, resample_data_or_seg_to_shape
import numpy as np
from datasets.preprocess_3D_data.cross_validation import generate_crossval_split
from multiprocessing import Pool
import SimpleITK as sitk
from datasets.preprocess_3D_data.blosc_helper import save_case, comp_blosc2_params
from batchgenerators.utilities.file_and_folder_operations import *
from datasets.preprocess_3D_data.normalization import ZScoreNormalization
import re
from typing import Tuple
from typing import List
from collections import defaultdict


def hd_bet_predict(in_folder: str):
    """
    Predicts brain masks for all images in a specified input folder using HD-BET.

    This function applies the HD-BET model to all images in the 'imagesTr' subdirectory of `in_folder`
    that end with `_0000.nii.gz`. The resulting brain masks are saved to the 'masks' subdirectory.

    Parameters:
        in_folder (str): Path to the base folder containing the 'imagesTr' directory.
                         The 'masks' directory will be created or overwritten in the same location.

    Notes:
        - Only brain masks are predicted (not the skull-stripped images).
        - Existing content in the 'masks' folder may be overwritten.
    """
    predictor = get_hdbet_predictor(in_folder)
    hdbet_predict(
        join(in_folder, 'imagesTr'),
        join(in_folder, 'masks'),
        predictor,
        keep_brain_mask=True,
        compute_brain_extracted_image=False
    )


def run_case(
        base_path: str,
        img_id: str,
        target_spacing: Tuple[float, float, float],
        crop_size: Tuple[int, int, int],
        output_dir: str,
        num_modalities: int,
        brain_extract: bool = True
) -> None:
    """
    Preprocesses a multi-modal medical image case by resampling to target spacing,
    optionally applying brain extraction, cropping around the mask center, normalizing,
    and saving the result as compressed blocks.

    Parameters:
        base_path (str): Base directory containing 'imagesTr' and 'masks' subfolders.
        img_id (str): Identifier of the image case (e.g., 'case123').
        target_spacing (Tuple[float, float, float]): Desired spacing for resampling.
        crop_size (Tuple[int, int, int]): Desired output patch size.
        output_dir (str): Path to directory where output files will be saved.
        num_modalities (int): Number of input image modalities (e.g., 4 for BraTS-style input).
        brain_extract (bool, optional): Whether to use brain mask to extract brain region. Default is True.

    Notes:
        - If a brain mask (`masks/{img_id}_0000_bet.nii.gz`) is found, it is used for brain extraction and cropping.
        - Applies Z-score normalization per modality.
        - Output is saved using Blosc2 compression via `save_case(...)`.
    """
    base_img_path = join(base_path, 'imagesTr', img_id)
    mask_path = join(base_path, 'masks', img_id + '_0000_bet.nii.gz')
    has_mask = isfile(mask_path)

    if has_mask:
        mask_img = sitk.ReadImage(mask_path)
        mask_data = sitk.GetArrayFromImage(mask_img)[np.newaxis, ...]
        original_spacing = mask_img.GetSpacing()[::-1]  # Convert from (x, y, z) to (z, y, x)
        mask_1mm = resample_data_or_seg_to_spacing(mask_data, original_spacing, target_spacing, is_seg=True)
        center = get_mask_center(mask_1mm[0])
        resized_mask = crop_center_with_padding_np(mask_1mm[0], center, crop_size)

    for mod_id in range(num_modalities):
        img = sitk.ReadImage(join(base_img_path + f'_000{mod_id}.nii.gz'))
        data = sitk.GetArrayFromImage(img)[np.newaxis, ...]
        data_1mm = resample_data_or_seg_to_spacing(data, original_spacing, target_spacing)

        if has_mask:
            if brain_extract:
                data_1mm *= mask_1mm
            resized_img = crop_center_with_padding_np(data_1mm[0], center, crop_size)[np.newaxis, ...]
        else:
            new_spacing = np.array([i / j * k for i, j, k in zip(target_spacing, crop_size, data_1mm.shape[1:])])
            resized_img = resample_data_or_seg_to_shape(data_1mm, crop_size, target_spacing, new_spacing)

        normalizer = ZScoreNormalization()
        if has_mask:
            normalizer.use_mask_for_norm = True
            data = normalizer.run(resized_img, resized_mask[np.newaxis, ...])
        else:
            data = normalizer.run(resized_img)

        block_size_data, chunk_size_data = comp_blosc2_params(resized_img.shape, crop_size, data.itemsize)
        out_path_truncated = join(output_dir, f'{img_id}_000{mod_id}')
        save_case(data, out_path_truncated, chunks=chunk_size_data, blocks=block_size_data)


def load_crop_brainextract_normalize_images(
        in_folder: str,
        out_folder: str,
        target_spacing: List[float],
        patch_size: List[int],
        brain_extract: bool = True,
        num_workers: int = 1
) -> None:
    """
    Preprocesses all multi-modal brain images in a dataset by performing:
      - consistency checks for missing modalities,
      - resampling to target spacing,
      - optional skull stripping (brain extraction),
      - cropping or resizing to a fixed patch size,
      - z-score normalization (possibly using brain mask),
      - and saving the results as compressed blocks.

    A cross-validation split file is also generated.

    Parameters:
        in_folder (str): Path to input directory containing 'imagesTr' and optionally 'masks'.
        out_folder (str): Directory where the processed images and split file will be saved.
        target_spacing (List[float]): The desired spacing in mm.
        patch_size (List[int]): The target output patch size.
        brain_extract (bool, optional): Whether to apply skull stripping if a brain mask exists. Default is True.
        num_workers (int, optional): Number of parallel workers to use. Default is 1.

    Raises:
        RuntimeError: If some image identifiers are missing one or more expected modalities.
    """
    os.makedirs(out_folder, exist_ok=True)
    all_images = os.listdir(join(in_folder, 'imagesTr'))

    # Get unique image identifiers
    unique_ids = list(set(re.sub(r'_\d{4}\.nii\.gz$', '', f) for f in all_images))

    # Collect modality info per identifier
    pattern = re.compile(r'(.+?)_(\d{4})\.nii\.gz')
    modalities = defaultdict(set)
    for f in all_images:
        m = pattern.match(f)
        if m:
            identifier, mod = m.groups()
            modalities[identifier].add(mod)

    # Find missing modalities
    expected = set.union(*modalities.values())
    incomplete = {k: sorted(expected - v) for k, v in modalities.items() if v != expected}
    if incomplete:
        msg = '\n'.join(f"{k} missing: {v}" for k, v in incomplete.items())
        raise RuntimeError(f"Some identifiers are missing modalities:\n{msg}")

    # Prepare args for parallel case processing
    args_list = [
        (in_folder, id, target_spacing, patch_size, out_folder, len(expected), brain_extract)
        for id in unique_ids
    ]

    with Pool(processes=num_workers) as pool:
        pool.starmap(run_case, args_list)

    # Save splits for cross-validation
    split_file = generate_crossval_split(unique_ids, n_splits=3)
    with open(os.path.join(out_folder, 'splits.json'), "w") as f:
        json.dump(split_file, f, indent=2)

def create_label_dict(path):
    label_dict = load_json(path)
    return label_dict

if __name__ == '__main__':
    '''
    1. organize the raw data like nnU-Net: raw folder has a lable dict (# {'unique_id1': 1, ...} and imagesTr with all images files
    case_identifier_0000.nii.gz and _000x.nii.gz for other modallities
    2. potentially use HD-Bet for brain extraction - saved in masks folder (only applied to modallity 0)
    3. resample mask and image to 1mm spacing
    4. crop image to center of mask with a fixed FOV patch size
    5. z-score normalization of resulting crop
    '''
    print('this is just an example - check abide_preprocessing.py')
