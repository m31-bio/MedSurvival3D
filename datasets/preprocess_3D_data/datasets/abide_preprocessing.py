import glob
import pandas as pd
from template_brain_preprocessing import *


if __name__ == '__main__':
    out_dir = "/home/c306h/cluster-data_all/c306h/ssl3d_data/classification/preprocessed/abide_1mm_cropped_160_ne1w"
    raw_data_dir = '/home/c306h/cluster-data_all/c306h/ssl3d_data/classification/raw/abide_1mm_cropped_160_new'
    base_dir = '/home/c306h/cluster-data_all/t006d/Datasets/ABIDE/ABIDE_img'
    csv_path = '/home/c306h/cluster-data_all/t006d/Datasets/ABIDE/ABIDE_2_20_2025.csv'

    # 1. Find all .nii files under MP-RAGE
    nii_files = glob.glob(os.path.join(base_dir, '**/MP-RAGE/**/*.nii'), recursive=True)


    # 3. Load label CSV
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # clean up header whitespace
    df['Label'] = df['Group'].apply(lambda x: 1 if str(x).strip() == 'Autism' else 0)

    # 4. Map Subject ID to Label
    label_dict_full = dict(zip(df['Subject'].astype(str), df['Label']))

    maybe_mkdir_p(join(raw_data_dir, 'imagesTr'))
    # 5. Build label dict with full path as key
    label_dict = {}

    ###copy data in expected nnU-Net format
    for full_file_name in nii_files:
        subject_id = full_file_name.split('/')[-1].split('_')[1]
        label_dict[subject_id] = label_dict_full.get(subject_id, None)
        img = sitk.ReadImage(join(full_file_name))
        sitk.WriteImage(img, join(raw_data_dir, 'imagesTr', subject_id + '_0000.nii.gz'))

    maybe_mkdir_p(out_dir)
    save_json(label_dict, join(out_dir, 'labels.json'))

    # predict brainmasks
    hd_bet_predict(raw_data_dir)
    load_crop_brainextract_normalize_images(raw_data_dir, out_dir, [1.,1.,1.], [160,160,160], brain_extract=True, num_workers=4)

