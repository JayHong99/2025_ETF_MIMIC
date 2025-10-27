import os
import random
import copy

import sys
src_path = os.path.abspath('../..')
print(src_path)
sys.path.append(src_path)

from src.utils import processed_data_path, load_pickle, set_seed, create_directory, write_txt


target_tmp_days = 90

def label_split(all_hosp_adm_dict, task):
    label_hosp_adm_dict = {}
    no_label_hosp_adm_dict = {}
    for admission_id, admission in all_hosp_adm_dict.items():
        label = getattr(admission, task)
        if label is not None:
            label_hosp_adm_dict[admission_id] = admission
        else:
            no_label_hosp_adm_dict[admission_id] = admission
    return label_hosp_adm_dict, no_label_hosp_adm_dict


def tvt_split(hosp_adm_dict, ratio=None):
    if ratio is None:
        ratio = [0.7, 0.1, 0.2]
    all_admission_ids = list(hosp_adm_dict.keys())
    random.shuffle(all_admission_ids)
    s1 = ratio[0]
    s2 = ratio[0] + ratio[1]
    train_admission_ids = all_admission_ids[:int(len(all_admission_ids) * s1)]
    val_admission_ids = all_admission_ids[int(len(all_admission_ids) * s1): int(len(all_admission_ids) * s2)]
    test_admission_ids = all_admission_ids[int(len(all_admission_ids) * s2):]
    return train_admission_ids, val_admission_ids, test_admission_ids


def main():
    task = 'readmission'
    target_tmp_days = 15
    print(f"Loading Pickle from {os.path.join(processed_data_path, f'mimic4/hosp_adm_dict_{target_tmp_days}days.pkl')}")
    all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, f"mimic4/hosp_adm_dict_{target_tmp_days}days.pkl"))
    print(f"Loaded : {len(all_hosp_adm_dict)} admissions")
    for seed in range(2026, 2029) : 
        set_seed(seed)
        label_hosp_adm_dict = {}
        for admission_id, admission in all_hosp_adm_dict.items():
            if getattr(admission, task) is not None:
                label_hosp_adm_dict[admission_id] = admission

        train_admission_ids, val_admission_ids, test_admission_ids = tvt_split(label_hosp_adm_dict)
        output_path = os.path.join(processed_data_path, f"mimic4/task:{task}_{target_tmp_days}days")
        create_directory(output_path)
        write_txt(os.path.join(output_path, f"train_admission_ids_seed_{seed}.txt"), train_admission_ids)
        write_txt(os.path.join(output_path, f"val_admission_ids_seed_{seed}.txt"), val_admission_ids)
        write_txt(os.path.join(output_path, f"test_admission_ids_seed_{seed}.txt"), test_admission_ids)
        print(f"task: {task}")
        print(f"train: {len(train_admission_ids)}")
        print(f"val: {len(val_admission_ids)}")
        print(f"test: {len(test_admission_ids)}")
    return


if __name__ == "__main__":
    main()
