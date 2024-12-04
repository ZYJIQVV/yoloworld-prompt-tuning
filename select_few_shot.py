import os
import shutil
from tqdm import tqdm
from collections import Counter
import numpy as np


shot = 100
dataset = 'object365'


meta_dict = {
    'coco':{
        'src_dataset_rt': '/data/zyj-ll/data/coco/yolo',
        'dst_dataset_rt': f'/data/zyj-ll/data/coco/yolo/few_shot_{shot}',
        'nc': 80,
        'src_img_rt_suffix': 'images/train',
        'src_lbl_rt_suffix': 'labels/train',
    },
    'object365':{
        'src_dataset_rt': '/data/zyj-ll/data/Object365/yolo',
        'dst_dataset_rt': f'/data/zyj-ll/data/Object365/yolo/few_shot_{shot}',
        'nc': 365,
        'src_img_rt_suffix': 'train/images',
        'src_lbl_rt_suffix': 'train/labels',
    },
    'my':{
        'src_dataset_rt': '/data/zyj-ll/data/H_All_Add_F_ZC_Data_copypaste_split_Other_datasets',
        'dst_dataset_rt': rf'/data/zyj-ll/data/H_All_Add_F_ZC_Data_copypaste_split_Other_datasets/few_shot_{shot}',
        'nc': 2,
        'src_img_rt_suffix': 'images/train',
        'src_lbl_rt_suffix': 'labels/train',
    },
}

src_dataset_rt = meta_dict[dataset]['src_dataset_rt']
src_img_rt_suffix = meta_dict[dataset]['src_img_rt_suffix']
src_lbl_rt_suffix = meta_dict[dataset]['src_lbl_rt_suffix']
dst_dataset_rt = meta_dict[dataset]['dst_dataset_rt']
nc = meta_dict[dataset]['nc']


def initialize(src_dataset_rt, dst_dataset_rt):
    # check if dst_dataset_rt exists, if yes, create a new one with a suffix without deleting the old one, if not, create it
    i = 1
    while os.path.exists(dst_dataset_rt + f'_{i}'):
        i += 1
    dst_dataset_rt = dst_dataset_rt + f'_{i}'

    src_img_rt = os.path.join(src_dataset_rt, src_img_rt_suffix)
    src_lbl_rt = os.path.join(src_dataset_rt, src_lbl_rt_suffix)
    dst_train_img_rt = os.path.join(dst_dataset_rt, 'train/images')
    dst_train_lbl_rt = os.path.join(dst_dataset_rt, 'train/labels')
    dst_test_img_rt = os.path.join(dst_dataset_rt, 'val/images')
    dst_test_lbl_rt = os.path.join(dst_dataset_rt, 'val/labels')

    os.makedirs(dst_train_img_rt, exist_ok=True)
    os.makedirs(dst_train_lbl_rt, exist_ok=True)
    os.makedirs(dst_test_img_rt, exist_ok=True)
    os.makedirs(dst_test_lbl_rt, exist_ok=True)

    # load file2category and file2category_counter if exists, otherwise, generate them
    file2category_filename = f'{dataset}_file2category.npy'
    file2category_counter_filename = f'{dataset}_file2category_counter.npy'
    category2file_filename = f'{dataset}_category2file.npy'
    category2file_counter_filename = f'{dataset}_category2file_counter.npy'
    if os.path.exists(file2category_filename):
        file2category = np.load(file2category_filename, allow_pickle=True).item()
    else:
        file2category = get_file2category(src_lbl_rt)
        np.save(file2category_filename, file2category)
    if os.path.exists(file2category_counter_filename):
        file2category_counter = np.load(file2category_counter_filename, allow_pickle=True).item()
    else:
        file2category_counter = {k: Counter(v) for k, v in file2category.items()}
        np.save(file2category_counter_filename, file2category_counter)
    if os.path.exists(category2file_filename):
        category2file = np.load(category2file_filename, allow_pickle=True).item()
    else:
        category2file = get_category2file(src_lbl_rt)
        np.save(category2file_filename, category2file)
    if os.path.exists(category2file_counter_filename):
        category2file_counter = np.load(category2file_counter_filename, allow_pickle=True).item()
    else:
        category2file_counter = {k: Counter(v) for k, v in category2file.items()}
        np.save(category2file_counter_filename, category2file_counter)

    return src_img_rt, src_lbl_rt, dst_train_img_rt, dst_train_lbl_rt, dst_test_img_rt, dst_test_lbl_rt, file2category, file2category_counter, category2file, category2file_counter

def get_category2file(src_lbl_rt):
    category2file = {}
    for lbl in tqdm(os.listdir(src_lbl_rt)):
        with open(os.path.join(src_lbl_rt, lbl), 'r') as f:
            lines = f.readlines()
        k = os.path.splitext(lbl)[0]
        for line in lines:
            category = int(line.split()[0])
            if category not in category2file:
                category2file[category] = []
            category2file[category].append(k)
    return category2file

def get_file2category(src_lbl_rt):
    file2category = {}
    for lbl in tqdm(os.listdir(src_lbl_rt)):
        with open(os.path.join(src_lbl_rt, lbl), 'r') as f:
            lines = f.readlines()
        k = os.path.splitext(lbl)[0]
        file2category[k] = []
        for line in lines:
            category = int(line.split()[0])
            file2category[k].append(category)
    return file2category

def class_driven_select_few_shot(category2file_counter, file2category_counter, n=16, nc=2, error=10):
    few_shot = []
    few_shot_cat_counter = {i:0 for i in range(nc)}
    # sort category2file_counter by the number of instances in each category
    category2file_counter = {k: v for k, v in sorted(category2file_counter.items(), key=lambda item: sum(item[1].values()), reverse=False)}
    
    for catid, cat_counter in tqdm(category2file_counter.items()):
        if len(cat_counter) == 0:
            continue
        files_containing_this_category = list(cat_counter.keys())
        for file in files_containing_this_category:
            file_counter = file2category_counter[file]
            catids_in_this_file = list(file_counter.keys())
            not_exceed = [catid for catid in catids_in_this_file if few_shot_cat_counter[catid] + file_counter[catid] <= n]
            if len(not_exceed) == len(catids_in_this_file):
                few_shot.append(file)
                for catid in catids_in_this_file:
                    few_shot_cat_counter[catid] += file_counter[catid]

    # supplement few_shot if some categories are not extracted enough
    for catid in tqdm(range(nc)):
        # if catid has not been extracted enough
        if few_shot_cat_counter[catid] < n:
            # sort files by the number of instances of catid in each file
            file_counter = {file:counter for file, counter in file2category_counter.items() if catid in file2category_counter[file] and file not in few_shot}
            file_counter = {k: v for k, v in sorted(file_counter.items(), key=lambda item: item[1][catid], reverse=False)}

            for file in file_counter:
                catids_in_this_file = list(file_counter[file].keys())
                not_exceed = [catid for catid in catids_in_this_file if few_shot_cat_counter[catid] + file_counter[file][catid] <= n + error]
                if len(not_exceed) == len(catids_in_this_file):
                    few_shot.append(file)
                    for catid in catids_in_this_file:
                        few_shot_cat_counter[catid] += file_counter[file][catid]
                    if few_shot_cat_counter[catid] >= n:
                        break
                    
    return few_shot

def select_few_shot(file2category_counter, n=16, nc=2):
    few_shot = []
    keys = list(file2category_counter.keys())
    np.random.shuffle(keys)
    few_shot_cat_counter = {i:0 for i in range(nc)}
    for file in tqdm(keys):
        cat_counter = file2category_counter[file]
        if len(cat_counter) == 0:
            continue
        catids = list(cat_counter.keys())
        
        not_exceed = [catid for catid in catids if few_shot_cat_counter[catid] + cat_counter[catid] <= n]
        if len(not_exceed) == len(catids):
            few_shot.append(file)
            for catid in catids:
                few_shot_cat_counter[catid] += cat_counter[catid]

    return few_shot

def copy_few_shot(src_img_rt, src_lbl_rt, dst_train_img_rt, dst_train_lbl_rt, dst_test_img_rt, dst_test_lbl_rt, few_shot):
    for file in tqdm(few_shot):
        for ext in ['jpg', 'bmp', 'png', 'jpeg', 'tif', 'tiff']:
            if os.path.exists(os.path.join(src_img_rt, file+'.'+ext)):
                shutil.copy(os.path.join(src_img_rt, file+'.'+ext), os.path.join(dst_train_img_rt, file+'.'+ext))
                shutil.copy(os.path.join(src_img_rt, file+'.'+ext), os.path.join(dst_test_img_rt, file+'.'+ext))
                break
        shutil.copy(os.path.join(src_lbl_rt, file+'.txt'), os.path.join(dst_train_lbl_rt, file+'.txt'))
        shutil.copy(os.path.join(src_lbl_rt, file+'.txt'), os.path.join(dst_test_lbl_rt, file+'.txt'))

def count_cats_in_few_shot(few_shot, file2category):
    cats = []
    for file in few_shot:
        cats.extend(file2category[file])
    return Counter(cats)

def count_cats(file2category):
    cats = []
    for k, v in file2category.items():
        cats.extend(v)
    return Counter(cats)

infos = initialize(src_dataset_rt, dst_dataset_rt)
src_img_rt, src_lbl_rt, dst_train_img_rt, dst_train_lbl_rt, dst_test_img_rt, dst_test_lbl_rt, file2category, file2category_counter, category2file, category2file_counter = infos
# few_shot = select_few_shot(file2category_counter, n=shot, nc=nc)
few_shot = class_driven_select_few_shot(category2file_counter, file2category_counter, n=shot, nc=nc)
few_shot_counter = count_cats_in_few_shot(few_shot, file2category)
cats_counter = count_cats(file2category)
copy_few_shot(src_img_rt, src_lbl_rt, dst_train_img_rt, dst_train_lbl_rt, dst_test_img_rt, dst_test_lbl_rt, few_shot)




