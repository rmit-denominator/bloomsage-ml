import os

import kaggle
from tqdm.auto import tqdm
import imagehash
from PIL import Image
from sklearn.model_selection import train_test_split

from src.data import dataset
from src.model import INPUT_IMG_DIM
from src.util.image import augment
from src.util.image import normalize_pixels
from src.util.image import remove_transparency
from src.util.image import resize_crop
from src.util.system import clean_dir
from src import PATH

dir_raw_dataset = PATH['DATASET']['RAW']
dir_train_dataset = PATH['DATASET']['PROCESSED']['TRAIN']
dir_test_dataset = PATH['DATASET']['PROCESSED']['TEST']
dir_recommender_database = PATH['DATASET']['PROCESSED']['RECOMMENDER']


def fetch(dataset_src: str = 'miketvo/rmit-flowers'):
    print(f'Fetching raw dataset from {dataset_src}...')
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_src, path=dir_raw_dataset, unzip=True, quiet=False)
    print('[ DONE ]')


def make(raw_dataset_path: str = dir_raw_dataset):
    print(f'Building processed dataset...')
    df_raw = dataset.load(raw_dataset_path)

    # Duplicate image detection
    image_hashes = {}
    with tqdm(total=len(df_raw), desc='Finding identical images', position=0, leave=True) as pbar:
        for i, row in df_raw.iterrows():
            pbar.update()
            with Image.open(os.path.join(raw_dataset_path, row["ImgPath"])) as im:
                image_hash = imagehash.average_hash(im, hash_size=8)
                if image_hash in image_hashes:
                    image_hashes[image_hash].append(row["ImgPath"])
                else:
                    image_hashes[image_hash] = [row["ImgPath"]]

    duplicated_image_hashes = {hash_val: paths for hash_val, paths in image_hashes.items() if
                               len(paths) > 1}  # Remove hashes with a single path

    duplicated_images_paths = []
    for paths in duplicated_image_hashes.values():
        for i, path in enumerate(paths):
            if i > 0:  # Skipping the first copy
                duplicated_images_paths.append(path)

    # Dataset splitting
    train, test = train_test_split(df_raw, shuffle=True, test_size=0.2, random_state=42)

    # Train set
    with tqdm(total=len(train), desc='Setting up train dataset', position=0, leave=True) as pbar:
        for i, row in train.iterrows():
            pbar.update()
            if not os.path.exists(os.path.join(dir_train_dataset, row["Class"])):
                os.makedirs(os.path.join(dir_train_dataset, row["Class"]))

            img_path = row['ImgPath']
            new_img_path = ''.join(img_path.split('.')[0:-1]) + '.jpg'
            with Image.open(os.path.join(raw_dataset_path, img_path)) as im:
                if im.mode == 'L':
                    continue  # Ignoring grayscale images

                if img_path in duplicated_images_paths:  # Augment duplicated image
                    im = augment(im, seed=42)
                im = remove_transparency(im)
                im = resize_crop(im, INPUT_IMG_DIM, INPUT_IMG_DIM)
                im = normalize_pixels(im)

                im.save(os.path.join(dir_train_dataset, new_img_path))

    # Test set
    with tqdm(total=len(test), desc='Setting up test dataset', position=0, leave=True) as pbar:
        for i, row in test.iterrows():
            pbar.update()
            if not os.path.exists(os.path.join(dir_test_dataset, row["Class"])):
                os.makedirs(os.path.join(dir_test_dataset, row["Class"]))

            img_path = row['ImgPath']
            new_img_path = ''.join(img_path.split('.')[0:-1]) + '.jpg'
            with Image.open(os.path.join(raw_dataset_path, img_path)) as im:
                if im.mode == 'L':
                    continue  # Ignoring grayscale images

                if img_path in duplicated_images_paths:  # Augment duplicated image
                    im = augment(im, seed=42)
                im = remove_transparency(im)
                im = resize_crop(im, INPUT_IMG_DIM, INPUT_IMG_DIM)
                im = normalize_pixels(im)

                im.save(os.path.join(dir_test_dataset, new_img_path))

    # Recommender database
    with tqdm(total=len(df_raw), desc='Setting up recommender database', position=0, leave=True) as pbar:
        for i, row in df_raw.iterrows():
            pbar.update()
            if not os.path.exists(os.path.join(dir_recommender_database, row["Class"])):
                os.makedirs(os.path.join(dir_recommender_database, row["Class"]))

            img_path = row['ImgPath']
            new_img_path = ''.join(img_path.split('.')[0:-1]) + '.jpg'
            with Image.open(os.path.join(raw_dataset_path, img_path)) as im:
                if im.mode == 'L':
                    continue  # Ignoring grayscale images

                if img_path not in duplicated_images_paths:
                    im = remove_transparency(im)
                    im.save(os.path.join(dir_recommender_database, new_img_path))

    print('[ DONE ]')


if __name__ == '__main__':
    print('Cleaning dataset directory...', end='')
    clean_dir(os.path.abspath('../../data'))
    print(' [ DONE ]')
    fetch()
    make()
