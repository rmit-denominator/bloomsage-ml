import os

from tqdm.auto import tqdm
import tensorflow as tf
from sklearn.cluster import KMeans
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import joblib

from src.data import dataset
from src import PATH, MODELS
from src.model import INPUT_IMG_DIM
from src.util.image import resize_crop
from src.util.image import normalize_pixels


dir_raw_dataset = PATH['DATASET']['RAW']
dir_train_dataset = PATH['DATASET']['PROCESSED']['TRAIN']
dir_test_dataset = PATH['DATASET']['PROCESSED']['TEST']
dir_recommender_database = PATH['DATASET']['PROCESSED']['RECOMMENDER']
path_recommender_database_metadata = PATH['DATASET']['PROCESSED']['RECOMMENDER_META']
dir_models = PATH['MODELS']

model_clf_name = MODELS['CLASSIFIER']
model_feature_extractor_name = MODELS['FEATURE_EXTRACTOR']
model_clustering_name = MODELS['CLUSTERING']


def train_clf():
    df = dataset.load(dir_train_dataset)
    train, val = train_test_split(df, shuffle=True, test_size=0.25, random_state=42)
    
    # Model Parameters
    BATCH_SIZE = 512
    INPUT_DIM = (INPUT_IMG_DIM, INPUT_IMG_DIM, 3)  # RGB - 3 channels images
    OUTPUT_CLASSES = 8  # One-hot encoded: 8 different classes

    # Training Parameters
    EPOCHS = 32
    LEARNING_RATE = 1e-3
    MOMENTUM = 0.9

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(4, (4, 4), activation='relu', input_shape=INPUT_DIM),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(8, (4, 4), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (4, 4), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (4, 4), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (4, 4), activation='relu'),
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5, seed=21),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5, seed=42),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(OUTPUT_CLASSES),
    ], name=model_clf_name)

    model.compile(
        optimizer=RMSprop(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['categorical_accuracy'],
    )

    train['Class'] = train['Class'].astype('str')
    train_datagen = ImageDataGenerator(data_format='channels_last')
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=dir_train_dataset,
        x_col='ImgPath',
        y_col='Class',
        target_size=(INPUT_IMG_DIM, INPUT_IMG_DIM),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=42
    )

    val['Class'] = val['Class'].astype('str')
    val_datagen = ImageDataGenerator(data_format='channels_last')
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val,
        directory=dir_train_dataset,
        x_col='ImgPath',
        y_col='Class',
        target_size=(INPUT_IMG_DIM, INPUT_IMG_DIM),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=42
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(PATH['LOG'], model_clf_name),
        write_graph=False
    )
    model.fit(
        train_generator, validation_data=val_generator,
        epochs=EPOCHS,
        steps_per_epoch=train.shape[0] // BATCH_SIZE,
        validation_steps=val.shape[0] // BATCH_SIZE,
        verbose=1, callbacks=[tensorboard_callback],
    )

    test = dataset.load(dir_test_dataset)
    test['Class'] = test['Class'].astype('str')
    test_datagen = ImageDataGenerator(data_format='channels_last')
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test,
        directory=dir_test_dataset,
        x_col='ImgPath',
        y_col='Class',
        target_size=(INPUT_IMG_DIM, INPUT_IMG_DIM),
        batch_size=1,
        class_mode='categorical',
        seed=42
    )

    model.evaluate(test_generator)
    model.save(
        os.path.join(PATH['MODELS'], model_clf_name),
        overwrite=True,
        save_format='h5'
    )


def build_fe():
    INPUT_DIM = (INPUT_IMG_DIM, INPUT_IMG_DIM, 3)  # RGB - 3 channels images
    FEATURE_VEC_DIM = 16

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (4, 4), activation='relu', input_shape=INPUT_DIM),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (4, 4), activation='relu'),
        tf.keras.layers.AveragePooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(FEATURE_VEC_DIM),
    ], name=model_feature_extractor_name)

    model.compile(optimizer='adam', loss='mse')
    model.save(
        os.path.join(PATH['MODELS'], model_feature_extractor_name),
        overwrite=True,
        save_format='h5'
    )

    df = dataset.load(dir_recommender_database)

    recommendations = {'ImgPath': [], 'Class': []}
    for j in range(FEATURE_VEC_DIM):
        recommendations[f'x{j}'] = []

    with tqdm(total=len(df), desc='Extracting feature vectors from recommender-database', position=0,
              leave=True) as pbar:
        for i, row in df.iterrows():
            pbar.update()
            recommendations['ImgPath'].append(row['ImgPath'])
            recommendations['Class'].append(row['Class'])
            with Image.open(f'{dir_recommender_database}{recommendations["ImgPath"][-1]}') as ref:
                ref_processed = resize_crop(ref, INPUT_IMG_DIM, INPUT_IMG_DIM)
                ref_processed = normalize_pixels(ref_processed)
                ref_processed = tf.expand_dims(ref_processed, axis=0)

                ref_feature_vector = model.predict(ref_processed, verbose=0)
                for j, feature in enumerate(ref_feature_vector.reshape(-1)):
                    recommendations[f'x{j}'].append(feature)

    df_feature_vectors = pd.DataFrame(recommendations)
    df_feature_vectors.to_csv(path_recommender_database_metadata, index=False)


def build_clu():
    clu = KMeans(init='k-means++', n_init='auto')
    joblib.dump(clu, os.path.join(PATH['MODELS'], model_clustering_name))


if __name__ == '__main__':
    print('Training classifier model...')
    train_clf()
    print('[ DONE ]')

    print('Building feature extractor (recommendation system)...')
    build_fe()
    print('[ DONE ]')

    print('Building clustering model (recommendation system)...')
    build_clu()
    print('[ DONE ]')
