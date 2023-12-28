import os

import pandas as pd
from transformers import AutoModel
from datasets import Dataset

from src import PATH
from src import MODELS
from src import REMOTE


dir_models = PATH['MODELS']
model_clf_name = MODELS['CLASSIFIER']
model_feature_extractor_name = MODELS['FEATURE_EXTRACTOR']
model_clustering_name = MODELS['CLUSTERING']


def push_model(model_file: str, remote: str):
    model = AutoModel.from_pretrained(model_file)
    model.push_to_hub(remote)


def push_dataset(dataset_csv: str, remote: str):
    data = pd.read_csv(dataset_csv)
    hf_dataset = Dataset.from_pandas(data)
    hf_dataset.push_to_hub(remote)


if __name__ == '__main__':
    push_model(os.path.join(PATH['MODELS'], model_clf_name), REMOTE['MODELS'])
    push_model(os.path.join(PATH['MODELS'], model_feature_extractor_name), REMOTE['MODELS'])
    push_model(os.path.join(PATH['MODELS'], model_clustering_name), REMOTE['MODELS'])
    push_dataset(PATH['DATASET']['PROCESSED']['RECOMMENDER_META'], REMOTE['RECOMMENDER_DATA'])
