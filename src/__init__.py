import os

PATH = {
    'DATASET': {
        'RAW': os.path.abspath('../../data/raw'),
        'PROCESSED': {
            'TRAIN': os.path.abspath('../../data/processed/train'),
            'TEST': os.path.abspath('../../data/processed/test'),
            'RECOMMENDER': os.path.abspath('../../data/processed/recommender'),
            'RECOMMENDER_META': os.path.abspath('../../data/processed/recommender.csv')
        },
    },
    'MODELS': os.path.abspath('../../models'),
    'LOG': os.path.abspath('../../log')
}

MODELS = {
    'CLASSIFIER': 'clf-cnn',
    'FEATURE_EXTRACTOR': 'fe-cnn',
    'CLUSTERING': 'clu-kmeans'
}

CLASS_LABELS = [
    'Baby',
    'Calimerio',
    'Chrysanthemum',
    'Hydrangeas',
    'Lisianthus',
    'Pingpong',
    'Rosy',
    'Tana'
]

REMOTE = {
    'MODELS': 'rmit-denominator/bloomsage',
    'RECOMMENDER_DATA': 'rmit-denominator/recommender-data'
}
