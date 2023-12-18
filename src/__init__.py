import os

DIR = {
    'DATASET': {
        'RAW': os.path.abspath('../../data/raw'),
        'PROCESSED': {
            'TRAIN': os.path.abspath('../../data/processed/train'),
            'TEST': os.path.abspath('../../data/processed/test'),
            'RECOMMENDER': os.path.abspath('../../data/processed/recommender'),
        },
    },
    'MODELS': os.path.abspath('../../models'),
    'LOG': os.path.abspath('../../log')
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
