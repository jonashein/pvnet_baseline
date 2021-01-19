from lib.config import cfg


class DatasetCatalog(object):
    dataset_attrs = {
        'SynColibriV1_Train': {
            'id': 'custom',
            'data_root': 'data/syn_colibri_v1_train',
            'ann_file': 'data/syn_colibri_v1_train/train.json',
            'split': 'train'
        },
        'SynColibriV1_Val': {
            'id': 'custom',
            'data_root': 'data/syn_colibri_v1_val',
            'ann_file': 'data/syn_colibri_v1_val/train.json',
            'split': 'test'
        },
        'SynColibriV1_Test': {
            'id': 'custom',
            'data_root': 'data/syn_colibri_v1_test',
            'ann_file': 'data/syn_colibri_v1_test/train.json',
            'split': 'test'
        },
        'RealColibriV1_Train': {
            'id': 'custom',
            'data_root': 'data/real_colibri_v1_train',
            'ann_file': 'data/real_colibri_v1_train/train.json',
            'split': 'train'
        },
        'RealColibriV1_Val': {
            'id': 'custom',
            'data_root': 'data/real_colibri_v1_val',
            'ann_file': 'data/real_colibri_v1_val/train.json',
            'split': 'test'
        },
        'RealColibriV1_Test': {
            'id': 'custom',
            'data_root': 'data/real_colibri_v1_test',
            'ann_file': 'data/real_colibri_v1_test/train.json',
            'split': 'test'
        }
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()
