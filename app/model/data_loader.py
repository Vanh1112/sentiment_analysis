import pandas as pd
import os
import logging

LEVELS = {'debug': logging.DEBUG,
          'info': logging.INFO,
          'warning': logging.WARNING,
          'error': logging.ERROR,
          'critical': logging.CRITICAL}


class DataLoader:
    """
    Read data from data_dir then put it into train and test dataframe
    """
    label_conf = {'0': 'negative', '1': 'positive', '2': 'neutral'}

    def __init__(self, log_level='info'):
        self.data_dir = 'app/resources/static/train_data'
        logging.basicConfig(level=LEVELS[log_level])
        self.log = logging.getLogger(self.__class__.__name__)

    def load_data(self):
        """
        :return: 2 dataframes both have columns list ['message','label']
        """
        if getattr(self, 'train', None) and getattr(self, 'test', None):
            return self.train, self.test
        else:
            self.train = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
            self.test = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
            self.log.info(f"Data: len(train)={len(self.train)}, len(test)={len(self.test)}")
            return self.train, self.test
