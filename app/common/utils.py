from glob import glob
import re
import pandas as pd


class DataLoader:
    """
    input: test_nhan_0.csv,test_nhan_1.csv,train_nhan_0.csv,train_nhan_1.csv

    Read data from data_dir then put it into train and test dataframe
    """
    label_conf = {'0': 'negative', '1': 'positive', '2': 'neutral'}

    def __init__(self):
        self.data_dir = 'app/resources/static/train_data'

    def load_data(self):
        """
        :return: 2 dataframes both have columns list ['message','label']
        """
        if getattr(self, 'train', None) and getattr(self, 'test', None):
            return self.train, self.test
        else:
            pattern = re.compile('nhan_(\d+).csv')
            train = {'message': [], 'label': []}
            test = {'message': [], 'label': []}
            for file in glob(self.data_dir + '/*.csv'):
                with open(file, 'r') as f:
                    data = f.read().splitlines()
                    # extract label number from data file then lookup in label config
                    label = [self.label_conf[re.findall(pattern, file)[0]]] * len(data)
                if 'train' in file.split('/')[-1]:
                    train['message'].extend(data)
                    train['label'].extend(label)
                else:
                    test['message'].extend(data)
                    test['label'].extend(label)
            self.train = pd.DataFrame(data=train)
            self.test = pd.DataFrame(data=test)
            return self.train, self.test
