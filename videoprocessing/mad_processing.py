import json
import os
import h5py
import pprint


class MADProcessing:
    def __init__(self, main_folder):
        self.main_folder = main_folder
        self.train_ann = None
        self.test_ann = None
        self.val_ann = None
        pass

    def load_train_annotations(self):
        self.train_ann = json.load(open(os.path.join(self.main_folder, 'annotations/MAD_train.json'), 'r'))
        self.test_ann = json.load(open(os.path.join(self.main_folder, 'annotations/MAD_test.json'), 'r'))
        self.val_ann = json.load(open(os.path.join(self.main_folder, 'annotations/MAD_val.json'), 'r'))

    def get_train_movies(self):
        # Train movies do not have titles, only number. The translation between the numbers and the titles is
        # possible via MAD_id2imdb.json and then https://www.imdb.com/title/IMDB_ID/
        return list(set([v['movie'] for v in self.train_ann.values()]))

    def get_test_movies(self):
        return list(set([v['movie'] for v in self.test_ann.values()]))

    def get_val_movies(self):
        return list(set([v['movie'] for v in self.val_ann.values()]))

    def get_movie_details(self, movie_name):
        for v in self.val_ann.values():
            if movie_name in v['movie']:
                print(v)

def test1():
    mad = MADProcessing('/home/paperspace/data/MAD/')
    mad.load_train_annotations()
    # mad.get_movie_details('27_Dresses')
    mad.get_movie_details('Lost_Weekend')
    # train_movies = mad.get_train_movies()
    # test_movies = mad.get_test_movies()
    # val_movies = mad.get_val_movies()
    pass

    # Get all the information from 27 Dresses movies (in validation)


import csv
def read_lsmdc(dataset_path):
    new_db = []
    with open(dataset_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            key = row[0].split()[0]
            name = key[:key.find('.') - 3]
            new_db.append(name)
    return set(new_db)

if __name__ == "__main__":
    print('Testing MAD dataset')
    # Example of processing a MAD movie
    name1 = '/home/paperspace/deployment/nebula3_videoprocessing/videoprocessing/dataset1/lsmdc/LSMDC16_annos_test.csv'
    name2 = '/home/paperspace/deployment/nebula3_videoprocessing/videoprocessing/dataset1/lsmdc/LSMDC16_annos_training.csv'
    name3 = '/home/paperspace/deployment/nebula3_videoprocessing/videoprocessing/dataset1/lsmdc/LSMDC16_annos_val.csv'

    db1 = read_lsmdc(name1)
    db2 = read_lsmdc(name2)
    db3 = read_lsmdc(name3)
    test1()