import sys
import os
import math
import random
import bisect
import pickle
import time
import numpy as np
from nebula3_database.database.arangodb import DatabaseConnector
from nebula3_database.config import NEBULA_CONF
# from movie_db import MOVIE_DB
import cv2
from pathlib import Path

from nebula3_database.movie_db import MOVIE_DB
import tqdm
from PIL import Image
from nebula3_videoprocessing.videoprocessing.ontology_implementation import SingleOntologyImplementation
import nebula_vg_driver.visual_genome.local as vg
import pandas as pd
import json


class PIPELINE:
    def __init__(self):
        self.config = NEBULA_CONF()
        self.db_host = self.config.get_database_host()
        self.database = self.config.get_playground_name()
        self.gdb = DatabaseConnector()
        self.db = self.gdb.connect_db(self.database)
        self.nre = MOVIE_DB()
        self.nre.change_db("visualgenome")
        self.db = self.nre.db

pipeline = PIPELINE()



result_path = os.path.join(os.getcwd(), 'vg_output')
res_file = os.path.join(result_path, "results_clip_vg.csv")
vgenome_images = '/datasets/visualgenome/VG/'
VG_DATA = "/notebooks/vg_data"
with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
    vg_objects = json.load(f)

with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
    images_data = json.load(f)

image_ids = [obj['image_id'] for obj in images_data]

sample_ids = random.sample(range(1, len(image_ids) - 1), 1000)

with open(os.path.join(result_path, 'sample_ids.txt'), 'w') as f:
    for item in sample_ids:
        f.write("%s\n" % item)

# Load ontology
obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

if not os.path.exists(result_path):
    os.makedirs(result_path)
    print("Created output directory")

ontology_imp = SingleOntologyImplementation('vg_objects', 'clip')

idx = 0

results = list()
for vg_ob in tqdm.tqdm(sample_ids):
    cur_image_data = images_data[sample_ids[idx]]
    print("Object id", cur_image_data['image_id'])
    fname = os.path.basename(cur_image_data['url'])
    full_fname = os.path.join(vgenome_images, fname)
    img = Image.open(full_fname)
    scores = ontology_imp.compute_scores(img)
    obj_dict = dict()
    obj_dict.update({'image_id' : cur_image_data['image_id']})
    for i in range(len(scores)):
        obj_dict.update({ontology_imp.ontology[i]: scores[i][1]})
    results.append(obj_dict)
    # Intermediate save 
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_path, res_file), index=False)
    idx += 1

df = pd.DataFrame(results)
df.to_csv(os.path.join(result_path, res_file), index=False)
