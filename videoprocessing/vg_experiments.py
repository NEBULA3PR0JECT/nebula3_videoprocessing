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
import csv

from nebula3_database.movie_db import MOVIE_DB
import tqdm
from PIL import Image
# from nebula3_videoprocessing.videoprocessing.ontology_implementation import SingleOntologyImplementation, EnsembleOntologyImplementation
from nebula3_videoprocessing.videoprocessing.vlm_factory import VlmFactory
import nebula_vg_driver.visual_genome.local as vg
import pandas as pd
import json
import itertools
import pandas as pd
from blip import BLIP_Captioner

from nebula3_videoprocessing.videoprocessing.utils.config import config

def open_json(json_path):
    with open(json_path) as f:
            data = json.load(f)
    return data

def open_csv(csv_path):
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

class VG_EXPERIMENT:
    def __init__(self):
        self.config_db = NEBULA_CONF()
        self.db_host = self.config_db.get_database_host()
        self.database = self.config_db.get_playground_name()
        self.gdb = DatabaseConnector()
        self.db = self.gdb.connect_db(self.database)
        self.nre = MOVIE_DB()
        self.nre.change_db("visualgenome")
        self.db = self.nre.db

    def vg_experiment_caption(self, captioner='blip'):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, f"results_captions_{captioner}_vg.json")
        vgenome_images = '/datasets/visualgenome/VG/'

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)
        
        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        blip_captioner = BLIP_Captioner()
    
        idx = 0

        results = dict()
        for vg_ob in tqdm.tqdm(sample_ids):
            cur_image_data = images_data[sample_ids[idx]]
            print("Object id", cur_image_data['image_id'])
            fname = os.path.basename(cur_image_data['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            processed_frame = blip_captioner.process_frame(img)
            # print(f"URL: {cur_image_data['url']}")
            caption = blip_captioner.generate_caption(processed_frame)
            # print(f"caption: {caption}")
            obj_dict = dict()
            obj_dict.update({str(cur_image_data['image_id']) : caption})
            results.update(obj_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)        
            idx += 1




    def vg_experiment_topdown(self, ontology_name = 'vg_objects', vlm_name='blip_itc'):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, f"results_{vlm_name}_{ontology_name}_vg.csv")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
            vg_objects = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids = [obj['image_id'] for obj in images_data]

        
        # sample_ids = random.sample(range(1, len(image_ids) - 1), 1000)

        # with open(os.path.join(result_path, 'sample_ids.txt'), 'w') as f:
        #     for item in sample_ids:
        #         f.write("%s\n" % item)

        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]

        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        ontology_imp = SingleOntologyImplementation(ontology_name, vlm_name)

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

    
    def vg_experiment_topdown_ensemble(self, ontology_name = 'vg_objects', vlm_names=['blip_itc', 'blip_itm']):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, f"results_{'_'.join(vlm_names)}_{ontology_name}_vg.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
            vg_objects = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids = [obj['image_id'] for obj in images_data]


        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]

        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        ontology_imp = MultipleOntologyImplementation(ontology_name, vlm_names)

        counter = 0
        results = dict()
        for vg_ob in tqdm.tqdm(sample_ids):
            cur_image_data = images_data[sample_ids[counter]]
            image_id = cur_image_data['image_id']
            print("Object id", image_id)
            fname = os.path.basename(cur_image_data['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            scores = ontology_imp.compute_scores(img)
            sorted_scores = [sorted(vlm_scores, key=lambda x: x[1], reverse=True) for vlm_scores in scores]
            obj_dict = dict()
            obj_dict.update({image_id : {}})
            averaged_results = dict()
            averaged_results.update({image_id : {}})
            # Iterate over the VLMs Scores
            for idx, vlm_tuples in enumerate(sorted_scores):
                obj_dict[image_id].update({idx: {}})
                # if idx == 0:
                #     averaged_results[image_id].update({idx: {}})
                # Iterate over all the ontology and its scores from the VLM
                for i in range(0, len(vlm_tuples)):
                    obj_dict[image_id][idx].update({vlm_tuples[i][0]: i})
                    if idx == 0:
                        averaged_results[image_id].update({vlm_tuples[i][0]: i})
                    if idx > 0: # Make sure we check the results from at least the 2nd VLM outputs.
                        # Sum indices (Average later)
                        averaged_results[image_id][vlm_tuples[i][0]] = (obj_dict[image_id][idx][vlm_tuples[i][0]] +
                        obj_dict[image_id][idx - 1][vlm_tuples[i][0]])
            # Average results (We already summed, so just divide)
            for image_id in averaged_results:
                for key, val in averaged_results[image_id].items():
                # Make sure we don't divide by zero if all the results is first index
                    if averaged_results[image_id][key] == 0:
                        averaged_results[image_id][key] = 0
                    else:
                        averaged_results[image_id][key] = averaged_results[image_id][key] / len(sorted_scores)
            averaged_results = dict(sorted(averaged_results[image_id].items(), key=lambda item: item[1]), reverse=True)
            results.update({image_id : averaged_results})
            with open(res_file, 'w') as fout:
                json.dump(results, fout)
            # Intermediate save 
            counter += 1

    
    def vg_experiment_topdown_paragraphs(self, vlm_name='blip_itc', with_split=False):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        if with_split:
            res_file = os.path.join(result_path, f"results_{vlm_name}_paragraphs_vg_split.json")
        else:
            res_file = os.path.join(result_path, f"results_{vlm_name}_paragraphs_vg.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
            vg_objects = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids = [obj['image_id'] for obj in images_data]


        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]

        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        vlm = VlmFactory().get_vlm(vlm_name)

        idx = 0
        results = {}
        for vg_ob in tqdm.tqdm(sample_ids):
            cur_image_data = images_data[sample_ids[idx]]
            image_id = cur_image_data['image_id']
            print("Object id", image_id)
            fname = os.path.basename(cur_image_data['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            if with_split:
                paragraphs = cur_image_data['paragraph'].split('.')
                paragraphs = [paragraph for paragraph in paragraphs if paragraph]
            else:
                paragraphs = cur_image_data['paragraph']
            scores = vlm.compute_similarity(img, paragraphs)

            paragraphs_dict = dict()
            paragraphs_dict.update({image_id :{}})
            for i in range(len(scores)):
                paragraphs_dict[image_id].update({paragraphs[i]: str(scores[i])})
            results.update(paragraphs_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)
            idx += 1
    
    def vg_experiment_topdown_paragraphs_random(self, vlm_name='blip_itc', with_split=False):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        if with_split:
            res_file = os.path.join(result_path, f"results_{vlm_name}_paragraphs_vg_random_split.json")
        else:
            res_file = os.path.join(result_path, f"results_{vlm_name}_paragraphs_vg_random.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
            vg_objects = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids = [obj['image_id'] for obj in images_data]


        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]


        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        all_paragraphs = [img_data['paragraph'] for img_data in images_data]

        vlm = VlmFactory().get_vlm(vlm_name)

        idx = 0
        results = {}
        for vg_ob in tqdm.tqdm(sample_ids):
            cur_image_data = images_data[sample_ids[idx]]
            image_id = cur_image_data['image_id']
            print("Object id", image_id)
            fname = os.path.basename(cur_image_data['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            paragraphs = random.sample(all_paragraphs, 5)
            if with_split:
                paragraphs = [paragraph.split('.') for paragraph in paragraphs]
                paragraphs = list(itertools.chain.from_iterable(paragraphs))
                paragraphs = [paragraph.lstrip() for paragraph in paragraphs if len(paragraph) > 1]
            else:
                paragraphs = cur_image_data['paragraph']
           
            scores = vlm.compute_similarity(img, paragraphs)

            paragraphs_dict = dict()
            paragraphs_dict.update({image_id :{}})
            for i in range(len(scores)):
                paragraphs_dict[image_id].update({paragraphs[i]: str(scores[i])})
            results.update(paragraphs_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)
            idx += 1

    def clip_vg_relations_experiment(self):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, "results_clip_vg_relations.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "ipc_triplet_relations.txt")) as f:
            vg_triplets = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids_to_index = [{obj['image_id']: idx} for idx, obj in enumerate(images_data) if str(obj['image_id']) in vg_triplets.keys()]
        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        # ontology_imp = SingleOntologyImplementation('vg_objects', 'clip')
        vlm_imp = VlmFactory().get_vlm('clip')

        vg_triplets_parsed = {}
        for image_id, triplets in vg_triplets.items():
            str_triplets = []
            for triplet in triplets:
                str_triplet = ' '.join(triplet).lower()
                str_triplets.append(str_triplet)
            vg_triplets_parsed.update({image_id: str_triplets})
            
        # idx = 0

        results = {}
        for data in tqdm.tqdm(image_ids_to_index):
            image_id, cur_idx = list(data.items())[0][0], list(data.items())[0][1]
            print("Object id", image_id)
            fname = os.path.basename(images_data[cur_idx]['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            text = vg_triplets_parsed[str(image_id)]
            if text:
                scores = vlm_imp.compute_similarity(img, text)
            else:
                print(f"Skipped {image_id}")
                continue
            triplets_dict = dict()
            triplets_dict.update({image_id :{}})
            for i in range(len(scores)):
                triplets_dict[image_id].update({vg_triplets_parsed[str(image_id)][i]: str(scores[i])})
            results.update(triplets_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)
            # idx += 1
        
    def blip_vg_relations_experiment(self):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, "results_blip_vg_relations.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "ipc_triplet_relations.txt")) as f:
            vg_triplets = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids_to_index = [{obj['image_id']: idx} for idx, obj in enumerate(images_data) if str(obj['image_id']) in vg_triplets.keys()]
        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        # ontology_imp = SingleOntologyImplementation('vg_objects', 'clip')
        vlm_imp = VlmFactory().get_vlm('blip_itc')

        vg_triplets_parsed = {}
        for image_id, triplets in vg_triplets.items():
            str_triplets = []
            for triplet in triplets:
                str_triplet = ' '.join(triplet).lower()
                str_triplets.append(str_triplet)
            vg_triplets_parsed.update({image_id: str_triplets})
            
        # idx = 0

        results = {}
        for data in tqdm.tqdm(image_ids_to_index):
            image_id, cur_idx = list(data.items())[0][0], list(data.items())[0][1]
            print("Object id", image_id)
            fname = os.path.basename(images_data[cur_idx]['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            text = vg_triplets_parsed[str(image_id)]
            if text:
                scores = vlm_imp.compute_similarity(img, text)
            else:
                print(f"Skipped {image_id}")
                continue
            triplets_dict = dict()
            triplets_dict.update({image_id :{}})
            for i in range(len(scores)):
                triplets_dict[image_id].update({vg_triplets_parsed[str(image_id)][i]: str(scores[i])})
            results.update(triplets_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)
            # idx += 1
    
    def blip_vg_relations_experiment_random(self):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, "results_blip_vg_relations_random.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "ipc_predicates.txt")) as f:
            vg_predicates = [line.strip() for line in f.readlines()]
        
        with open(os.path.join(VG_DATA, "ipc_triplet_relations.txt")) as f:
            vg_triplets = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids_to_index = [{obj['image_id']: idx} for idx, obj in enumerate(images_data) if str(obj['image_id']) in vg_triplets.keys()]
        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        # ontology_imp = SingleOntologyImplementation('vg_objects', 'clip')
        vlm_imp = VlmFactory().get_vlm('blip_itc')

        vg_triplets_parsed = {}
        for image_id, triplets in vg_triplets.items():
            str_triplets = []
            for triplet in triplets:
                random_predicate = random.choice(vg_predicates)
                triplet[1] = random_predicate
                str_triplet = ' '.join(triplet).lower()
                str_triplets.append(str_triplet)
            vg_triplets_parsed.update({image_id: str_triplets})

        results = {}
        for data in tqdm.tqdm(image_ids_to_index):
            image_id, cur_idx = list(data.items())[0][0], list(data.items())[0][1]
            print("Object id", image_id)
            fname = os.path.basename(images_data[cur_idx]['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            text = vg_triplets_parsed[str(image_id)]
            if text:
                scores = vlm_imp.compute_similarity(img, text)
            else:
                print(f"Skipped {image_id}")
                continue
            triplets_dict = dict()
            triplets_dict.update({image_id :{}})
            for i in range(len(scores)):
                triplets_dict[image_id].update({vg_triplets_parsed[str(image_id)][i]: str(scores[i])})
            results.update(triplets_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)
            # idx += 1

    def clip_vg_relations_experiment_random(self):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, "results_clip_vg_relations_random.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "ipc_predicates.txt")) as f:
            vg_predicates = [line.strip() for line in f.readlines()]
        
        with open(os.path.join(VG_DATA, "ipc_triplet_relations.txt")) as f:
            vg_triplets = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids_to_index = [{obj['image_id']: idx} for idx, obj in enumerate(images_data) if str(obj['image_id']) in vg_triplets.keys()]
        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        # ontology_imp = SingleOntologyImplementation('vg_objects', 'clip')
        vlm_imp = VlmFactory().get_vlm('clip')

        vg_triplets_parsed = {}
        for image_id, triplets in vg_triplets.items():
            str_triplets = []
            for triplet in triplets:
                random_predicate = random.choice(vg_predicates)
                triplet[1] = random_predicate
                str_triplet = ' '.join(triplet).lower()
                str_triplets.append(str_triplet)
            vg_triplets_parsed.update({image_id: str_triplets})

        results = {}
        for data in tqdm.tqdm(image_ids_to_index):
            image_id, cur_idx = list(data.items())[0][0], list(data.items())[0][1]
            print("Object id", image_id)
            fname = os.path.basename(images_data[cur_idx]['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            text = vg_triplets_parsed[str(image_id)]
            if text:
                scores = vlm_imp.compute_similarity(img, text)
            else:
                print(f"Skipped {image_id}")
                continue
            triplets_dict = dict()
            triplets_dict.update({image_id :{}})
            for i in range(len(scores)):
                triplets_dict[image_id].update({vg_triplets_parsed[str(image_id)][i]: str(scores[i])})
            results.update(triplets_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)
            # idx += 1
    
    def compute_average_relations(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        average_lst = []
        for image_id, triplet_to_scores in data.items():
            all_values = triplet_to_scores.values()
            all_values = [float(score) for score in all_values]
            avg = np.average(all_values) 
            average_lst.append(avg)
        output_avg = np.average(average_lst)
        print(f"Average: {output_avg}")

    def insert_vg_experiments_to_db(self, objects_path, captions_path, persons_path, scenes_path, ipc_path, db_name="nebula_playground", source='visualgenome'):
        self.nre.change_db(db_name)
        self.db = self.nre.db
        
        captions_data = open_json(captions_path)
        ipc_data = open_json(ipc_path)
        objects_data = open_csv(objects_path)
        persons_data = open_csv(persons_path)
        scenes_data = open_csv(scenes_path)

        img_ids_to_ontology = {}
        # Get all the image ids to the dict
        for img_id, _ in captions_data.items():
            img_ids_to_ontology.update({img_id: {"objects": {"blip": {}}, "captions": {"blip": {}}, "persons": {"blip": {}}, "scenes": {"blip": {}}}})

        idx = 1

        for img_id, _ in img_ids_to_ontology.items():
            
            # Get all the ontologies of the current image id
            captions = captions_data[img_id]
            # CSV to JSON
            persons_data_dict, scenes_data_dict, objects_data_dict = {}, {}, {}
            # Add all the ontology objects and their respective scores to the current image
            for j in range(1, len(persons_data[0])):
                if persons_data[idx][0] == img_id:
                    persons_data_dict.update({persons_data[0][j] : persons_data[idx][j]})
                else:
                    print("Error.")

             # Add all the ontology objects and their respective scores to the current image
            for j in range(1, len(scenes_data[0])):
                if scenes_data[idx][0] == img_id:
                    scenes_data_dict.update({scenes_data[0][j] : scenes_data[idx][j]})
                else:
                    print("Error.")
            
             # Add all the ontology objects and their respective scores to the current image
            for j in range(1, len(objects_data[0])):
                if objects_data[idx][0] == img_id:
                    objects_data_dict.update({objects_data[0][j] : objects_data[idx][j]})
                else:
                    print("Error.")
            idx +=1

            # a = sorted(objects_data_dict.items(), key=lambda x: x[1], reverse=True)
            # objects_data_dicts = [{key: val} for key,val in a]
            
            # objects_data_dicts = [{key,val} for key,val in objects_data_dict]
            # Insert all the ontologies to the current image id
            img_ids_to_ontology[img_id]['objects']['blip'] = [{k: float(v)} for (k, v) in sorted(objects_data_dict.items(), key=lambda x: x[1], reverse=True)]
            img_ids_to_ontology[img_id]['captions']['blip'] = captions
            img_ids_to_ontology[img_id]['persons']['blip'] = [{k: float(v)} for (k, v) in sorted(persons_data_dict.items(), key=lambda x: x[1], reverse=True)]
            img_ids_to_ontology[img_id]['scenes']['blip'] = [{k: float(v)} for (k, v) in sorted(scenes_data_dict.items(), key=lambda x: x[1], reverse=True)]


            query = 'UPSERT { image_id: @image_id } INSERT  \
                { image_id: @image_id, url: @url, global_objects: @global_objects, global_captions: @global_captions,\
                            global_persons: @global_persons, global_scenes: @global_scenes, source: @source\
                        } UPDATE {image_id: @image_id, url: @url, global_objects: @global_objects, global_captions: @global_captions,\
                            global_persons: @global_persons, global_scenes: @global_scenes, \
                            source: @source} IN s3_global_tokens1'
                            
            img_url = os.path.join('https://cs.stanford.edu/people/rak248/VG_100K', img_id + '.jpg')
            bind_vars = {
                            "image_id": int(img_id),
                            "global_objects": {"blip" : img_ids_to_ontology[img_id]['objects']['blip']},
                            "global_captions": {"blip" : img_ids_to_ontology[img_id]['captions']['blip']},
                            "global_persons": {"blip" : img_ids_to_ontology[img_id]['persons']['blip']},
                            "global_scenes": {"blip" : img_ids_to_ontology[img_id]['scenes']['blip']},
                            "url": img_url,
                            "source": source
                            }
            # print(bind_vars)
            print(idx)
            self.db.aql.execute(query, bind_vars=bind_vars)

    

def main():
    vg_experiment = VG_EXPERIMENT()
    # vg_experiment.vg_experiment_caption(captioner='blip')
    # vg_experiment.vg_experiment_topdown(ontology_name='scenes', vlm_name='blip_itc')
    # vg_experiment.vg_experiment_topdown_ensemble(ontology_name='vg_objects', vlm_names=['clip','blip_itc'])
    # df = pd.read_csv('/notebooks/vg_output/results_blip_itc_persons_vg.csv')
    # print(len(df))
    # a=0
    # vg_experiment.vg_experiment_topdown_paragraphs(vlm_name='blip_itc', with_split=True)
    # vg_experiment.vg_experiment_topdown_paragraphs_random(vlm_name='blip_itc', with_split=True)
    # file_path = "/notebooks/vg_output/results_blip_itc_paragraphs_vg_split.json"
    # vg_experiment.compute_average_relations(file_path)
    # file_path = "/notebooks/vg_output/results_blip_itc_paragraphs_vg_random_split.json"
    # vg_experiment.compute_average_relations(file_path)
    # vg_experiment.blip_vg_relations_experiment()
    # vg_experiment.blip_vg_relations_experiment_random()
    # vg_experiment.clip_vg_relations_experiment_random()
    # file_path = "/notebooks/vg_output/results_blip_vg_relations.json"
    # vg_experiment.compute_average_relations(file_path)
    # file_path = "/notebooks/vg_output/results_blip_vg_relations_random.json"
    # vg_experiment.compute_average_relations(file_path)
    result_path_base = os.path.join(os.getcwd(), 'vg_output')
    objects_path = os.path.join(result_path_base, "results_clip_vg.csv")
    captions_path = os.path.join(result_path_base, "results_captions_blip_vg.json")
    persons_path = os.path.join(result_path_base, "results_blip_itc_persons_vg.csv")
    scenes_path = os.path.join(result_path_base, "results_blip_itc_scenes_vg.csv")
    ipc_path = "/storage/ipc_data/paragraphs_v1.json"
    vg_experiment.insert_vg_experiments_to_db(objects_path, captions_path, persons_path, scenes_path, ipc_path, db_name="nebula_playground")
if __name__ == '__main__':
    main()