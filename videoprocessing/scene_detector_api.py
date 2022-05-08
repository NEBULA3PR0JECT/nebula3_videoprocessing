from posixpath import basename
from typing import Counter
from scenedetect.video_splitter import split_video_ffmpeg, is_ffmpeg_available
# Standard PySceneDetect imports:
from scenedetect import VideoManager, scene_detector
from scenedetect import SceneManager
# For content-aware scene detection:
from scenedetect.detectors import ContentDetector
from pickle import STACK_GLOBAL
import cv2
import os
import logging
import uuid
# from arango import ArangoClient
from benchmark.clip_benchmark import NebulaVideoEvaluation
import numpy as np
import glob
#import redis
import boto3
import csv
import json
import sys
sys.path.insert(0, './')
sys.path.insert(0, 'nebula3_database/')
import torch
from PIL import Image
from nebula3_database.movie_db import MOVIE_DB
import clip

class NEBULA_SCENE_DETECTOR():
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            level=logging.INFO)
        self.video_eval = NebulaVideoEvaluation()
        self.nre = MOVIE_DB() 
        self.db = self.nre.db
        self.s3 = boto3.client('s3', region_name='eu-central-1')


    def detect_scene_elements(self, video_file):
        print("DEBUG: ", video_file)
        scenes = []
        video_manager = VideoManager([video_file])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30.0))
    # Improve processing speed by downscaling before processing.
        video_manager.set_downscale_factor()
    # Start the video manager and perform the scene detection.
        video_manager.start()     
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            stop_frame = scene[1].get_frames()
            scenes.append([start_frame,stop_frame])
        print("Scenes elements: ", scenes)
        return(scenes)

    def divide_movie_into_frames(self, movie_in_path, movie_out_folder, save_frames=False):
        """
        Input: Diretory path of the movies and directory of where to save the frames.
        Output: frames (and if save_frames=True it also saves in the destination)
        """
        frames = []
        cap = cv2.VideoCapture(movie_in_path)
        ret, frame = cap.read()
        frames.append(frame)
        num = 0
        if save_frames:
            cv2.imwrite(os.path.join(movie_out_folder, f'frame{num:04}.jpg'), frame)
        while cap.isOpened() and ret:
            num = num + 1
            ret, frame = cap.read()
            frames.append(frame)
            if frame is not None and save_frames:
                cv2.imwrite(os.path.join(movie_out_folder,
                           f'frame{num:04}.jpg'), frame)
        return frames

    def store_frames_to_s3(self, movie_id, frames_folder, video_file):
        bucket_name = "nebula-frames"
        folder_name = movie_id
        self.s3.put_object(Bucket=bucket_name, Key=(folder_name+'/'))
        print(frames_folder)
        if not os.path.exists(frames_folder):
            os.mkdir(frames_folder)
        else:
            for f in os.listdir(frames_folder):
                if os.path.isfile(os.path.join(frames_folder, f)):
                    os.remove(os.path.join(frames_folder, f))
        num_frames = self.divide_movie_into_frames(video_file, frames_folder)
        # SAVE TO REDIS - TBD
        if num_frames > 0:
            for k in range(num_frames):
                img_name = os.path.join(
                    frames_folder, f'frame{k:04}.jpg')
                self.s3.upload_file(img_name, bucket_name, folder_name +
                            '/' + f'frame{k:04}.jpg')

    def get_video_metadata(self, video_file):
        """
        Input: video
        Output: width, height and fps of the video.
        """
        cap = cv2.VideoCapture(video_file)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return ({'width': W, 'height': H, 'fps': fps})

    def detect_scenes(self, video_file):
        """
        Input: video
        Output: Scenes of the video.
        """
        scenes = []
        
        video_manager = VideoManager([video_file])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=70.0))
    # Improve processing speed by downscaling before processing.
        video_manager.set_downscale_factor()
    # Start the video manager and perform the scene detection.
        video_manager.start()     
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            stop_frame = scene[1].get_frames()
            scenes.append([start_frame,stop_frame])
        print("Scenes: ", scenes)
        return(scenes)
    
    def detect_mdf(self, video_file, scene_elements, method='3frames'):
        """
        :param video_file:
        :param scene_elements: list of list, [[2, 4], [33, 55]] - start and end frame of each scene
        :param method: '3frames' - choose three good frames, 'clip_segment' - use clip segmentation to choose 3 MDFs
        :return:
        """
        print("Detecting MDFs..")
        if method == '3frames':
            mdfs = []
            for scene_element in scene_elements:
                scene_mdfs = []
                start_frame = scene_element[0]
                stop_frame = scene_element[1]
                frame_qual = self.video_eval.mark_blurred_frames(video_file, start_frame, stop_frame, 100)
                # Ignore the blurred images
                frame_qual[0:3] = 0
                frame_qual[-3:] = 0
                middle_frame = start_frame + ((stop_frame - start_frame) // 2)
                good_frames = np.where(frame_qual > 0)[0]
                if len(good_frames > 5):
                    stop_frame = start_frame + good_frames[-1]
                    start_frame = start_frame + good_frames[0]
                    middle_frame = start_frame + good_frames[len(good_frames) // 2]
                    scene_mdfs.append(int(start_frame))
                    scene_mdfs.append(int(middle_frame))
                    scene_mdfs.append(int(stop_frame))
                else:
                    scene_mdfs.append(int(start_frame) + 2)
                    scene_mdfs.append(int(middle_frame))
                    scene_mdfs.append(int(stop_frame) - 2)
                mdfs.append(scene_mdfs)
            # go over all frames and compute the average derivative
        elif method == 'clip_segment':
            pass
        else:
            raise Exception('Unsupported method')
        # mdfs = []
        # for scene_element in scene_elements:
        #     scene_mdfs = []
        #     start_frame = scene_element[0]
        #     stop_frame = scene_element[1]
        #     scene_mdfs.append(start_frame + 2)
        #     middle_frame = start_frame + ((stop_frame- start_frame) // 2)
        #     scene_mdfs.append(middle_frame)
        #     scene_mdfs.append(stop_frame - 2)
        #     mdfs.append(scene_mdfs)
        return(mdfs)


    def save_mdfs_to_jpg(self, full_path, mdfs, save_path="/dataset/lsmdc/mdfs_of_20_clips/"):
        vidcap = cv2.VideoCapture(full_path)
        success, image = vidcap.read()
        count = 0
        counted_mdfs = 0
        if ".mp4" in full_path:
            full_path_modified = full_path.split("/")[-1].replace(".mp4","")
        if ".avi" in full_path:
            full_path_modified = full_path.split("/")[-1].replace(".avi","")  
        while success:
            success, image = vidcap.read()
            if count in mdfs:
                cv2.imwrite(f"{save_path}{full_path_modified}__{count}.jpg", image)     # save frame as JPEG file      
                print('Read a new frame: ', success)
                counted_mdfs += 1
            if counted_mdfs == len(mdfs):
                print("Done")
                return
            count += 1
            # exit of infinite loop
            if count > 9999:
                return

    def generate_captions_on_mdfs(self, images, mdfs_path, prefix_link = "http://ec2-18-159-140-240.eu-central-1.compute.amazonaws.com:7000/static/dataset1/mdfs/"):
        from clipcap import ClipCap
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        model, preprocess = clip.load("ViT-B/32", device=device)
        clip_cap = ClipCap()
        mdfs_path = "/dataset/lsmdc/mdfs_of_20_clips/"
        images = [os.path.join(mdfs_path, image) for image in os.listdir(mdfs_path)]
        output_dict = {}
        for img_path in images:
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
            embedding = model.encode_image(img)
            output = clip_cap.generate_text(embedding, use_beam_search=False)
            image_name = img_path.split("/")[-1]
            print("Input:")
            print(image_name)
            link_path = os.path.join(prefix_link, image_name)
            print(f"Link: {link_path}")
            print("Output")


    def convert_avi_to_mp4(self, avi_file_path, output_name):
        os.system("ffmpeg -y -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(
            input=avi_file_path, output=output_name))
        return True

    # file_name - file name without path
    # movie_name - if available, if no - same file name
    # tags - if available
    # full_path - path to file, for video processing
    # url - url to S3
    # last_frame - number of frames in movie
    # metadata - available metadata - fps, resolution....
    def insert_movie(self, full_path, url, source, db_name="ilan_test"):
        print("Changing database to {}.".format(db_name))
        self.nre.change_db(db_name)
        query = 'UPSERT { File: @File} INSERT  \
            { File: @File, \
                url_path: @url, \
                scenes: @scenes, scene_elements: @scene_elements, mdfs: @mdfs, meta: @metadata,\
                     scenes: @scenes, scene_elements: @scene_elements, mdfs: @mdfs, updates: 1, \
                         source: "lsmdc_concatenated"\
                    } UPDATE \
                { updates: OLD.updates + 1, \
                File: @File, url_path: @url, \
                scenes: @scenes, scene_elements: @scene_elements, mdfs: @mdfs, meta: @metadata, source: @source, \
                     scenes: @scenes, scene_elements: @scene_elements, mdfs: @mdfs \
                } IN Movies_v2 \
                    RETURN { doc: NEW, type: OLD ? \'update\' : \'insert\' }'
        scene_elements = self.detect_scene_elements(full_path)
        mdfs = self.detect_mdf(full_path, scene_elements)
        mdf_outputs = []
        for mdf in mdfs:
            self.save_mdfs_to_jpg(full_path, mdf, save_path="/dataset/lsmdc/mdfs_of_20_clips/")
            generate_captions_on_mdfs()
        scenes = self.detect_scenes(full_path)
        metadata = self.get_video_metadata(full_path)
        bind_vars = {
                        'File': full_path,
                        'url': url,
                        'scenes': scenes,
                        'scene_elements': scene_elements,
                        'mdfs': mdfs,
                        'metadata': metadata,
                        'source': source
                        }
        print(bind_vars)
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        for doc in cursor:
            doc=doc
        return(doc['doc']['_id'])

    def get_movie_id_from_url(self, url):
        movie_id = url.split("/")[-1].replace(".","_")
        if "_mp4" in movie_id:
            return movie_id.replace("_mp4", "")
        elif "_avi" in movie_id:
            return movie_id.replace("_avi", "")
        else:
            raise Exception("Unknown video format. Please use mp4/avi")

    def insert_movie_sprint_2(self, full_path, url, concat_url, source, db_name="ilan_test"):
        print("Changing database to {}.".format(db_name))
        self.nre.change_db(db_name)
        query = 'UPSERT { File: @File} INSERT  \
        { File: @File, \
            url_path: @url, \
            scenes: @scenes, scene_elements: @scene_elements, mdfs: @mdfs, meta: @metadata,\
                    scenes: @scenes, scene_elements: @scene_elements, mdfs: @mdfs, updates: 1, \
                        source: "lsmdc_concatenated"\
                } UPDATE \
            { updates: OLD.updates + 1, \
            File: @File, url_path: @url, \
            scenes: @scenes, scene_elements: @scene_elements, mdfs: @mdfs, meta: @metadata, source: @source, \
                    scenes: @scenes, scene_elements: @scene_elements, mdfs: @mdfs \
            } IN sprint2_lsmdc \
                RETURN { doc: NEW, type: OLD ? \'update\' : \'insert\' }'
        scene_element = self.detect_scenes(full_path)
        mdfs = self.detect_mdf(full_path, scene_element)
        movie_id = self.get_movie_id_from_url(url)
        bind_vars = {
                        'File': full_path,
                        'movie_id': movie_id,
                        'clip_url': url,
                        'concatenated_url': concat_url,
                        'mdfs': mdfs,
                        'source': source
                        }
        print(bind_vars)
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        for doc in cursor:
            doc=doc
        return(doc['doc']['_id'])

    def new_movies_batch_processing(self, upload_dir, storage_dir, dataset):
        _files = glob.glob(upload_dir + '/*')
        movies = []
        for _file in _files:
            file_name = basename(_file)
            movie_mame = file_name.split(".")[0]            
            print("New movie found")
            print(file_name)
            print(movie_mame)
            self.convert_avi_to_mp4(_file, storage_dir + "/" + movie_mame)
            movie_id = self.insert_movie(file_name, movie_mame, ["lsmdc", "pegasus", "visual genome"],
                              storage_dir + "/" + movie_mame + ".mp4", "static/" + dataset + "/" + movie_mame + ".mp4", 300, "created")
            movies.append((_file, movie_id))
        return(movies)
            
    def divide_movie_by_timestamp(self, movie_in_path, t_start, t_end, dim=(224, 224)):

        cap = cv2.VideoCapture(movie_in_path)
        ret, frame = cap.read()
        num = 0
        frames = []
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        movie_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_start = t_start * fps
        frame_end = t_end * fps
        if frame_start > movie_frames_count or frame_end > movie_frames_count:
            print(f'Error, the provided t_start {t_start} or t_end {t_end} \
                multiplied by fps {fps} is higher than number of frames {movie_frames_count}')
            return -1
        while cap.isOpened() and ret:
            num = num + 1
            ret, frame = cap.read()
            if frame is not None and num >= frame_start and num <= frame_end:
                resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                frames.append(resized_frame)
        return frames

    def preprocess_dataset(self, dataset_path, db_type="COMET"):
        """
        Preprocess LSMDC & VSCOMET datasets to dictionaries
        """    
        new_db = []
        if db_type == "COMET":
            test_json = open(dataset_path)
            data = json.load(test_json)
            for idx, key in enumerate(data):
                if "lsmdc" in key['img_fn']:
                    new_key = self.get_frame_dict(key)
                    new_db.append(new_key)

        elif db_type == "LSMDC":
            with open(dataset_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    key = row[0].split()[0]
                    txt = ''
                    if len(row) > 1:
                        txt = row[1]
                    new_db.append({'clip_id': key, 'txt': txt})
            return new_db
        else: #db_type is LSMDC
            print("Unknown DB-TYPE")

        return new_db
    
    def time2secs(self, ts):
        hh, mm, ss, ms = ts.split(".")
        return int(hh)*3600 + int(mm)*60 + int(ss) + (float(ms) / 1000000)
    
    def get_gt_clips(self, path):
        output = []
        with open(path) as fp:
            lines = fp.readlines()
            cnt = 1
            for line in lines:
                output.append(line.rstrip("\n"))
        return output

    def divide_to_parts(self, dataset, clip_time=20, only_gt_clips=False):
        """
         Divide movies to 20 second clips. (With holes)
         If `only_gt_clips=True`, then we take the 44 ground-truth clips only.
        """
        new_dataset = [[]]
        time_ctr = clip_time
        new_dataset_idx = 0
        start_merging = False
        counter = 0
        if only_gt_clips:
            # clip_names = self.get_playground_movies()
            clips_gt_list_path = "./utils/44_clips_list.txt"
            clip_names = self.get_gt_clips(clips_gt_list_path)
            start_merging = False
        for i in range(1, len(dataset)):
            cur_clip = dataset[i]
            clip_id = cur_clip['clip_id']
            # If this is true, we'll process only 44 clips instead of all the clips.
            if only_gt_clips and not start_merging:
                if clip_id.replace(".","_") not in clip_names:
                    continue
                else:
                    start_merging = True
                    time_ctr = clip_time
                    counter += 1
                    print(clip_id.replace(".","_"))
            ts_start, ts_end = clip_id.split("_")[-1].split("-")
            ts_start, ts_end = self.time2secs(ts_start), self.time2secs(ts_end)
            clip_length = ts_end - ts_start

            # clip is over `clip_time=20seconds`, add it alone
            if clip_length > clip_time:
                new_dataset[new_dataset_idx].append(cur_clip)
                time_ctr = clip_time
                new_dataset.append([])
                new_dataset_idx += 1
                if start_merging:
                    start_merging = False
                continue

            # clip is less than 20 seconds, we add it to new list that can contain
            # more clips in that list.
            if time_ctr - clip_length > 0:
                new_dataset[new_dataset_idx].append(cur_clip)
                time_ctr -= clip_length
                continue
            
            # If we add a clip that exceeds `time_ctr` of current list of clips
            # we add it to a new list.
            if time_ctr - clip_length <= 0:
                if start_merging:
                    start_merging = False
                    time_ctr = clip_time
                    new_dataset_idx += 1
                    new_dataset.append([])
                    continue
                new_dataset.append([])
                new_dataset_idx += 1
                new_dataset[new_dataset_idx].append(cur_clip)
                time_ctr = clip_time
                # If the added clip has bigger value then clip_time, we create new level
                if clip_length > clip_time:
                    new_dataset.append([])
                    new_dataset_idx += 1
                # just substract clip_length in the new level.
                else:
                    time_ctr -= clip_length
                continue
        print(counter)
        # Sometimes the last element in the list is empty, so we remove i
        if not new_dataset[-1]:
            return new_dataset[:-1]
        return new_dataset

    def remove_holes_in_concatenated_clips(self, dataset, delta_error=1):
        """
         There are clips that we concatenate that are not sequential.
         For e.g. if clip starts at 00:01:00 and ends at 00:01:20, the next clip,
         should start at approximately 00:01:20 as well, if it starts at 00:02:00 it's bad,
         so we will skip it.
         `delta_error`: maximum time allowed (in seconds) between end of clip and start of new one.
        """
        new_dataset = []
        for data in dataset:
            time_stamps = []
            for i in range(len(data)):
                 ts_start, ts_end = data[i]['clip_id'].split("_")[-1].split("-")
                 ts_start, ts_end = self.time2secs(ts_start), self.time2secs(ts_end)
                 time_stamps.append([ts_start, ts_end])

            if len(time_stamps) == 1:
                new_dataset.append(data)
                ts_start, ts_end = data[0]['clip_id'].split("_")[-1].split("-")
                ts_start, ts_end = self.time2secs(ts_start), self.time2secs(ts_end)
                length = ts_end - ts_start
                # print("-"*30)
                # print(f"Movie Name: {data[0]['clip_id']}")
                # print(f"Length: {length}")
            else:
                length = 0
                for i in range(len(time_stamps) - 1):
                    delta = time_stamps[i + 1][0] - time_stamps[i][1]
                    if delta > delta_error:
                        position = i
                        break
                if position != 0:
                    new_dataset.append(data[:position + 1])
                
                for i in range(len(data[:position + 1])):
                     ts_start, ts_end = data[i]['clip_id'].split("_")[-1].split("-")
                     ts_start, ts_end = self.time2secs(ts_start), self.time2secs(ts_end)
                     length += ts_end - ts_start
                if position != 0:
                    print("-"*30)
                    print(f"Movie Name: {data[0]['clip_id']}")
                    print(f"Length: {length}")
        print(f"Length of dataset: {len(new_dataset)}")

        # Sort the database by the length in a descending order, therefore we will get the longest videos w/o holes.
        new_dataset.sort(key=len, reverse=True)
        return new_dataset

    def split_modified(self, strng, sep, pos):
        """
        There're movies like Spider-man...timestamp-timestamp..
        I want to split only by the last "-"
        """
        strng = strng.split(sep)
        return sep.join(strng[:pos]), sep.join(strng[pos:])

    def check_if_merged_mp4_exists(self, path, movie_names):
        """
        For debugging purposes. Skipping merged .mp4 files.
        """
        from natsort import natsorted
        if not movie_names:
            return
        files = natsorted(movie_names)
        # Beginning timestamp of first clip, and end timestamp of last clip.
        prefix_str = self.split_modified(files[0].split("/")[-1], "-", -1)[0]
        unique_clip_name = prefix_str + "-" + \
                            files[-1].split("/")[-1].split("-")[-1].replace(".avi", ".mp4")
        if unique_clip_name in os.listdir(path):
            print(f"Merged clip {unique_clip_name} already exists! Skipping...")
            return True
        return False

    def process_specific_avi_to_mp4(self, upload_dir, storage_dir, clips_names = ['']):
        """
        Convert LSMDC .avi clips to .mp4 clips.
        If `clips_names` is [''] we return.
        Otherwise, we iterate only over the `clips_names` list.
        """
        print("Processing .avi files.")
        if clips_names != ['']:
            print(f"We iterate only over {clips_names}")
        else:
            print("Error! `clip_names` is empty.")
            return
        # Check if .mp4 and doesn't exist. Otherwise, we proceed and convert .avi to .mp4
        for clip in clips_names:
            if clip.replace(".avi",".mp4") in os.listdir(storage_dir):
                print("Skipping. Found at least one clip in folder.")
                return
        # check if merged .mp4 doesn't exist.
        if self.check_if_merged_mp4_exists(storage_dir, clips_names):
            print("Skipping. Found the merged .mp4 in folder.")
            return
            
        _dirs = glob.glob(upload_dir + '/*')
        movies_output = []
        counter = 0
        length_clips = len(clips_names)
        for _dir in _dirs:
            _files = glob.glob(_dir + '/*')
            for _file in _files:
                file_name = basename(_file)
                movie_name = file_name
                if movie_name in clips_names:
                    self.convert_avi_to_mp4(_file , storage_dir + "/" + movie_name.replace(".avi",""))
                    movies_output.append(storage_dir + "/" + movie_name.replace(".avi",".mp4"))
        return movies_output
    
    def merge_multiple_mp4_to_one(self, path, movie_names):
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        import os
        from natsort import natsorted
        # check if merged .mp4 doesn't exist.
        if not movie_names:
            print("Skipping. no `movie_names`.")
            return
        if len(movie_names) == 1:
            print(f"Skipping. We have only one {movie_names} .mp4 file, so we don't need to merge it.")
            return
        print("Merging mp4 clips to one mp4 clip..")
        L =[]
        _files = [os.path.join(path, f) for f in os.listdir(path) if f.replace(".mp4",".avi") in movie_names]
        files = natsorted(_files)
        # Beginning timestamp of first clip, and end timestamp of last clip.
        prefix_str = self.split_modified(files[0].split("/")[-1], "-", -1)[0]
        unique_clip_name = prefix_str + "-" + \
                            files[-1].split("/")[-1].split("-")[-1]
        if unique_clip_name in path:
            print(f"Merged clip {unique_clip_name} already exists! Skipping...")
        for file in files:
            if file != unique_clip_name:
                if os.path.splitext(file)[1] == '.mp4':
                    filePath = os.path.join(path, file)
                    video = VideoFileClip(filePath)
                    L.append(video)

        final_clip = concatenate_videoclips(L)
        final_clip.to_videofile(os.path.join(path,unique_clip_name), fps=24, remove_temp=False)
        print(f"Done merge the clips! Merged file: {os.path.join(path,unique_clip_name)}")
        prefix_link = "http://ec2-18-159-140-240.eu-central-1.compute.amazonaws.com:7000/static/"
        print(f"Can be access in: {os.path.join(os.path.join(prefix_link, path) ,unique_clip_name)}")


    def concat_mp4(self, clips_dict, base_path = "/dataset1/", dest_path = "/dataset1/concatenated_mp4_50"):
        """
        Merging multiple mp4 files into one mp4 files.
        1. Take the relevant .avi files.
        2. Convert them to .mp4 files and save them.
        3. Merge these files to one output mp4 file.
        """
        for ctr, clips in enumerate(clips_dict):
            # 1. Take the relevant .avi files.
            clip_names = [clip['clip_id']+".avi" for clip in clips]
            # 2. Convert them to .mp4 files and save them.
            movies_output = self.process_specific_avi_to_mp4(base_path, dest_path, clip_names)
            # 3. Merge these files to one output mp4 file.
            if movies_output:
                self.merge_multiple_mp4_to_one(dest_path, clip_names)
            print(f"Finished {ctr+1}/{len(clips_dict)} clips.")
            
            print("Removing all the merged .mp4 files and moving to next clip.")
            clips_to_del = [clip['clip_id']+".mp4" for clip in clips]
            if len(clips_to_del) > 1: # If it's one clip, it's also the .merged one so we skip it
                for clip_to_del in clips_to_del:
                    ## If file exists, delete it. 
                    clip_name = os.path.join(dest_path, clip_to_del)
                    if os.path.isfile(clip_name) and ".mp4" in clip_name:
                        os.remove(clip_name)
                    else:
                        print("Error: %s file not found" % clip_name)

    def create_concat_lsmdc_movies(self, lsmdc_dir, num_of_movies=55, base_path = "/dataset1/", dest_path = "/dataset1/concatenated_mp4_20"):

        # Paths to LSMDC .JSONs
        lsmdc_train_path = os.path.join(lsmdc_dir, "LSMDC16_annos_training.csv")
        lsmdc_val_path = os.path.join(lsmdc_dir, "LSMDC16_annos_val.csv")
        lsmdc_test_path = os.path.join(lsmdc_dir, "LSMDC16_annos_test.csv")

        # Process LSMDC annotation files
        lsmdc_train_dict = self.preprocess_dataset(lsmdc_train_path, "LSMDC")
        lsmdc_val_dict = self.preprocess_dataset(lsmdc_val_path, "LSMDC")
        lsmdc_test_dict = self.preprocess_dataset(lsmdc_test_path, "LSMDC")

        # Divide LSMDC annotation files with holes.
        lsmdc_train_dict_merged = self.divide_to_parts(lsmdc_train_dict, clip_time=20, only_gt_clips=False)
        # lsmdc_val_dict_merged = self.divide_to_parts(lsmdc_val_dict, clip_time=20, only_gt_clips=False)
        # lsmdc_test_dict_merged = self.divide_to_parts(lsmdc_test_dict, clip_time=20, only_gt_clips=False)

        # Divide LSMDC annotation files without holes.
        lsmdc_train_dict_merged = self.remove_holes_in_concatenated_clips(lsmdc_train_dict_merged)
        # lsmdc_val_dict_merged = self.remove_holes_in_concatenated_clips(lsmdc_val_dict_merged)
        # lsmdc_test_dict_merged = self.remove_holes_in_concatenated_clips(lsmdc_test_dict_merged)

        self.concat_mp4(lsmdc_train_dict_merged[:num_of_movies], base_path, dest_path)


def main():
    scene_detector = NEBULA_SCENE_DETECTOR()

    concate_movies = False
    # We concatenate 20 movies of LSMDC
    if concate_movies:
        lsmdc_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset1/lsmdc/")
        scene_detector.create_concat_lsmdc_movies(lsmdc_dir=lsmdc_path, num_of_movies=20, base_path = "/dataset1/", dest_path = "/dataset1/concatenated_mp4_20")

    # path = '/dataset/development/1010_TITANIC_00_41_32_072-00_41_40_196.mp4'
    # scene_detector.divide_movie_by_timestamp(path, 3, 4)

    # scene_detector.new_movies_batch_processing()

    #scene_detector.init_new_db("nebula_datadriven")

    # INSERT concatenated movies to database
    mp4_path = "/dataset1/concatenated_mp4_55/"
    _files = [f for f in os.listdir(mp4_path) if f.endswith(".mp4")]
    prefix_link = "http://ec2-18-159-140-240.eu-central-1.compute.amazonaws.com:7000/static"
    source = "lsmdc_concatenated"
    #Example usage
    for _file in _files:
        file_name = basename(_file)
        movie_name = file_name.split(".mp4")[0]
        url_link = os.path.join(prefix_link+mp4_path, movie_name + ".mp4")
        full_path = os.path.join(mp4_path, movie_name + ".mp4")
        scene_detector.insert_movie(full_path, url_link, source)
if __name__ == "__main__":
    main()


    
