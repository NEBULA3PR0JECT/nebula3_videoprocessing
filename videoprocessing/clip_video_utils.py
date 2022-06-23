import torch
import clip
import numpy as np
import cv2 as cv
from PIL import Image
from copy import deepcopy
from sklearn.cluster import MeanShift


class ClipVideoUtils:
    """
    The class provides a number of utils for CLIP based video processing
    """
    def __init__(self, model_name='RN50x4'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model_res = 640
        if model_name == 'ViT-B/32':
            self.model_res = 512
        if model_name == 'ViT-L/14':
            self.model_res = 768

    def is_sharp(self, color_img, blur_threshold=100):
        """
        :param color_img: OpenCV,  bgr format
        :param blur_threshold:
        :return: 1 if sharp
        """
        gray = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
        dst = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)
        fm = cv.Laplacian(dst, cv.CV_64F).var()
        if fm > blur_threshold:
            return 1
        else:
            return 0

    def segment_embedding_array(self, embedding_array, dist_th) -> list:
        """
        option 1 - greedy, going from left to right, and stopping when the variance of the embeddings becomes too large
        or when the largest distance from the tested embedding to any embedding in the test is too large
        :return:
        """
        boundaries = []
        boundaries.append(0)
        for k in range(embedding_array.shape[0]):
            pass
            # compute distance from the current point to all other points
            max_dist = 10
            for m in range(boundaries[-1], k):
                d = np.sum(embedding_array[k]*embedding_array[m])
                # d = np.linalg.norm(embedding_array[k] - embedding_array[m])
                if d < max_dist:
                    max_dist = d
            if max_dist < dist_th:
                boundaries.append(k)
        boundaries.append(embedding_array.shape[0] - 1)
        # Remove very short shots
        # Create list of tuples
        ret_boundaries = []
        good_frame_len = 0
        for k in range(1, len(boundaries)):
            if boundaries[k] - boundaries[k - 1] > 9:
                ret_boundaries.append((boundaries[k - 1], boundaries[k]))
                good_frame_len = good_frame_len + boundaries[k] - boundaries[k - 1]

        return ret_boundaries, good_frame_len

    def get_embedding_diffs(self, movie_name, start_time=-1, end_time=10000000, frame_or_time=1):
        """
        Given the movie name and the start / end time, return the CLIP embedding differeneces
        :param movie_name:
        :param start_time:
        :param end_time:
        :param frame_or_time if 0, we treat input as frame, if 1 as time
        :return:
        """
        cap = cv.VideoCapture(movie_name)
        init = 0
        old_embeddings = None
        diff_list = []
        ret, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        embedding_array = np.zeros((0, self.model_res))
        frame_num = 0
        fps = cap.get(cv.CAP_PROP_FPS)
        if frame_or_time == 0:
            fps = 1
        while cap.isOpened() and ret:
            if (frame_num >= start_time * fps) and (frame_num < end_time * fps):
                with torch.no_grad():
                    img = self.preprocess(Image.fromarray(frame)).unsqueeze(0).to(self.device)
                    embeddings = self.model.encode_image(img)
                    embeddings = embeddings / np.linalg.norm(embeddings)
                    embedding_array = np.append(embedding_array, embeddings, axis=0)
                    if init == 0:
                        init = 1
                    else:
                        diff = embeddings - old_embeddings
                        diff_list.append((diff * diff).sum())
                    old_embeddings = embeddings
            frame_num = frame_num + 1
            ret, frame = cap.read()
            if frame is not None:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        return embedding_array, fps

    def choose_frames_with_meanshift(self, movie_name, start_frame=-1, end_frame=1000000000, bandwidth=0.45):
        """
        Choose the best frame using meanshift algorithm on CLIP embedding
        :param movie_name: the name of the movie
        :param start_frame: we look only at the section between the start_frame and the end_frame
        :param end_frame:
        :param bandwidth: the meanshift parameter. The smaller the value, the more clusters we'll have. It's possible
            not to pass the parameter aannd let the aalgorithm to detect the optimal value
        :return: a list of frames and a list of
        """
        ret_frame_list = []
        ret_image_list = []
        embedding_array, fps = self.get_embedding_diffs(movie_name, start_frame, end_frame, 0)
        clustering = MeanShift(bandwidth=bandwidth).fit(embedding_array)
        num_clusters = clustering.cluster_centers_.shape[0]
        ret_cluster_size = []
        for k in range(num_clusters):
            cluster_size = clustering.labels_.count(k)
            ret_cluster_size.append(cluster_size)
            # choose the frame closest to the cluster
            dist_emb = embedding_array - np.matmul(np.ones((embedding_array.shape[0], 1)),
                                        np.reshape(clustering.cluster_centers_[k, :], (1, embedding_array.shape[1])))
            ret = np.argmin(np.sum(dist_emb * dist_emb, 1)) + start_frame
            ret_img = self.get_specific_frames(movie_name, [[ret]])

            ret_frame_list.append(ret)
            ret_image_list.append(ret_img)

        return ret_frame_list, ret_image_list, ret_cluster_size

    def choose_best_frame(self, movie_name, start_frame, end_frame):
        """
        The function choose the best frame from a given leg
        :param movie_name:
        :param start_frame:
        :param end_frame:
        :return:
        """

        embedding_array, fps = self.get_embedding_diffs(movie_name)
        rel_embeddings = deepcopy(embedding_array[start_frame:end_frame])
        mean_emb = np.mean(rel_embeddings, 0)
        dist_emb = rel_embeddings - np.matmul(np.ones((end_frame-start_frame, 1)),
                                              np.reshape(mean_emb, (1, embedding_array.shape[1])))
        # Remove 20% outliers
        total_dist = np.sum(dist_emb * dist_emb, 1)
        indexes = np.argsort(total_dist)
        # ret = np.argmin(np.sum(dist_emb * dist_emb, 1))
        inlier_indexes = indexes[0:np.max([int(len(total_dist) * 0.8), 2])]
        mean_emb = np.mean(rel_embeddings[inlier_indexes, :], 0)
        dist_emb = rel_embeddings[inlier_indexes, :] - \
                   np.matmul(np.ones((len(inlier_indexes), 1)), np.reshape(mean_emb, (1, embedding_array.shape[1])))
        total_dist = np.sum(dist_emb * dist_emb, 1)
        ret = inlier_indexes[np.argmin(np.sum(dist_emb * dist_emb, 1))] + start_frame

        ret_img = self.get_specific_frames(movie_name, [[ret]])

        return ret, ret_img


    def get_scene_elements_from_embeddings(self, movie_name, th = 0.8, start_time=-1, end_time=10000000) -> list:
        """
        Given the movie, we can divide it into scene elements
        :param movie_name:
        :param start_time:
        :param end_time:
        :return:
        """
        embedding_array, fps = self.get_embedding_diffs(movie_name, start_time, end_time)
        ret_boundaries, good_frame_len = self.segment_embedding_array(embedding_array, th)
        return ret_boundaries, good_frame_len

    def get_adaptive_movie_threshold(self, movie_name, start_frame, end_frame):
        """
        :param movie_name: - path of the movie
        :param start_frame: we ignore frames before start (not including)
        :param end_frame: we ignore frames after end
        :return: Given a scene element (usually described by start and end frame of the movie), return the
            optimal threshold that will separate between blurry and non blurry frames
        """
        cap = cv.VideoCapture(movie_name)
        ret, frame = cap.read()
        frame_num = 0
        fft_center_list = []
        fft_border_list = []
        values = []
        max_v = []
        len_v = []
        while cap.isOpened() and ret:
            if frame_num >= start_frame and frame_num <= end_frame:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                dst = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)
                fm = cv.Laplacian(dst, cv.CV_64F).var()
                values.append(fm)
                fft = np.fft.fft2(gray)
                fftShift = np.fft.fftshift(fft)
                magnitude = np.log(np.abs(fftShift))
                (h, w) = gray.shape
                (cX, cY) = (int(w / 2.0), int(h / 2.0))
                size_window = 50
                fft_center_list.append(
                    np.mean(magnitude[cY - size_window:cY + size_window, cX - size_window:cX + size_window]))
                magnitude[cY - size_window:cY + size_window, cX - size_window:cX + size_window] = 0
                fft_border_list.append(np.mean(magnitude))
            frame_num = frame_num + 1
            # prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            ret, frame = cap.read()

        blur_threshold = np.median(values) / 1.25

        return blur_threshold, values, fft_center_list, fft_border_list

    def get_specific_frames(self, movie_in_path, frames_list):
        """
        Return the specific frames in a movie
        :param movie_in_path:
        :param frames_list: list of lists
        :return:
        """
        cap = cv.VideoCapture(movie_in_path)
        ret, frame = cap.read()
        num = 0
        ret_list = []
        while cap.isOpened() and ret:
            for frames in frames_list:
                if num in frames:
                    ret_list.append(frame)
                    break
            num = num + 1
            ret, frame = cap.read()
        return ret_list

    def mark_blurred_frames(self, movie_name, start_frame, end_frame, blur_threshold=-1):
        """
        :param movie_name: the full path
        :param start_frame: we are testing the movie only between frames start_frame and end_frame
        :param end_frame:
        :param blur_threshold if -1, we compute adaptively, otherwise we use the threshold
        :return: return the list of 1s and 0s and from start_frame (including) until end_frame (including)
        0 == BAD
        """
        if blur_threshold < 0:
            blur_threshold, values, len_v, max_v = self.get_adaptive_movie_threshold(movie_name, start_frame, end_frame)
            ret_list = []
            for val in values:
                if val > blur_threshold:
                    ret_list.append(1)
                else:
                    ret_list.append(0)
        else:
            cap = cv.VideoCapture(movie_name)
            ret, frame = cap.read()
            frame_num = 0
            ret_list = []
            values = []
            while cap.isOpened() and ret:
                if frame_num >= start_frame and frame_num <= end_frame:
                    ret_list.append(self.is_sharp(frame, blur_threshold))
                frame_num = frame_num + 1
                ret, frame = cap.read()
                pass

        return np.array(ret_list)