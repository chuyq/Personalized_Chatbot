import glob
import os
import json
import pickle
import random
import time
import itertools
import pandas as pd
import json

import torch.nn.functional as F

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import torch
from torch.utils.data import Dataset
import webdataset as wds
import cv2

from minigpt4.datasets.datasets.base_dataset import BaseDataset

class FeatureFaceDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):

        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor


        self.caption_instruction_pool = [
            "Analyze the speaker's facial expressions, tone of voice, and speaking style in the video to infer their personality traits.",
            "Describe the speaker's expressions, tone, and communication style. What personality tendencies do they exhibit?",
            "Identify patterns in the speaker’s tone, pace, and body language. What do these suggest about their personality type?",
            "Assess the video for signs of introversion/extroversion, thinking/feeling, or other personality traits based on expression and tone.",
            "Observe the speaker’s communication style—does it reflect structured, logical thinking or spontaneous, intuitive insights?",
            "Break down the speaker’s expressions and vocal tone in the video. How do these features reflect their personality type?",
        ]

        self.emotion_instruction_pool = [
            "Please determine which personality label in the video represents: Te, Ti, Fe, Fi, Se, Si, Ne, Ni",
        ]

        self.reason_instruction_pool = [
        "You are a  therapist. Provide emotional support to the user, tailoring your response to their specific personality type. Be  thoughtful, professional, and efficient responses that specifically address the user's issues.",
        "You are a therapist. Generate a response to help the user alleviate their distress, note that tailoring your response to their specific personality type.",
        ]


        self.task_pool = [
           "personality",
           "reason"
        ]


        print("ann_path: ", ann_path)
        self.ann_path = ann_path
        self.file_path = os.path.dirname(ann_path)
        self.tmp = [x.strip().split(' ') for x in open(ann_path)]
        print(('video number:%d' % (len(self.tmp))))

        personas = ['Te', 'Ti', 'Fe', 'Fi', 'Se', 'Si', 'Ne', 'Ni']

        self.per2idx, self.idx2per = {}, {}
        for ii, per in enumerate(personas): self.per2idx[per] = ii
        for ii, per in enumerate(personas): self.idx2per[ii] = per

        json_file_path = "xxxxxxxxxxxxxxxxxxxxxxxxxxx" 
        with open(json_file_path, 'r') as json_file:
            # self.MERR_coarse_grained_dict = json.load(json_file)
            self.MPER_coarse_grained_dict = json.load(json_file)


        reason_json_file_path = "xxxxxxxxxxxxxxxxxxxxxxxx"
        with open(reason_json_file_path, 'r') as json_file:
            self.MPER_fine_grained_dict = json.load(json_file)

        self.character_lines = pd.read_csv('xxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, index):
        t = self.tmp[index]
        video_name = t[0]

        video_path = os.path.join(self.vis_root, video_name + ".mp4")
        if os.path.exists(video_path):
            image = self.extract_frame(video_path)
        else:
            video_path = os.path.join(self.vis_root, video_name + ".avi")
            image = self.extract_frame(video_path)

        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)


        # image_file = '{}.jpg'.format(video_name)
        # image_path = os.path.join(self.vis_root, image_file)
        # image = Image.open(image_path).convert("RGB")
        # image = self.vis_processor(image)


        FaceMAE_feats, VideoMAE_feats, Audio_feats = self.get(video_name)
        if len(VideoMAE_feats.shape) == 1:
            VideoMAE_feats = VideoMAE_feats.unsqueeze(0)
        if len(Audio_feats.shape) == 1:
            Audio_feats = Audio_feats.unsqueeze(0)
        if len(FaceMAE_feats.shape) == 1:
            FaceMAE_feats = FaceMAE_feats.unsqueeze(0)
        video_features = torch.cat((FaceMAE_feats, VideoMAE_feats, Audio_feats), dim=0)


        # random task
        task = random.choice(self.task_pool)
        if task == "personality":
            caption = t[2] # llama2 putput only emotion class
            caption = self.text_processor(caption)
            instruction_pool = self.emotion_instruction_pool
        elif task == "reason":
            caption = self.MPER_coarse_grained_dict[video_name]['sys']

            caption = self.text_processor(caption)
            instruction_pool = self.reason_instruction_pool

        elif task == "reason_v2":
            caption = self.MPER_fine_grained_dict[video_name]['reason_generate']

            # caption = "" # for test reasoning

            if caption is None:
                print(f"[Warning] interpretation is None for video: {video_name}")

            caption = self.text_processor(caption)
            instruction_pool = self.reason_instruction_pool


        # emotion = self.emo2idx[t[2]]
        # personality = t[2]
        # print("t:", t)
        # print("personality:", personality)

        personality = self.per2idx[t[2]]
        sentence = self.character_lines.loc[self.character_lines['name'] == video_name, 'sentence'].values[0]
        character_line = "The person in video says: {}. ".format(sentence)
        
        instruction = "<video><VideoHere></video> <feature><FeatureHere></feature> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        return {
            "image": image,
            "video_features": video_features,
            "instruction_input": instruction,
            "answer": caption,
            "personality": personality,
            "image_id": video_name
        }
    
    def extract_frame(self, video_path):
        video_capture = cv2.VideoCapture(video_path) #read the first frame
        success, frame = video_capture.read()
        if not success:
            raise ValueError("Failed to read video file:", video_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_capture.release()

        return frame_rgb


    def get(self, video_name):
        # FaceMAE feature
        FaceMAE_feats_path = os.path.join(self.file_path, 'mae_340_UTT', video_name + '.npy')
        FaceMAE_feats = torch.tensor(np.load(FaceMAE_feats_path))

        # VideoMAE feature
        VideoMAE_feats_path = os.path.join(self.file_path, 'maeV_399_UTT', video_name + '.npy')
        VideoMAE_feats = torch.tensor(np.load(VideoMAE_feats_path))

        # Audio feature
        Audio_feats_path = os.path.join(self.file_path, 'wav2vec2-large-960h-UTT', video_name + '.npy')
        Audio_feats = torch.tensor(np.load(Audio_feats_path))

        return FaceMAE_feats, VideoMAE_feats, Audio_feats
