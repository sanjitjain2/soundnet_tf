from extract_feat import extract_feat
from util import load_from_link
from model import Model
import tensorflow as tf 
import numpy as np 
import sys

def extract_feat_from_link(model, audio_path, config):
    sound_sample = load_from_link(audio_path)

    features = extract_feat(model, sound_sample)
    
    return features