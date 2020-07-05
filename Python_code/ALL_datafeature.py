# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:05:23 2019

@author: yokoo takaya
"""

import sidekit
import os
import sys
import multiprocessing
import matplotlib.pyplot as mpl
import logging
import numpy as np
# sys.path.append("C:/Users/yokoo takaya/Desktop/Spyder/libsvm-master/python/")


# logging.basicConfig(filename='log/JVPD_ubm-gmm.log',level=logging.DEBUG)

distribNb = 512  # number of Gaussian distributions for each GMM
JVPD_Path = r'C:\Users\yokoo takaya\Desktop\JVPD'

# Default for RSR2015
audioDir = os.path.join(JVPD_Path, 'JVPD_ALLsound')

# Automatically set the number of parallel process to run.
# The number of threads to run is set equal to the number of cores available
# on the machine minus one or to 1 if the machine has a single core.
nbThread = max(multiprocessing.cpu_count()-1, 1)

print('Load task definition')
enroll_idmap = sidekit.IdMap('idmap_JVPD.h5')
test_ndx = sidekit.Ndx('ndx_JVPD.h5')
key = sidekit.Key('key_JVPD.h5')
with open('JVPD_filename_all.txt') as inputFile:
    ubmList = inputFile.read().split('\n')
    

logging.info("Initialize FeaturesExtractor")
extractor = sidekit.FeaturesExtractor(audio_filename_structure=audioDir+"/{}.wav",
                                      feature_filename_structure="./features_PLP/{}.h5",
                                      sampling_frequency=16000,
                                      lower_frequency=133.3333,
                                      higher_frequency=6955.4976,
                                      filter_bank="lin",
                                      filter_bank_size=40,
                                      window_size=0.025,
                                      shift=0.01,
                                      ceps_number=19,
                                      vad="snr",
                                      snr=40,
                                      pre_emphasis=0.97,
                                      save_param=["vad", "energy", "cep" , "fb"],
                                      keep_all_features=False)

#:param save_param: list of strings that indicate which parameters to save. The strings can be:
# "cep" for cepstral coefficients, "fb" for filter-banks, "energy" for the log-energy,
# "bnf" for bottle-neck features and "vad" for the frame selection labels. In the resulting files, parameters are
#  always concatenated in the following order: (energy,fb, cep, bnf, vad_label).

# Get the complete list of features to extract
show_list = np.unique(np.hstack([ubmList, enroll_idmap.rightids, np.unique(test_ndx.segset)]))
show_list = show_list.tolist()
channel_list = np.zeros_like(show_list, dtype=int)
logging.info("Extract features and save to disk")
extractor.save_list(show_list=show_list,
                    channel_list=channel_list)



