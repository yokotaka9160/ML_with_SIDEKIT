import h5py
import os
import csv
import numpy as np
import librosa

# with open('JVPD_filename_all.txt') as inputFile:
#     ALLList = inputFile.read().split('\n')

with open('JVPD_filename_physical.txt') as inputFile:
    ALLList = inputFile.read().split('\n')

with open('JVPD_ALLfeatures_mfcc.csv', 'w',newline="") as f2:
    header = ['filename','sex','sex_num','energy_mean','energy_std','mfcc_cep_std_max','mfcc_cep_std_min',
              'mfcc_cep_s_D_max_min','mfcc_fb_mean_max','mfcc_fb_mean_min','mfcc_fb_M_D_max_min',
              'fb_std_max','fb_std_min','fb_std_D_max_min']
    cep_num = 19
    width = 5
    for i in range(cep_num):
        header.append("mfcc_cep_mean_"+str(i+1))
    for i in range(cep_num):
        header.append("mfcc_cep_mean_delta"+str(i+1))

    writer = csv.writer(f2)
    writer.writerow(header)

    for name in ALLList:
            fileDir = os.path.join('features', name)
            featureDir = fileDir + '.h5'
            with h5py.File(featureDir,'r') as f:
                #print(f.keys())
                #print(f[name].keys())
                ene_M = f[name]['energy_mean'].value
                ene_s = f[name]['energy_std'].value
                # ene_min = f[name]['energy_min_range'][0]
                # ene_range = f[name]['energy_min_range'][1]
                # ene_D_min_range = ene_min + ene_range
                # cep_min = f[name]['cep_min_range'][0]
                # cep_range = f[name]['cep_min_range'][1]
                # cep_D_min_range = cep_min + cep_range
                cep_s_max = np.max(f[name]['cep_std'].value)
                cep_s_min = np.amin(f[name]['cep_std'].value)
                cep_s_D_max_min = cep_s_max - cep_s_min
                fb_M_max = np.max(f[name]['fb_mean'].value)
                fb_M_min = np.amin(f[name]['fb_mean'].value)
                fb_M_D_max_min = fb_M_max - fb_M_min
                fb_s_max = np.max(f[name]['fb_std'].value)
                fb_s_min = np.amin(f[name]['fb_std'].value)
                fb_s_D_max_min = fb_s_max - fb_s_min
                mfcc_mean = f[name]['cep_mean'].value
                mfcc_delta = librosa.feature.delta(mfcc_mean,width=width)
                mfcc_mean = mfcc_mean.tolist()
                mfcc_delta = mfcc_delta.tolist()
                if 'f' in name:
                    element = [name,"female",1,ene_M,ene_s,cep_s_max,cep_s_min,cep_s_D_max_min,fb_M_max,fb_M_min,
                                     fb_M_D_max_min,fb_s_max,fb_s_min,fb_s_D_max_min]
                    element = element + mfcc_mean + mfcc_delta
                    writer.writerow(element)
                elif 'm' in name:
                    element = [name, "male", 0, ene_M, ene_s, cep_s_max, cep_s_min, cep_s_D_max_min, fb_M_max, fb_M_min,
                               fb_M_D_max_min, fb_s_max, fb_s_min, fb_s_D_max_min]
                    element = element + mfcc_mean + mfcc_delta
                    writer.writerow(element)
    f2.close()
