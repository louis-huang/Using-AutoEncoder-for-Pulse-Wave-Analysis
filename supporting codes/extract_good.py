#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 20:12:15 2018

@author: louis
"""
import numpy as np
import pandas as pd
import detect_peaks
import matplotlib.pyplot as plt
import SGfilter

def build_diff(x, ind_peak):
    a=[]
    for i in ind_peak:
        a.append(x[i])
    diff = [abs(t - s) for s, t in zip(a, a[1:])]
    return diff
    
def find_good(diff, length, threshold, degree):
    cur_idx = 0
    id_list = []
    while cur_idx < len(diff) - (length - 1):
        cur_data = diff[cur_idx:(cur_idx + length)]
        std_dev = np.std(cur_data)
        if std_dev > threshold:
            cur_idx += 1
            continue
        mean = np.mean(cur_data)
        
        #find valid data points from current index to current index plus length(10), if they are all within 2 standard deviation, then it can be considered
        #as a good section
        stable = [i for i in range(cur_idx, cur_idx + length) if (diff[i] > mean -degree *std_dev) & (diff[i] < mean + degree *std_dev)]
        if len(stable) == length:
            id_list.append(stable)
        cur_idx += 1
    #remove duplicates and save as a list
    all_good = list(set([item for sublist in id_list for item in sublist]))
    
    return all_good
    
def extract(x, mpd, length, degree, threshold, show):
    x = x[np.array(~np.isnan(x))]
    #Do find good sections for peaks and valleys
    peak = detect_peaks.detect_peaks(x, mpd=mpd, show=show) 
    peak_diff = build_diff(x, peak)
    good_peak = find_good(peak_diff, length, threshold, degree)
    
    good_peak_x = peak[good_peak]
    
    valley = detect_peaks.detect_peaks(x, valley = True, mpd=mpd, show=show)
    valley_diff = build_diff(x, valley)
    good_valley = find_good(valley_diff, length, threshold, degree)
   
    good_valley_x = valley[good_valley]
    if show:
        plt.figure(figsize=(20,5))
        plt.plot(x)
        plt.scatter(x = good_peak_x, y = x[good_peak_x], color = 'red')
        plt.scatter(x = good_valley_x, y = x[good_valley_x], color = 'black')
        plt.show()
    return peak, good_peak, good_peak_x, valley, good_valley, good_valley_x
    
def longSeq(seq, length):
    long_idx = -1
    start_list = []
    end_list = []
    cur_idx = 0
    cur_length = 1
    while cur_idx < len(seq) - 1:
        if seq[cur_idx] == seq[cur_idx + 1] - 1:
            cur_length += 1
            cur_idx += 1
            if cur_length >= length:
                long_idx = cur_idx - cur_length + 1
                if seq[long_idx] not in start_list:
                    start_list.append(seq[long_idx])
        else:
            if cur_length >= length:
                end_list.append(seq[cur_idx])
            cur_length = 1
            cur_idx += 1
    if cur_length >= length:
        end_list.append(seq[cur_idx])
    
    return list(zip(start_list, end_list))

def set_range(seq, length, x_values):
    positions = longSeq(sorted(seq), length)
    range_list = []
    for i in positions:
        start = x_values[i[0]]
        end = x_values[i[1]]
        range_list.append([start, end])
    return range_list
        
    
def merge_range(r1, r2):
    final_r = []
    idx_1 = 0
    idx_2 = 0
    l1 = len(r1)
    l2 = len(r2)
    while idx_1 < l1 and idx_2 < l2:
        if r1[idx_1][0] < r2[idx_2][0] and r1[idx_1][1] > r2[idx_2][1]:
            final_r.append([r2[idx_2][0],r2[idx_2][1]])
            idx_2 += 1
        elif r1[idx_1][0] > r2[idx_2][0] and r1[idx_1][1] < r2[idx_2][1]:
            final_r.append([r1[idx_1][0],r1[idx_1][1]])
            idx_1 += 1
        elif r1[idx_1][0] < r2[idx_2][0] and r1[idx_1][1] < r2[idx_2][1] and r2[idx_2][0] < r1[idx_1][1]:
            final_r.append([r2[idx_2][0],r1[idx_1][1]])
            idx_1 += 1
        elif r1[idx_1][0] > r2[idx_2][0] and r1[idx_1][1] > r2[idx_2][1] and r1[idx_1][0] < r2[idx_2][1]:
            final_r.append([r1[idx_1][0],r2[idx_2][1]])
            idx_2 += 1
        elif r1[idx_1][1] < r2[idx_2][0]:
            idx_1 += 1
        elif r2[idx_2][1] < r1[idx_1][0]:
            idx_2 += 1
    
    return final_r


def wave_height(window, dt):
    heights = []
    length = len(dt)
    num_100 = int(length/window)
    for i in range(num_100 + 2):
        if i == (num_100 + 1):
             start = num_100 * window
             tmp = dt[start:length]
        else:
            start = i * window
            end = (i+1) * window
            tmp = dt[start:end]
        height = np.amax(tmp) -  np.amin(tmp)
        heights.append(height)
    return heights
   
def ex_good(cur_dt, mpd, length, degree, show):
    peak, good_peak, good_peak_x, valley, good_valley, good_valley_x = extract(cur_dt, mpd, length, degree, show)
    good_peak_ranges = set_range(good_peak, 10, peak)
    good_valley_ranges = set_range(good_valley, 10, valley)
    
    final_range = merge_range(good_peak_ranges, good_valley_ranges)
    
    if show:
        plt.figure(figsize=(20,5))
        for i in final_range:
            idxx = [j for j in range(i[0],i[1] + 1)]
            plt.scatter(idxx,cur_dt[idxx], color = 'red')
        plt.plot(cur_dt)
        plt.show()
'''
data = pd.read_csv("data_cleaned.csv")
idx = 61
dt = np.array(data.iloc[idx], dtype = np.float64)
peak, good_peak, good_peak_x, valley, good_valley, good_valley_x = extract(dt, 80, 10, 2, 10000,True)
'''