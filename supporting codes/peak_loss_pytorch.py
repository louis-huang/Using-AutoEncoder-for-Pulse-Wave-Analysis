#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 13:26:58 2018

@author: jimmy
"""

import detect_peaks

def cal_loss(yp, yt):
    total_loss = 0
    for i in range(len(yt)):
        cur_y = yt[i]
        peak = detect_peaks.detect_peaks(cur_y, mpd=150, show = False)
        valley = detect_peaks.detect_peaks(cur_y, valley = True, mpd=150, show = False)
        total_loss += ((yt[i][peak] - yp[i][peak]) **2).sum() + ((yt[i][valley] - yp[i][valley]) **2).sum()
    total_loss = total_loss / len(yt)
    return total_loss

