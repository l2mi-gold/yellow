#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Program that tests the Yellow challenge Data Manager class.
"""

from zDataManager import DataManager
input_dir = "../public_data"
output_dir = "../res"

basename = 'Iris'
D = DataManager(basename, input_dir)
print D
    
D.DataStats('train')
D.ShowScatter(1, 2, 'train')