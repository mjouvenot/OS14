#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 17:05:42 2018

@author: martinjouvenot
"""
import csv
import ast
import matplotlib.pyplot as plt

def find(relationship, change_id, dataset):
    #depth first search
    visited, stack = set(), [change_id]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(dataset[vertex][relationship] - visited)
    return visited
    

def get_list_from_text(input_string):
    result = set()
    if input_string:
        result = set(ast.literal_eval(input_string.replace(';',',')))
    return result

dataset = {}
with open('CR87_dataset.csv', 'r') as csv_dataset:
    dictReader = csv.DictReader(csv_dataset)
    for row in dictReader:
        dataset[int(row['id'])] = {'parents': get_list_from_text(row['parents']),
                                  'children': get_list_from_text(row['children'])}

#dataset is ordered by column then row
complete_dataset = []
key_list = list(dataset.keys())

for key in key_list:
    parents = find('parents', key, dataset)
    row = [0 for i in key_list]
    for parent in parents:
        row[key_list.index(parent)] = 1
    complete_dataset.append(row)

print(complete_dataset)

plt.matshow(complete_dataset)
