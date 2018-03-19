#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 17:05:42 2018

@author: martinjouvenot
"""
import csv
import ast
import matplotlib.pyplot as plt
import numpy as np

def find_all(relationship, change_id, dataset):
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

def to_dsm(dataset, with_magnitude=False, with_siblings=False, transitive_closure=False, weights=None):
    #dataset is ordered by column then row
    complete_dataset = []
    key_list = list(dataset.keys())
    weights_map = {'parent': max(weights) if weights else 1,
                   'sibling': min(weights) if weights else 1}
    
    for key in key_list:
        #If we ask for transitive closure, we perform a depth first search to 
        #find all the ancestors
        magnitude = dataset[key]['magnitude'] if with_magnitude else 1
        if transitive_closure:
            parents = find_all('parents', key, dataset)
        #else we just stick to the parents
        else:
            parents = set(dataset[key]['parents'])
            parents.add(key)
            
        row = [0 for i in key_list]
        for parent in parents:
            row[key_list.index(parent)] = weights_map['parent']*magnitude
            
        #Check for siblings if needed
        if with_siblings:
            for sibling in dataset[key]['siblings']:
                row[key_list.index(sibling)] = weights_map['sibling']*magnitude
        
        complete_dataset.append(row)
    
    return key_list, complete_dataset

def to_aggregated_dsm(dataset, keyword):
    #initialize list of keys and DSM with zeros
    keyset = set()
    for key,value in dataset.items():
        keyset.add(value[keyword])
    keylist = sorted(list(keyset))
    dsm = [[0 for i in keylist] for j in keylist]       
                
    #loop through dataset
    for key, value in dataset.items():
        i = keylist.index(value[keyword])
        dsm[i][i] = 1
        for parent in value['parents']:
            j = keylist.index(dataset[parent][keyword])
            dsm[i][j] += 1
    
    return keylist, dsm

def to_csv(csv_output, keylist, dsm):
    #Write header
    csv_output.write(',')
    csv_output.write(','.join([str(x) for x in keylist]))
    for i in range(len(keylist)):
        csv_output.write('\n')
        csv_output.write(str(keylist[i])+',')
        csv_output.write(','.join([str(x) for x in dsm[i]]))
        
def to_csv_col_header(csv_output, keylist, dsm):
    #Write header
    csv_output.write(',,')
    csv_output.write(','.join([str(i) for i in range(len(keylist))]))
    for i in range(len(keylist)):
        csv_output.write('\n')
        csv_output.write(str(keylist[i])+','+str(i)+',')
        csv_output.write(','.join([str(x) for x in dsm[i]]))

dataset = {}
with open('CR87_dataset.csv', 'r') as csv_dataset:
    dictReader = csv.DictReader(csv_dataset)
    for row in dictReader:
        dataset[int(row['id'])] = {'parents': get_list_from_text(row['parents']),
                                  'children': get_list_from_text(row['children']),
                                  'siblings': get_list_from_text(row['siblings']),
                                  'magnitude': int(row['magnitude'])+1,
                                  'person': row['person'],
                                  'area': int(get_list_from_text(row['area']).pop()),
                                  'status': int(row['status']) if row['status'] != '' else 0}

def plot_dsm(keylist, dsm):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(dsm, cmap='Blues')
    ax.set_yticks(range(len(keylist)))
    ax.set_yticklabels(keylist)
    ax.set_xticks(range(len(keylist)))
    ax.set_xticklabels(keylist, rotation=90)
    fig.colorbar(cax)
    plt.show()

keylist, dsm = to_dsm(dataset)
plot_dsm(keylist, dsm)
keylist, dsm = to_dsm(dataset, with_magnitude=True)
plt.matshow(dsm, cmap='Blues')
keylist, dsm = to_dsm(dataset, with_magnitude=True, with_siblings=True)
plt.matshow(dsm, cmap='coolwarm')

keylist, dsm = to_aggregated_dsm(dataset, 'person')
plot_dsm(keylist, dsm)

keylist, dsm = to_aggregated_dsm(dataset, 'area')
plot_dsm(keylist, dsm)

with open('CR87_changes_dsm.csv', 'w') as csv_output:
    keylist, dsm = to_dsm(dataset)
    to_csv_col_header(csv_output, keylist, dsm)
    
with open('CR87_employees_dsm.csv', 'w') as csv_output:
    keylist, dsm = to_aggregated_dsm(dataset, 'person')
    to_csv(csv_output, keylist, dsm)
    
with open('CR87_areas_dsm.csv', 'w') as csv_output:
    keylist, dsm = to_aggregated_dsm(dataset, 'area')
    to_csv(csv_output, keylist, dsm)
#%%
keylist, dsm = to_dsm(dataset)
for i in range(len(keylist)):
    print('{1},{0}'.format(i, keylist[i]))
    
keylist, dsm = to_aggregated_dsm(dataset, 'person')
for i in range(len(keylist)):
    print('{1},{0}'.format(i, keylist[i]))
#%%
# =============================================================================
# def set_pos(pos, node, x_pos, y_pos, nodestep, direction=1):
#     if np.absolute(x_pos + direction * nodestep) < 1:
#         x_pos = x_pos + direction * nodestep
#     else:
#         direction = -1 * direction
#         x_pos = (-1 + nodestep/2) * direction
#         y_pos = y_pos - nodestep
#     pos[node] = [x_pos, y_pos]
#     return x_pos, y_pos, direction
# 
# x_pos = -1 + node_step/2
# y_pos = 1 - node_step/2
# my_pos={}
# direction = 1
# while node_list:
#     node = node_list.pop()
#     #set position for parent node
#     x_pos, y_pos, direction = set_pos(my_pos, node, x_pos, y_pos, node_step, direction)
#     desc = sorted(nx.descendants(G,node), key=lambda x: len(nx.descendants(G,x)), reverse=True)
#     for child in desc:
#         node_list.remove(child)
#         x_pos, y_pos, direction = set_pos(my_pos, child, x_pos, y_pos, node_step, direction)
# =============================================================================
def set_pos(pos, cluster, x, y, line_width, step):
    #check first if there is enough space on x axis for the cluster
    if x + cluster['size'][0]*step > 1:
        x = -1 + step/2
        y = y - (line_width+1) * step
        line_width = 1
    
    #loop on the levels
    y0 = y
    for lev in range(cluster['levels']+1):
        x0 = x
        for node in cluster[lev]:
            pos[node] = [x, y]
            x += step
        y -= step
        x = x0
        
    return x+cluster['size'][0]*step, y0, max([line_width, cluster['size'][1]])

def create_pos(G, total_squares=100):
    my_pos_2 = {}
    node_list = sorted(G.nodes, key=lambda node: len(nx.descendants(G,node)))
    clusters = []
    while node_list:
        node = node_list.pop()
        nodes = {0: [node]}
        descendants = nx.descendants(G,node)
        nodes['length'] = 1 + len(descendants)
        levels = 0
        for child in descendants:
            print(child)
            if child in node_list:
                node_list.remove(child)
            n = nx.shortest_path_length(G, node, child)
            if n in nodes:
                nodes[n].append(child)
            else:
                nodes[n] = [child]
                
            #levels is the max of n
            if n > levels:
                levels = n
                
        nodes['levels'] = levels
        nodes['size'] = (max([len(nodes[x]) for x in range(levels+1)]), levels+1)
            
        clusters.append(nodes)
        
    print(clusters)

    #Approximate preferred number of 'squares' per lines
    total_area = 2*2
    step = np.sqrt(total_area/total_squares)
            
    x_pos = -1 + step/2
    y_pos = 1 - step/2
    line_width = 1
    for cluster in clusters:
        x_pos ,y_pos, line_width = set_pos(my_pos_2, cluster, x_pos, y_pos, line_width, step)
        
    return my_pos_2

#Create graph
import networkx as nx

G = nx.DiGraph()

edges_list = []
nodes_list = []
with open('CR87_dataset.csv', 'r') as csv_dataset:
    dictReader = csv.DictReader(csv_dataset)
    for child in get_list_from_text(row['children']):
        dataset[int(row['id'])] = {'parents': get_list_from_text(row['parents']),
                      'children': get_list_from_text(row['children']),
                      'siblings': get_list_from_text(row['siblings']),
                      'magnitude': int(row['magnitude'])+1,
                      'person': row['person'],
                      'area': int(get_list_from_text(row['area']).pop()),
                      'status': int(row['status']) if row['status'] != '' else 0}
        
    nodes_list = list(dataset.keys())
    for key,value in dataset.items():
        G.add_node(nodes_list.index(key), status=value['status'], magnitude=value['magnitude'])
        edges_list += [(nodes_list.index(key), nodes_list.index(x)) for x in value['children']]

print(edges_list)
#G.add_nodes_from(range(len(nodes_list)))
G.add_edges_from(edges_list)
    
#Sort nodes depending on their status
colors = []
node_sizes = []
for node in G:
    status = G.node[node]['status']
    if status == -1:
        colors.append('xkcd:white')
    elif status == 0:
        colors.append('xkcd:silver')
    else:
        colors.append('xkcd:lightblue')
    
    node_sizes.append((G.node[node]['magnitude']+1)*100)

#pos = nx.spring_layout(G)
pos = create_pos(G, 100)
cax = nx.draw_networkx(G, pos, node_color=colors, arrows=True, cmap='Set1', node_size = node_sizes)
#nx.draw_networkx_labels(G, pos)
#nx.draw_networkx_edges(G, pos, edge_list=G.edges(), arrows=True)
plt.show()

#%%
#Measure node degrees
# =============================================================================
# x_data = G.nodes
# y_data = [G.degree(x) for x in x_data]
# plt.figure(figsize=(15,7))
# plt.bar(x_data, y_data)
# plt.axis([0,86,0,max(y_data)])
# plt.xlabel('Change (node)')
# plt.ylabel('Node Degree (in+out)')
# plt.xticks(x_data, rotation=90)
# plt.show()
# =============================================================================
import csv
keylist, dsm = to_dsm(dataset)
with open('degrees_change2change.csv', 'w') as csv_out:
    writer = csv.DictWriter(csv_out, 
                            fieldnames=['Node #', 'Change ID', 'Degree'])
    for node in sorted(G.nodes, key=lambda x: G.degree(x), reverse=True):
        writer.writerow({'Node #': node,
                         'Change ID': keylist[node],
                         'Degree': G.degree(node)})
    
#%%
#Take care of the network for areas
import networkx as nx
from operator import itemgetter
G_area = nx.DiGraph()
                                                 
keylist, dsm = to_aggregated_dsm(dataset, 'area')
G_area.add_nodes_from(keylist)
for i in range(len(dsm)):
    for j in range(len(dsm)):
        if dsm[i][j] > 0 and i is not j:
            G_area.add_edge(keylist[j], keylist[i], weight=dsm[i][j])
            
#remove 'unknowns'
nodes_to_draw = G_area.nodes - ['unknown']
print(nodes_to_draw)
        
#nx.draw_networkx(G_area)
#pos = create_pos(G_area, 50)
pos = nx.spring_layout(G_area)
edge_colors=[G_area[x[0]][x[1]]['weight'] for x in G_area.edges]
print(edge_colors)
nx.draw_networkx(G_area, pos=nx.circular_layout(G_area), nodelist=nodes_to_draw, 
                 arrows=True, node_size=1000, width=2, node_color='xkcd:lightblue',
                 edge_color=edge_colors, edge_cmap=plt.cm.cool)
plt.show()

#%%
nx.draw_networkx(G_area, pos=nx.circular_layout(G_area), nodelist=nodes_to_draw, 
                 arrows=True, node_size=500, width=2, node_color='xkcd:lightblue')
plt.show()

#%%
degrees_list = sorted(G_area.degree(), key=itemgetter(1))
print(G_area)
print(degrees_list)
(largest_hub, degree) = degrees_list[-1]
hub_ego = nx.ego_graph(G_area, largest_hub)
# Draw graph
pos = nx.spring_layout(hub_ego)
plt.figure(figsize=(3,3))
nx.draw(hub_ego, pos, node_color='b', node_size=50)
# Draw ego as large and red
nx.draw_networkx_nodes(hub_ego, pos, nodelist=[largest_hub], node_size=300, node_color='r')
plt.show()

plt.show()