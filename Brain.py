import utils as u
import os
import torch
#erase
import time
import tarfile
import itertools
import numpy as np


class Elliptic_Temporal_Dataset():
	def __init__(self,args):
		args.elliptic_args = u.Namespace(args.elliptic_args)

		data = np.load('data/brain/Brain_5000nodes.npz')

		self.nodes_labels_times = self.load_node_labels(data)

		self.edges = self.load_transactions(data)

		self.nodes, self.nodes_feats = self.load_node_feats(data)

	def load_node_feats(self, data):
		# data = u.load_data_from_tar(elliptic_args.feats_file, tar_archive, starting_line=0)
		features = data['attmats']
		nodes_feats = []
		for i in range(12):
			nodes_feats.append(torch.FloatTensor(features)[:, i])

		self.num_nodes = 5000
		print(self.num_nodes)
		self.feats_per_node = len(nodes_feats[0])

		return nodes_feats, nodes_feats


	def load_node_labels(self, data):
		# labels = u.load_data_from_tar(elliptic_args.classes_file, tar_archive, replace_unknow=True).long()
		# times = u.load_data_from_tar(elliptic_args.times_file, tar_archive, replace_unknow=True).long()
		lcols = u.Namespace({'nid': 0,
							 'label': 1})
		# tcols = u.Namespace({'nid':0, 'time':1})


		labels = data['labels']

		nodes_labels_times =[]
		for i in range(len(labels)):
			label = labels[i].tolist().index(1)
			nodes_labels_times.append([i, label])

		nodes_labels_times = torch.LongTensor(nodes_labels_times)

		return nodes_labels_times

	def load_transactions(self, data):
		adj = data['adjs']

		tcols = u.Namespace({'source': 0,
							 'target': 1,
							 'time': 2})
		
		data = []

		t = 0
		for graph in adj:
			for i in range(len(graph)):
				temp = np.concatenate((np.ones(len(np.where(graph[i] == 1)[0])).reshape(-1,1),np.where(graph[i] == 1)[0].reshape(-1,1), np.ones(len(np.where(graph[i] == 1)[0])).reshape(-1,1) * t) , 1).astype(int).tolist()
				data.extend(temp)
			t += 1

		data= torch.LongTensor(data)

		self.max_time = torch.FloatTensor([11])
		self.min_time = 0

		print(data.size(0))

		return {'idx': data, 'vals': torch.ones(data.size(0))}
