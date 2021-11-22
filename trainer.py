import torch
import utils as u
import logger
import time
import pandas as pd
import numpy as np
from torch.nn import functional as F

class Trainer():
	def __init__(self,args, splitter, gcn, classifier, classifier2, adapter, adapter2, time_predictor, comp_loss, dataset, num_classes):
		self.args = args
		self.splitter = splitter
		self.tasker = splitter.tasker
		self.gcn = gcn
		self.classifier = classifier
		self.classifier2 = classifier2
		self.adapter = adapter
		# self.adapter2 = adapter2
		self.time_predictor = time_predictor
		self.comp_loss = comp_loss

		self.num_nodes = dataset.num_nodes
		self.data = dataset
		self.num_classes = num_classes

		self.logger = logger.Logger(args, self.num_classes)

		self.init_optimizers(args)

	def init_optimizers(self,args):
		params = self.gcn.parameters()
		self.gcn_opt = torch.optim.Adam(params, lr = args.learning_rate)
		params = self.classifier.parameters()
		self.classifier_opt = torch.optim.Adam(params, lr = args.learning_rate)
		params = self.classifier2.parameters()
		self.classifier_opt2 = torch.optim.Adam(params, lr = args.learning_rate)
		params = self.adapter.parameters()
		self.adapter_opt = torch.optim.Adam(params, lr = args.learning_rate)
		params = self.time_predictor.parameters()
		self.time_opt = torch.optim.Adam(params, lr = args.learning_rate)
		# params = self.adapter2.parameters()
		# self.adapter2_opt = torch.optim.Adam(params, lr = args.learning_rate)

		

		self.gcn_opt.zero_grad()
		self.classifier_opt.zero_grad()
		self.classifier_opt2.zero_grad()
		self.adapter_opt.zero_grad()
		self.time_opt.zero_grad()
		# self.adapter2_opt.zero_grad()

	def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
		torch.save(state, filename)

	def load_checkpoint(self, filename, model):
		if os.path.isfile(filename):
			print("=> loading checkpoint '{}'".format(filename))
			checkpoint = torch.load(filename)
			epoch = checkpoint['epoch']
			self.gcn.load_state_dict(checkpoint['gcn_dict'])
			self.classifier.load_state_dict(checkpoint['classifier_dict'])
			self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
			self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
			self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
			return epoch
		else:
			self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
			return 0

	def train(self):
		self.tr_step = 0
		best_eval_valid = 0
		eval_valid = 0
		epochs_without_impr = 0

		for e in range(self.args.num_epochs):
			eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, 'TRAIN', grad = True)
			if len(self.splitter.dev)>0 and e>self.args.eval_after_epochs:
				eval_valid, _ = self.run_epoch(self.splitter.dev, e, 'VALID', grad = True)
				if eval_valid>best_eval_valid:
					best_eval_valid = eval_valid
					epochs_without_impr = 0
					print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Best valid measure:'+str(eval_valid))
				else:
					epochs_without_impr+=1
					if epochs_without_impr>self.args.early_stop_patience:
						print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Early stop.')
						break

			if len(self.splitter.test)>0 and eval_valid==best_eval_valid and e>self.args.eval_after_epochs:
				eval_test, _ = self.run_epoch(self.splitter.test, e, 'TEST', grad = True)

				if self.args.save_node_embeddings:
					self.save_node_embs_csv(nodes_embs, self.splitter.train_idx, log_file+'_train_nodeembs.csv.gz')
					self.save_node_embs_csv(nodes_embs, self.splitter.dev_idx, log_file+'_valid_nodeembs.csv.gz')
					self.save_node_embs_csv(nodes_embs, self.splitter.test_idx, log_file+'_test_nodeembs.csv.gz')


	def run_epoch(self, split, epoch, set_name, grad):
		t0 = time.time()
		log_interval=999
		if set_name=='TEST':
			log_interval=1
		self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)

		# torch.set_grad_enabled(grad)
		for s in split:
			# print(s)
			if self.tasker.is_static:
				s = self.prepare_static_sample(s)
			else:
				s = self.prepare_sample(s, set_name)

			if set_name not in ['TEST', 'VALID']:
				predictions, nodes_embs, losses = self.meta_learning(s.hist_adj_list,  # torch sparse tensor, edge index 2 * N_e, edge weight, 1 * N
													s.hist_ndFeats_list,  # node features, one-hot degree feature for sbm
													s.label_sp['idx'],  # all edges including existing and non-existing
													s.label_sp['vals'],
													s.node_mask_list,
													s.next_label,
													time=s.idx)  
			else:
				predictions, nodes_embs, loss = self.predict(s.hist_adj_list,  # torch sparse tensor, edge index 2 * N_e, edge weight, 1 * N
													s.hist_ndFeats_list,  # node features, one-hot degree feature for sbm
													s.label_sp['idx'],
													s.label_sp['vals'],
													s.node_mask_list,
													s.next_label,
													s.idx)  

			if set_name not in ['TEST', 'VALID']:
				loss = torch.stack(losses).mean()
			# print(loss)
			if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
			else:
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())

			if set_name not in ['TEST', 'VALID']:
				self.optim_step(loss)

		# torch.set_grad_enabled(True)
		eval_measure = self.logger.log_epoch_done()

		return eval_measure, nodes_embs
	
	def meta_learning(self, hist_adj_list, hist_ndFeats_list, node_indices, node_labels, mask_list, next_label_adj = None, time=None):
		'''
		In inner loop, we optimize the MLP and the adapter and we do not optimize GCN
		'''
		# self.gcn.requires_grad = False
		fast_weights = list(self.adapter.parameters())
		fast_weights1 = list(self.gcn.parameters())

		self.gcn.train()

		losses = []

		for k in range(len(hist_adj_list)):
			# print(k)

			predict_batch_size = 100000
			nodes_embs = self.gcn(hist_adj_list[k],  # true edges
								hist_ndFeats_list[k],  # node features
								mask_list[k],
								fast_weights1) 

			mask = self.adapter(nodes_embs, fast_weights)
			other_embs = torch.mul(mask, nodes_embs)  # for MLP
			temporal_embs = torch.mul(1 - mask, nodes_embs)   # for temporal matching

			if self.args.task in ["link_pred", "edge_cls"]:
				loss = F.smooth_l1_loss(self.time_predictor(temporal_embs.mean(0)), torch.FloatTensor([k+1]).to(temporal_embs.device))
			else:
				loss = F.smooth_l1_loss(self.time_predictor(temporal_embs.mean(0)), torch.FloatTensor([k+1]).to(temporal_embs.device))
			
			grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
			grad1 = torch.autograd.grad(loss, fast_weights1)

			fast_weights = list(map(lambda p: p[1] - self.args.learning_rate * 10. * p[0], zip(grad, fast_weights)))
			fast_weights1 = list(map(lambda p: p[1] - self.args.learning_rate * 10. * p[0], zip(grad1, fast_weights1)))

			del grad, grad1, loss, other_embs, temporal_embs, mask, nodes_embs

			'''
			Outer loop
			'''

			nodes_embs = self.gcn(hist_adj_list[-1],  # true edges
							hist_ndFeats_list[-1],  # node features
							mask_list[-1],
							fast_weights1) 

			predict_batch_size = 100000

			mask = self.adapter(nodes_embs, fast_weights)
			other_embs = torch.mul(mask, nodes_embs)  # for MLP
			temporal_embs = torch.mul(1 - mask, nodes_embs)   # for temporal matching

			gather_predictions=[]
			for i in range(1 +(node_indices.size(1)//predict_batch_size)):
				cls_input1 = self.gather_node_embs(other_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
				cls_input2 = self.gather_node_embs(temporal_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
				predictions = self.classifier(cls_input1) + self.classifier2(cls_input2)
				gather_predictions.append(predictions)
			gather_predictions=torch.cat(gather_predictions, dim=0)

			loss = self.comp_loss(gather_predictions, node_labels) + 0.1 * F.smooth_l1_loss(self.time_predictor(temporal_embs.mean(0)), torch.FloatTensor([len(hist_adj_list)]).to(temporal_embs.device))
			# loss = self.comp_loss(gather_predictions, node_labels)
			losses.append(loss)

			# torch.cuda.empty_cache()

		return gather_predictions, nodes_embs, losses

	def predict(self, hist_adj_list, hist_ndFeats_list, node_indices, node_labels, mask_list, next_label_adj, time):

		self.gcn.train()

		fast_weights = list(self.adapter.parameters())
		fast_weights1 = list(self.gcn.parameters())

		for k in range(len(hist_adj_list)):
			predict_batch_size = 100000

			nodes_embs = self.gcn(hist_adj_list[k],  # true edges
								hist_ndFeats_list[k],  # node features
								mask_list[k],
								fast_weights1) 

			mask = self.adapter(nodes_embs, fast_weights)
			other_embs = torch.mul(mask, nodes_embs)  # for MLP
			temporal_embs = torch.mul(1 - mask, nodes_embs)   # for temporal matching

			loss = F.smooth_l1_loss(self.time_predictor(temporal_embs.mean(0)), torch.FloatTensor([k+1]).to(temporal_embs.device))
			grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
			grad1 = torch.autograd.grad(loss, fast_weights1)

			fast_weights = list(map(lambda p: p[1] - self.args.learning_rate * 10. * p[0], zip(grad, fast_weights)))
			fast_weights1 = list(map(lambda p: p[1] - self.args.learning_rate * 10. * p[0], zip(grad1, fast_weights1)))
		
		with torch.no_grad():
			self.gcn.eval()

			nodes_embs = self.gcn(hist_adj_list[-1],  # true edges
								hist_ndFeats_list[-1],  # node features
								mask_list[-1],
								fast_weights1)      # nodes

			mask = self.adapter(nodes_embs, fast_weights)
			other_embs = torch.mul(mask, nodes_embs)  # for MLP
			temporal_embs = torch.mul(1 - mask, nodes_embs)   # for temporal matching

			predict_batch_size = 100000
			gather_predictions=[]
			for i in range(1 +(node_indices.size(1)//predict_batch_size)):
				cls_input1 = self.gather_node_embs(other_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
				cls_input2 = self.gather_node_embs(temporal_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
				predictions = self.classifier(cls_input1) + self.classifier2(cls_input2)
				gather_predictions.append(predictions)
			gather_predictions=torch.cat(gather_predictions, dim=0)

			loss = self.comp_loss(gather_predictions, node_labels) + 0.1 * F.smooth_l1_loss(self.time_predictor(temporal_embs.mean(0)), torch.FloatTensor([len(hist_adj_list)]).to(temporal_embs.device))
			# loss = self.comp_loss(gather_predictions, node_labels)

		return gather_predictions, nodes_embs, loss

	def gather_node_embs(self,nodes_embs,node_indices):
		cls_input = []

		for node_set in node_indices:
			cls_input.append(nodes_embs[node_set])
		return torch.cat(cls_input,dim = 1)

	def optim_step(self,loss):

		self.gcn_opt.zero_grad()
		self.classifier_opt.zero_grad()
		self.classifier_opt2.zero_grad()
		self.adapter_opt.zero_grad()
		# self.adapter2_opt.zero_grad()
		self.time_opt.zero_grad()

		loss.backward()

		torch.nn.utils.clip_grad_norm_(self.gcn.parameters(), 1)
		torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1)
		torch.nn.utils.clip_grad_norm_(self.classifier2.parameters(), 1)
		torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1)
		torch.nn.utils.clip_grad_norm_(self.time_predictor.parameters(), 1)
		# torch.nn.utils.clip_grad_norm_(self.adapter2.parameters(), 1)

		self.gcn_opt.step()
		self.classifier_opt.step()
		self.classifier_opt2.step()
		self.adapter_opt.step()
		# self.adapter2_opt.step()
		self.time_opt.step()


	def prepare_sample(self,sample, set_name):
		sample = u.Namespace(sample)
		for i,adj in enumerate(sample.hist_adj_list):

			if self.args.model == 'gat':
				sample.hist_adj_list[i] = torch.LongTensor(adj['idx'][0].t()).to(self.args.device)
			else:
				adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes]) # torch sparse tensor, edge index 2 * N_e, normalized adj values, 1 * N_e
				sample.hist_adj_list[i] = adj.to(self.args.device)

			nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i]) # node features, one-hot degree feature for sbm

			sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
			node_mask = sample.node_mask_list[i]
			sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer


		label_sp = self.ignore_batch_dim(sample.label_sp)  # label of all edges, 1 for existing / 0 for non-existing in link prediction

		if self.args.task in ["link_pred", "edge_cls"]:
			label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   
		else:
			label_sp['idx'] = label_sp['idx'].to(self.args.device)

		label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
		sample.label_sp = label_sp

		return sample

	def prepare_static_sample(self,sample):
		sample = u.Namespace(sample)

		sample.hist_adj_list = self.hist_adj_list

		sample.hist_ndFeats_list = self.hist_ndFeats_list

		label_sp = {}
		label_sp['idx'] =  [sample.idx]
		label_sp['vals'] = sample.label
		sample.label_sp = label_sp

		return sample

	def ignore_batch_dim(self,adj):
		if self.args.task in ["link_pred", "edge_cls"]:
			adj['idx'] = adj['idx'][0]
		adj['vals'] = adj['vals'][0]
		return adj

	def save_node_embs_csv(self, nodes_embs, indexes, file_name):
		csv_node_embs = []
		for node_id in indexes:
			orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

			csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())

		pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
