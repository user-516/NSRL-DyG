import torch
import numpy as np
import random


class Data:
	def __init__(self, src, dst, timestamps, edge_idxs, labels):
		self.src = src
		self.dst = dst
		self.timestamps = timestamps
		self.edge_idxs = edge_idxs
		self.labels = labels
		self.n_interactions = len(src)
		self.unique_nodes = set(src) | set(dst)
		self.n_unique_nodes = len(self.unique_nodes)


# 向节点嵌入中添加噪声
def add_noise_to_node_features(node_features, noise_level):
	# 生成与节点特征形状相同的噪声
    noise = np.random.normal(0, noise_level, node_features.shape)
    
    # 将噪声添加到节点特征中
    noisy_node_features = node_features + noise
    
    return noisy_node_features


def _dataset_paths(DatasetName, data_root):
	"""Locate the three files of a dataset.

	Two layouts are accepted: `<root>/<Name>/ml_<Name>.*` (one directory per dataset, as
	in E:/dataset/Ada-DyGNN-dataset) and the flat `<root>/ml_<Name>.*`.
	"""
	import os
	candidates = [os.path.join(data_root, DatasetName), data_root]
	for base in candidates:
		graph = os.path.join(base, "ml_{}.csv".format(DatasetName))
		if os.path.exists(graph):
			return (graph,
					os.path.join(base, "ml_{}.npy".format(DatasetName)),
					os.path.join(base, "ml_{}_node.npy".format(DatasetName)))
	raise FileNotFoundError(
		"ml_{0}.csv not found under {1} (looked in {2}/{0} and {2})".format(
			DatasetName, data_root, data_root))


def _load_graph_csv(path):
	"""Read an ml_*.csv into named numpy arrays, mapping columns by header name.

	The files in the wild come in two layouts — `,u,i,ts,label,idx` (uci, wikipedia,
	lastfm, enron, USLegis, CanParl) and `,u,i,ts,label,timestamp,idx` (UCI-Msg) — so the
	column index of `idx` differs. Resolving by name handles both.

	Kept on numpy rather than pandas so the pipeline needs no extra dependency.
	"""
	with open(path, 'r') as fh:
		header = [c.strip() for c in fh.readline().strip().split(',')]
	arr = np.loadtxt(path, delimiter=',', skiprows=1)
	out = {}
	for name in ('u', 'i', 'ts', 'label', 'idx'):
		if name not in header:
			raise ValueError("column %r missing from %s (header: %s)" % (name, path, header))
		out[name] = arr[:, header.index(name)].astype(np.int64)
	return out


def get_data(DatasetName, data_root="./data"):
	graph_path, edge_path, node_path = _dataset_paths(DatasetName, data_root)
	graph = _load_graph_csv(graph_path)
	edge_features = np.load(edge_path)
	node_features = np.load(node_path)
	# fommer_node_features = np.load("./data/ml_{}_node.npy".format(DatasetName))
	# node_features = add_noise_to_node_features(fommer_node_features,0.1)
	val_time, test_time = list(np.quantile(graph['ts'], [0.8, 0.9]))

	src = graph['u']
	dst = graph['i']
	edge_idxs = graph['idx'] - 1
	labels = graph['label']
	timestamps = graph['ts']

	full_data = Data(src, dst, timestamps, edge_idxs, labels)

	random.seed(2020)
	node_set = set(src) | set(dst)
	n_total_nodes = len(node_set)

	test_node_set = set(src[timestamps > val_time]) | (set(dst[timestamps > val_time]))

	new_test_node_set = set(random.sample(sorted(test_node_set), int(0.1 * n_total_nodes)))
	held_out = np.fromiter(new_test_node_set, dtype=src.dtype, count=len(new_test_node_set))
	new_test_src_mask = np.isin(src, held_out)
	new_test_dst_mask = np.isin(dst, held_out)

	observed_edges_mask = np.logical_and(~new_test_src_mask, ~new_test_dst_mask)

	train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
	train_data = Data(src[train_mask], dst[train_mask], timestamps[train_mask],
					  edge_idxs[train_mask], labels[train_mask])
	train_node_set = set(train_data.src) | set(train_data.dst)

	assert len(train_node_set & new_test_node_set) == 0

	new_node_set = node_set - train_node_set

	val_mask = np.logical_and(timestamps > val_time, timestamps <= test_time)
	test_mask = timestamps > test_time

	new_arr = np.fromiter(new_node_set, dtype=src.dtype, count=len(new_node_set))
	edge_contains_new_node_mask = np.logical_or(np.isin(src, new_arr), np.isin(dst, new_arr))
	new_node_val_mask = np.logical_and(edge_contains_new_node_mask, val_mask)
	new_node_test_mask = np.logical_and(edge_contains_new_node_mask, test_mask)

	val_data = Data(src[val_mask], dst[val_mask], timestamps[val_mask],
					edge_idxs[val_mask], labels[val_mask])
	test_data = Data(src[test_mask], dst[test_mask], timestamps[test_mask],
					 edge_idxs[test_mask], labels[test_mask])
	new_node_val_data = Data(src[new_node_val_mask], dst[new_node_val_mask],
							 timestamps[new_node_val_mask], edge_idxs[new_node_val_mask],
							 labels[new_node_val_mask])
	new_node_test_data = Data(src[new_node_test_mask], dst[new_node_test_mask],
							  timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
							  labels[new_node_test_mask])

	print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
																				 full_data.n_unique_nodes))
	print("The training dataset has {} interactions, involving {} different nodes".format(
		train_data.n_interactions, train_data.n_unique_nodes))
	print("The validation dataset has {} interactions, involving {} different nodes".format(
		val_data.n_interactions, val_data.n_unique_nodes))
	print("The test dataset has {} interactions, involving {} different nodes".format(
		test_data.n_interactions, test_data.n_unique_nodes))
	print("The new node validation dataset has {} interactions, involving {} different nodes".format(
		new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
	print("The new node test dataset has {} interactions, involving {} different nodes".format(
		new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
	print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
		len(new_test_node_set)))

	return node_features, edge_features, full_data, train_data, \
		   val_data, test_data, new_node_val_data, new_node_test_data,timestamps[-1]


def computer_time_statics(src, dst, timestamps):
	last_timestamp_src = dict()
	last_timestamp_dst = dict()
	all_timediffs_src = []
	all_timediffs_dst = []
	for k in range(len(src)):
		src_id = src[k]
		dst_id = dst[k]
		cur_timestamp = timestamps[k]
		if src_id not in last_timestamp_src.keys():
			last_timestamp_src[src_id] = 0
		if dst_id not in last_timestamp_dst.keys():
			last_timestamp_dst[dst_id] = 0
		all_timediffs_src.append(cur_timestamp - last_timestamp_src[src_id])
		all_timediffs_dst.append(cur_timestamp - last_timestamp_dst[dst_id])
		last_timestamp_src[src_id] = cur_timestamp
		last_timestamp_dst[dst_id] = cur_timestamp
	assert len(all_timediffs_src) == len(src)
	assert len(all_timediffs_dst) == len(dst)
	mean_time_shift_src = np.mean(all_timediffs_src)
	std_time_shift_src = np.std(all_timediffs_src)
	mean_time_shift_dst = np.mean(all_timediffs_dst)
	std_time_shift_dst = np.std(all_timediffs_dst)
	return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
