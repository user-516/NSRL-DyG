import torch
import math
from tqdm import tqdm
import argparse
from pathlib import Path
from models.NSRLDyG import NSRLDyG
from utils.data_processing import get_data
from utils.utils import get_neighbor_finder, RandEdgeSampler, EarlyStopMonitor
from utils.evaluationAUC import eval_edge_prediction_auc
from utils.log_and_checkpoints import set_logger, get_checkpoint_path
from utils.utils import get_parameter_sizes


# n_degree 是用于聚合邻居信息时考虑的邻居数量，而 n_update_degree 是在强化学习决策过程中考虑的邻居数量(即要更新的邻居数量)
ModelName = 'NSRLDyG'
parser = argparse.ArgumentParser('NSRLDyG')
parser.add_argument('-d', '--data', type=str, default='UCI-Msg')
parser.add_argument('--bs', type=int, default=16, help='Batch_size')
# n_degree默认200
parser.add_argument('--n_degree', type=int, default=20, help='Number of neighbors to sample')
# n_update_degree参数的大小要和类MultiHeadAttention中x_expanded = x.unsqueeze(1).expand(-1, 200, -1).to(self.device) 同时变化
# n_update_degree默认20
parser.add_argument('--n_update_degree', type=int, default=20, help='Number of neighbors to sample')
parser.add_argument('--n_epoch', type=int, default=30, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
log_to_file = True
args = parser.parse_args()
dataset = args.data
Epoch = args.n_epoch
Batchsize = args.bs
n_neighbors = args.n_degree
n_update_neighbors = args.n_update_degree
lr = args.lr
logger, time_now = set_logger(ModelName, dataset, "", log_to_file)
Path("log/{}/{}/checkpoints".format(ModelName, time_now)).mkdir(parents=True, exist_ok=True)
Path("./Ap-result/").mkdir(parents=True, exist_ok=True)
f = open("./Ap-result/{}.txt".format(dataset), "a+")
f.write("bs = {}, degree = {}, up_degree = {}, lr = {}".format(Batchsize, n_neighbors, n_update_neighbors, lr))
f.write("\n")
# data processing
node_features, edge_features, full_data, train_data, \
val_data, test_data, new_node_val_data, new_node_test_data, tot_time = get_data(dataset)
# initialize temporal graph
train_neighbor_finder = get_neighbor_finder(train_data, False)
full_neighbor_finder = get_neighbor_finder(full_data, False)
# initialize negative samplers
train_rand_sampler = RandEdgeSampler(train_data.src, train_data.dst, seed=0)
val_rand_sampler = RandEdgeSampler(full_data.src, full_data.dst, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.src, new_node_val_data.dst, seed=1)
test_rand_sampler = RandEdgeSampler(full_data.src, full_data.dst, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.src, new_node_test_data.dst, seed=3)

device = 'cuda'
model = NSRLDyG(node_features.shape[0], n_neighbors=n_neighbors, n_update_neighbors=n_update_neighbors,
				 edge_dim=edge_features.shape[1], emb_dim=64, message_dim=64,neighbor_finder=train_neighbor_finder,
				 edge_feat=edge_features, tot_time=tot_time, device=device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
LOSS = []
Policy_LOSS = []
acc, ap, auc = [], [], []
nacc, nap, nauc = [], [], []
check_path = []
early_stopper = EarlyStopMonitor(max_round=100)
# =====================================
logger.debug(f'model name: {ModelName}, #parameters: {get_parameter_sizes(model) * 4} B, '
			 f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')
# =====================================
for e in tqdm(range(Epoch)):
	logger.debug('start {} epoch'.format(e))
	num_batch = math.ceil(len(train_data.src) / Batchsize)
	Loss = 0
	Policy_Loss = 0
	Reward = 0
	cnt = 0
	sum_size1 = 0
	sum_size2 = 0
	model.reset_graph()
	model.set_neighbor_finder(train_neighbor_finder)
	model.train()

	for i in range(num_batch):
		st_idx = i * Batchsize
		ed_idx = min((i + 1) * Batchsize, len(train_data.src))
		src_batch = train_data.src[st_idx:ed_idx]
		dst_batch = train_data.dst[st_idx:ed_idx]
		edge_batch = train_data.edge_idxs[st_idx:ed_idx]
		timestamp_batch = train_data.timestamps[st_idx:ed_idx]
		size = len(src_batch)
		loss = 0
		optimizer.zero_grad()
		_, negatives_batch = train_rand_sampler.sample(size)
		with torch.no_grad():
			pos_label = torch.ones(size, dtype=torch.float, device=device)
			neg_label = torch.zeros(size, dtype=torch.float, device=device)
		pos_prob, neg_prob, reinforce_loss, reward = model(src_batch, dst_batch, negatives_batch, edge_batch, timestamp_batch)
		loss += criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
		loss /= size
		Loss += loss.item()
		Policy_Loss += reinforce_loss
		Reward += reward
		loss += reinforce_loss
		loss.backward()
		optimizer.step()
		model.detach_memory()
	LOSS.append(Loss)
	Policy_LOSS.append(Policy_Loss)
	logger.debug("loss in whole dataset = {}".format(Loss))
	# =====================================================
	logger.debug("loss in Reinforcement = {}".format(Policy_Loss))
	logger.debug("Reward  = {}".format(Reward))
	# =====================================================
	# validation
	train_memory_backup = model.back_up_memory()
	model.eval()
	model.set_neighbor_finder(full_neighbor_finder)
	val_acc, val_ap, val_auc = \
		eval_edge_prediction_auc(model, val_rand_sampler, val_data)
	acc.append(val_acc)
	ap.append(val_ap)
	auc.append(val_auc)
	logger.debug("In validation, acc = {}, ap = {}, auc = {}".format(val_acc, val_ap, val_auc))
	val_memory_backup = model.back_up_memory()
	model.restore_memory(train_memory_backup)
	val_n_acc, val_n_ap, val_n_auc = \
		eval_edge_prediction_auc(model, nn_val_rand_sampler, new_node_val_data)
	nacc.append(val_n_acc)
	nap.append(val_n_ap)
	nauc.append(val_n_auc)
	# ==================================================================================================
	logger.debug("In validation, nacc = {}, nap = {}, nauc = {}".format(val_n_acc, val_n_ap, val_n_auc))
	# ==================================================================================================
	model.restore_memory(val_memory_backup)
	path = get_checkpoint_path(ModelName, time_now, e)
	check_path.append(path)
	if early_stopper.early_stop_check(val_ap):
		logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
		logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
		best_model_path = get_checkpoint_path(ModelName, time_now, early_stopper.best_epoch)
		model = torch.load(best_model_path)
		logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
		model.eval()
		break
	else:
		torch.save(model, path)

memory_backup = model.back_up_memory()
model.eval()
model.set_neighbor_finder(full_neighbor_finder)
test_acc, test_ap, test_auc = \
	eval_edge_prediction_auc(model, test_rand_sampler, test_data)
model.restore_memory(memory_backup)
test_n_acc, test_n_ap, test_n_auc = \
	eval_edge_prediction_auc(model, nn_test_rand_sampler, new_node_test_data)
logger.info("in test, acc = {}, ap = {}, auc = {}"
			.format(test_acc, test_ap, test_auc))
logger.info("in new node test, acc = {}, ap = {}, auc = {}"
			.format(test_n_acc, test_n_ap, test_n_auc))
print("in test, acc = {}, ap = {}, auc = {}".format(test_acc, test_ap, test_auc))
print("in new node test, acc = {}, ap = {}, auc = {}".format(test_acc, test_n_ap, test_n_auc))
f.write("test_ap = {:.4f}, auc = {:.4f} ,new node test = {:.4f}, auc = {:.4f}".format(test_ap, test_auc, test_n_ap, test_n_auc))
f.write("\n")
