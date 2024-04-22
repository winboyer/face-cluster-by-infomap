
import numpy as np
from tqdm import tqdm
import infomap
import time
from multiprocessing.dummy import Pool as Threadpool
from multiprocessing import Pool
import multiprocessing as mp
import os
from utils import Timer
from evaluation import evaluate, accuracy

import sklearn

def l2norm(vec):
    """
    归一化
    :param vec: 
    :return: 
    """
#     vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    vec1 = vec/np.linalg.norm(vec, axis=1).reshape(-1, 1)
#     vec2 = sklearn.preprocessing.normalize(vec)          #上下的归一化方式基本没有区别
#     vec3 = vec/np.sqrt(np.sum(vec**2, -1, keepdims=True))  #与vec1的归一化方式结果一致
    
# #     verification11 test
#     feat0,feat1,feat2,feat3,feat4,feat5 = vec1[0],vec1[1],vec1[12],vec2[0],vec2[1],vec2[12]
#     similarity_score1,similarity_score2,similarity_score3,similarity_score4 = np.sum(feat1 * feat2, -1), np.sum(feat1 * feat0, -1), np.sum(feat4 * feat5, -1), np.sum(feat4 * feat3, -1)
#     sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
#     print('similarity_score========', similarity_score1, similarity_score1.flatten(), similarity_score2, similarity_score2.flatten())
#     print('similarity_score========', similarity_score3, similarity_score3.flatten(), similarity_score4, similarity_score4.flatten())

# #     verification1N test
#     feat1 = vec1[1]
#     similarity = np.dot(feat1, vec1.T)
#     print('len(similarity)=========', len(similarity))
# #     print('similarity=========', similarity)
#     top_inds = np.argsort(-similarity)
#     print('top_inds=========', top_inds)
    
# #     verificationMN test
#     feat2 = [vec1[0], vec1[1439], vec1[1580], vec1[740]]
#     feat2 = np.array(feat2)
    
# #     feat3 = [vec1[1580]]
#     feat3 = [vec1[0]]
#     feat3 = np.array(feat3)
# #     similarity = np.dot(feat2, vec1.T)
#     similarity = np.dot(feat2, feat3.T)
#     top_inds = np.argsort(-similarity)
#     print('similarity.shape, top_inds.shape, similarity==========', similarity.shape, top_inds.shape, similarity)
    
    return vec1


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def read_meta(fn_meta, start_pos=0, verbose=True):
    """
    idx2lb：每一个顶点对应一个类
    lb2idxs：每个类对应一个id
    """
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


class knn():
    def __init__(self, feats, k, index_path='', verbose=True):
        pass

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return th_nbrs, th_dists

    def get_knns(self, th=None):
        if th is None or th <= 0.:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer('filter edges by th {} (CPU={})'.format(th, nproc),
                   self.verbose):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns


class knn_faiss(knn):
    """
    内积暴力循环
    归一化特征的内积等价于余弦相似度
    """
    def __init__(self,
                 feats,
                 k,
                 index_path='',
                 knn_method='faiss-cpu',
                 verbose=True):
        self.verbose = verbose
        import faiss
        with Timer('[{}] build index {}'.format(knn_method, k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                print('[{}] read knns from {}'.format(knn_method, knn_ofn))
                self.knns = np.load(knn_ofn)['data']
            else:
                feats = feats.astype('float32')
                size, dim = feats.shape
                if knn_method == 'faiss-gpu':
                    import math
                    i = math.ceil(size/1000000)
                    if i > 1:
                        i = (i-1)*4
                    # res = faiss.StandardGpuResources()
                    # res.setTempMemory(i * 1024 * 1024 * 1024)
                    # index = faiss.GpuIndexFlatIP(res, dim)
                    import torch
                    device_count=torch.cuda.device_count() 
                    gpu_devices = device_count
                    print('device_count, gpu_devices==========', device_count, gpu_devices)
                    res = [faiss.StandardGpuResources() for i in range(gpu_devices)]
                    flat_config = []
                    for i in range(gpu_devices):
                        cfg = faiss.GpuIndexFlatConfig()
                        cfg.device = i
                        flat_config.append(cfg)
                    if gpu_devices == 1:
                        index = faiss.GpuIndexFlatIP(res[0], dim, flat_config[0])
                    else:
                        indexes = [faiss.GpuIndexFlatIP(res[i], dim, flat_config[i]) for i in range(gpu_devices)]
                        index = faiss.IndexReplicas()
                        for sub_index in indexes:
                            index.addIndex(sub_index)                    
                else:
                    index = faiss.IndexFlatIP(dim)
                index.add(feats)
        with Timer('[{}] query topk {}'.format(knn_method, k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                pass
            else:
                sims, nbrs = index.search(feats, k=k)
#                 print('sims, nbrs===========', sims, nbrs)
                # torch.cuda.empty_cache()
                self.knns = [(np.array(nbr, dtype=np.int32),
                              1 - np.array(sim, dtype=np.float32))
                             for nbr, sim in zip(nbrs, sims)]


def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs


# 构造边
def get_links(single, links, nbrs, dists):
    for i in tqdm(range(nbrs.shape[0])):
        count = 0
        for j in range(0, len(nbrs[i])):
            # 排除本身节点
            if i == nbrs[i][j]:
                pass
            elif dists[i][j] <= 1 - min_sim:
                count += 1
                links[(i, nbrs[i][j])] = float(1 - dists[i][j])
            else:
                break
        # 统计孤立点
        if count == 0:
            single.append(i)
    return single, links


def cluster_by_infomap(nbrs, dists, pred_label_path, save_result=False):
    """
    基于infomap的聚类
    :param nbrs: 
    :param dists: 
    :param pred_label_path: 
    :return: 
    """
    single = []
    links = {}
    with Timer('get links', verbose=True):
        single, links = get_links(single=single, links=links, nbrs=nbrs, dists=dists)

    infomapWrapper = infomap.Infomap("--two-level --directed")
#     infomapWrapper = infomap.Infomap("--directed")
    for (i, j), sim in tqdm(links.items()):
        _ = infomapWrapper.addLink(int(i), int(j), sim)

    # 聚类运算
    infomapWrapper.run()

    label2idx = {}
    idx2label = {}

    # 聚类结果统计
    count=0
    for node in infomapWrapper.iterTree():
        # node.physicalId 特征向量的编号
        # node.moduleIndex() 聚类的编号
        count+=1
#         idx2label[node.physicalId] = node.moduleIndex()
        if node.moduleIndex() not in label2idx:
            label2idx[node.moduleIndex()] = []
        label2idx[node.moduleIndex()].append(node.physicalId)

    node_count = 0
    for k, v in label2idx.items():
#         print('k,v==========',k,v)
        if k == 0:
            node_count += len(v[2:])
            label2idx[k] = v[2:]
#             print(k, len(v[2:]), v[2:])
            for v_value in v[2:]:
                idx2label[v_value] = k
        else:
            node_count += len(v[1:])
            label2idx[k] = v[1:]
            # print(k, v[1:])
            for v_value in v[1:]:
                idx2label[v_value] = k

    # print(node_count)
    # 孤立点个数
    print("孤立点数：{}".format(len(single)))

    keys_len = len(list(label2idx.keys()))

    # 孤立点放入到结果中
    for single_node in single:
        idx2label[single_node] = keys_len
        label2idx[keys_len] = [single_node]
        keys_len += 1

    print("总类别数：{}".format(keys_len))

    idx_len = len(list(idx2label.keys()))
    print("总节点数：{}".format(idx_len))

    # 保存结果
    if save_result:
        with open(pred_label_path, 'w') as of:
            for idx in range(idx_len):
                of.write(str(idx2label[idx]) + '\n')

    if label_path is not None:
        pred_labels = intdict2ndarray(idx2label)
        true_lb2idxs, true_idx2lb = read_meta(label_path)
        gt_labels = intdict2ndarray(true_idx2lb)
        for metric in metrics:
            evaluate(gt_labels, pred_labels, metric)


def get_dist_nbr(feature_path, k=80, knn_method='faiss-cpu'):
    features = np.fromfile(feature_path, dtype=np.float32)
    # print(features, features.shape)
    features = features.reshape(-1, 256)
    # print(features.shape)
    features = l2norm(features)

    index = knn_faiss(feats=features, k=k, knn_method=knn_method)
    knns = index.get_knns()
#     knns = index.get_knns(th=min_sim)
    dists, nbrs = knns2ordered_nbrs(knns)
    return dists, nbrs

# knn_method = 'faiss-cpu'
knn_method = 'faiss-gpu'
metrics = ['pairwise', 'bcubed', 'nmi']
min_sim = 0.58
# min_sim = 0.5
# min_sim = 0.65
# k = 50
# k = 250
# k = 800
k = 1000

# true_label

# label_path = '/root/jinyfeng/projects/learn-to-cluster/data/labels/part1_test.meta'
# feature_path = '/root/jinyfeng/projects/learn-to-cluster/data/features/part1_test.bin'
# pred_label_path = './part1_test_predict.txt'


# label_path=None
# feature_path='/root/jinyfeng/datas/feat_face_ms1mv3_r50_align.bin'
# pred_label_path = './0801_facefeat_align_predict_t0.5.txt'


label_path = None
feature_path='/data2/ossdata/mz/dy_wb_xhs_alignfacefeat_ms1mv3_r50.bin'
pred_label_path='/data2/ossdata/mz/dy_wb_xhs_alignfacefeat_ms1mv3_r50_pred_v3.txt'

# feature_path='/data2/ossdata/mz/dy_wb_xhs_alignfacefeat_glint360k_r50.bin'
# pred_label_path='/data2/ossdata/mz/dy_wb_xhs_alignfacefeat_glint360k_r50_pred_v3.txt'

# feature_path = '/root/jinyfeng/datas/sensoro/sensoro_person_info_yolo7face.bin'
# pred_label_path = './sensoro_person_info_yolo7face.txt'
# feature_path = '/root/jinyfeng/datas/sensoro/sensoro_person_info_yolo7face_v2.bin'
# pred_label_path = './sensoro_person_info_yolo7face_v2.txt'

# feature_path = '/root/jinyfeng/datas/sensoro/sensoro_quality_face_alignface_yolov7face.bin'
# pred_label_path = './sensoro_quality_face_yolo7face.txt'

# # feature_path='/root/jinyfeng/datas/sensoro/facefeat_v1.bin'
# feature_path='/root/jinyfeng/datas/sensoro/facefeat_align_v1.bin'
# # feature_path = '/root/jinyfeng/datas/feat_face_ms1mv3_r50_v1.bin'
# # feature_path = '/root/jinyfeng/datas/feat_face_ms1mv3_r50_align.bin'
# # pred_label_path = './facefeat_v1_predict.txt'
# pred_label_path = './facefeat_align_v1_predict.txt'
# # pred_label_path = './feat_face_predict.txt'
# # pred_label_path = './feat_face_align_predict_t0.5.txt'



with Timer('All face cluster step'):
    dists, nbrs = get_dist_nbr(feature_path=feature_path, k=k, knn_method=knn_method)
#     print(dists.shape, nbrs.shape)
#     print(dists, nbrs)
#     cluster_by_infomap(nbrs, dists, pred_label_path, save_result=False)
    cluster_by_infomap(nbrs, dists, pred_label_path, save_result=True)


