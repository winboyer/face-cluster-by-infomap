
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
import shutil

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
#     feat2 = [vec1[1], vec1[13]]
#     feat2 = np.array(feat2)
#     similarity = np.dot(feat2, vec1.T)
#     top_inds = np.argsort(-similarity)
#     print('similarity.shape, top_inds.shape==========', similarity.shape, top_inds.shape)
    
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





def cluster_by_infomap(nbrs, dists, pred_label_path, save_result=False):
    """
    基于infomap的聚类
    :param nbrs: 
    :param dists: 
    :param pred_label_path: 
    :return: 
    """
    
    label2idx = {}
    idx2label = {}

    node_count = 0
    for k, v in label2idx.items():
        if k == 0:
            node_count += len(v[2:])
            label2idx[k] = v[2:]
            # print(k, v[2:])
        else:
            node_count += len(v[1:])
            label2idx[k] = v[1:]
            # print(k, v[1:])

    # print(node_count)
    # 孤立点个数
    print("孤立点数：{}".format(len(single)))

    keys_len = len(list(label2idx.keys()))
    # print(keys_len)

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


gallery_imgfolder = '/root/jinyfeng/datas/sensoro/sensoro_person_info'
gallery_txtfile = '/root/jinyfeng/datas/sensoro/sensoro_person_info.txt'
gallery_featfile = '/root/jinyfeng/datas/sensoro/sensoro_person_info_yolo7face_v2.bin'

query_imgfolder = '/root/jinyfeng/datas/sensoro/sensoro_quality_face'
query_featfile = '/root/jinyfeng/datas/sensoro/sensoro_quality_face_alignface_yolov7face.bin'
query_txtfile = '/root/jinyfeng/datas/sensoro/sensoro_quality_face.txt'
query_pred_idx = '/root/jinyfeng/projects/face-cluster-by-infomap/sensoro_quality_face_yolo7face_sim0.5.txt'

savefolder = '/root/jinyfeng/datas/sensoro/sensoro_person_info_faceVer'
savefile = '/root/jinyfeng/datas/sensoro/sensoro_person_info_faceVer.txt'

savefile_w = open(savefile, 'w')

gallery_list_read = open(gallery_txtfile, 'r')
gallery_list = gallery_list_read.readlines()
gallery_list_read.close()
# print(len(gallery_list))
query_list_read = open(query_txtfile, 'r')
query_list = query_list_read.readlines()
query_list_read.close()
# print(len(query_list))
query_pred_idx_list_read = open(query_pred_idx, 'r')
query_pred_idx_list = query_pred_idx_list_read.readlines()
query_pred_idx_list_read.close()

gallery_feats = np.fromfile(gallery_featfile, dtype=np.float32)
gallery_feats = gallery_feats.reshape(-1, 256)
gallery_feats = l2norm(gallery_feats)
print(len(gallery_feats))
query_feats = np.fromfile(query_featfile, dtype=np.float32)
query_feats = query_feats.reshape(-1, 256)
query_feats = l2norm(query_feats)
print(len(query_feats))

# # verification1N test
# for feat in query_feats:
#     similarity = np.dot(feat, gallery_feats.T)
#     print('len(similarity)=========', len(similarity))
# #     print('similarity=========', similarity)
#     top_inds = np.argsort(-similarity)
#     print('top_inds=========', top_inds, similarity[top_inds])
    
# verificationMN test
similarity = np.dot(query_feats, gallery_feats.T)
top_inds = np.argsort(-similarity)
print('similarity.shape, top_inds.shape==========', similarity.shape, top_inds.shape, top_inds)

for idx in range(top_inds.shape[0]):
    # top1
#     top_ind_pred = top_inds[idx]
    top_ind_pred = top_inds[idx, 0]
    sim_score = similarity[idx, top_ind_pred]
    if sim_score<0.5:
        continue
    print(top_ind_pred, sim_score)
    
#     # top5
#     top_ind_pred = top_inds[pred_idx, 0:5]
    
    person_name = gallery_list[top_ind_pred].split('\n')[0]
#     print(person_name)
    subfolder = os.path.join(savefolder, person_name)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    ori_path = os.path.join(gallery_imgfolder, person_name)
    dst_path = os.path.join(subfolder, person_name)
    shutil.copy(ori_path, dst_path)
    
    query_name = query_list[idx].split('\n')[0]
#     print(query_name)
    ori_path = os.path.join(query_imgfolder, query_name)
    dst_path = os.path.join(subfolder, query_name)
    shutil.copy(ori_path, dst_path)
    
print('finished !')    
    
