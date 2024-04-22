import os
import numpy as np
import shutil

# cluster_file = './data/work_dir/cfg_test_gcne_facefeat/feat_face_ms1mv3_r50_gcne_k_160_th_0.0_ig_0.8/tau_0.6_pred_labels.txt'
# filename_list = '/root/jinyfeng/projects/insightface/recognition/arcface_torch/face_feat0.log'

# cluster_file = './data/work_dir/cfg_test_gcne_facefeat/feat_face_ms1mv3_r50_align_gcne_k_160_th_0.0_ig_0.8/tau_0.6_pred_labels.txt'


# # cluster_file = './feat_face_predict.txt'
# cluster_file = './0801_facefeat_align_predict_t0.5.txt'
# # cluster_file = './facefeat_v0_predict.txt'
# # cluster_file = './facefeat_v1_predict.txt'
# # cluster_file = './facefeat_align_v1_predict.txt'

# # filename_list = '/root/jinyfeng/projects/insightface/recognition/arcface_torch/face_feat.log'
# # filename_list = '/root/jinyfeng/projects/insightface/recognition/arcface_torch/alignface_feat.log'
# filename_list = '/root/jinyfeng/projects/insightface/recognition/arcface_torch/20230801_v2_alignface_feat.log'
# # filename_list = '/root/jinyfeng/projects/insightface/recognition/arcface_torch/20230823face_feat.log'
# # filename_list = '/root/jinyfeng/projects/insightface/recognition/arcface_torch/20230823face_feat_v1.log'
# # filename_list = '/root/jinyfeng/projects/insightface/recognition/arcface_torch/20230823facealign_feat_v1.log'

# img_folder = '/root/jinyfeng/datas/20230801_faces_v2_2619id'
# # img_folder = '/root/jinyfeng/datas/sensoro/20230823_faces'

# # result_path = './20230801_faces_v2_2619id_cluster_result'
# result_path = './20230801_faces_v2_2619id_cluster_align_result'
# # result_path = './20230823_faces_cluster_result'
# # result_path = './20230823_faces_cluster_result_align'

# cluster_file = './sensoro_person_info_yolo7face_v2.txt'
# filename_list = '/root/jinyfeng/projects/insightface/recognition/arcface_torch/sensoro_person_info_yolov7face_feat.log'
# img_folder = '/root/jinyfeng/datas/sensoro/sensoro_person_info'
# result_path = './sensoro_person_info_cluster_v2'

cluster_file = './sensoro_quality_face_yolo7face_sim0.5.txt'
filename_list = '/root/jinyfeng/projects/insightface/recognition/arcface_torch/sensoro_quality_face.log'
img_folder = '/root/jinyfeng/datas/sensoro/sensoro_quality_face'
result_path = './sensoro_quality_face_cluster_sim0.5'

cluster_result_read = open(cluster_file, 'r')
cluster_result_lines = cluster_result_read.readlines()
filename_list_read = open(filename_list, 'r')
filename_lines = filename_list_read.readlines()

line_num=0
for idx_str in cluster_result_lines:
    idx_str = idx_str.strip('\n')
    subfolder = os.path.join(result_path, idx_str)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    filename = filename_lines[line_num]
#     print('filename========', filename)
    sub_str = '.jpg'
    flag = (sub_str in filename)
#     print('flag=======', flag)
    while not flag:
        line_num+=1
        filename = filename_lines[line_num]
#         print('filename========', filename)
        flag = (sub_str in filename)
#         print('flag=======', flag)
    
    filename = filename.split(' ')[1]
    print('idx_str, filename========', idx_str, filename)
    
    ori_path = os.path.join(img_folder, filename)
    dst_path = os.path.join(subfolder, filename)
    shutil.copy(ori_path, dst_path)
    
    line_num+=1


cluster_result_read.close()
filename_list_read.close()
    
    

