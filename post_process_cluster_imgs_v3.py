import os
import numpy as np
import shutil


# cluster_file = './sensoro_quality_face_yolo7face_sim0.5.txt'
# filename_list = '/root/jinyfeng/datas/sensoro/sensoro_quality_face.txt'
# img_folder = '/root/jinyfeng/datas/sensoro/sensoro_quality_face'
# result_path = './sensoro_quality_face_cluster_sim0.5'

# cluster_file = '/data2/ossdata/mz/dy_wb_xhs_alignfacefeat_ms1mv3_r50_pred_filter.txt'
# filename_list = '/data2/ossdata/mz/dy_wb_xhs_alignface.txt'
# img_folder = '/data2/ossdata/mz/dy_wb_xhs_faceimgs'
# result_path = '/data2/ossdata/mz/dy_wb_xhs_ms1mv3_r50_pred_filter'

cluster_file = '/data2/ossdata/mz/dy_wb_xhs_alignfacefeat_glint360k_r50_pred_filter.txt'
filename_list = '/data2/ossdata/mz/dy_wb_xhs_alignface.txt'
img_folder = '/data2/ossdata/mz/dy_wb_xhs_faceimgs'
result_path = '/data2/ossdata/mz/dy_wb_xhs_glint360k_r50_pred_filter'

if not os.path.exists(result_path):
    os.makedirs(result_path)

cluster_result_read = open(cluster_file, 'r')
cluster_result_lines = cluster_result_read.readlines()
filename_list_read = open(filename_list, 'r')
filename_lines = filename_list_read.readlines()

line_num=0
for idx_str in cluster_result_lines:
    idx_str = idx_str.strip('\n')
    filename = filename_lines[line_num]
    # print('filename========', filename)
    flag = ('.jpg' in filename or '.jpeg' in filename)
#     print('flag=======', flag)
    while not flag:
        print('filename========', filename, 'is not jpg/jpeg file')
        line_num+=1
        filename = filename_lines[line_num]
        # print('filename========', filename)
        flag = ('.jpg' in filename or '.jpeg' in filename)
#         print('flag=======', flag)
    line_num+=1
    if line_num%100==0:
        print('post {} images'.format(line_num))

    if int(idx_str)>20:
        # print('idx_str========', idx_str)
        continue

    subfolder = os.path.join(result_path, idx_str)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    
    # filename = filename.split(' ')[1]
    filename = filename.split('\n')[0]
    # print('idx_str, filename========', idx_str, filename)
    
    ori_path = os.path.join(img_folder, filename)
    dst_path = os.path.join(subfolder, filename)
    shutil.copy(ori_path, dst_path)
    
    


cluster_result_read.close()
filename_list_read.close()
    
    

