import os
import numpy as np
import shutil

# faceinfo_file = '/data3/ossdata/mz/wb_down_yolov7-faceinfo.txt'
# # faceinfo_file = '/data3/ossdata/mz/xhs_down_yolov7-faceinfo.txt'
# # faceinfo_file = '/data3/ossdata/mz/dy_down_yolov7-faceinfo.txt'

# save_folder = '/data3/ossdata/mz/image_faceinfo'
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)

# facedetinfo = open(faceinfo_file, 'r')
# infolists = facedetinfo.readlines()
# facedetinfo.close()
# print('len(infolists)==========', len(infolists))

# idx=0
# for info_line in infolists:
#     # print(info_line)
#     faceinfo = info_line.split('\n')[0]
#     filename = faceinfo.split(',')[0]
#     ext_flag = filename.endswith(".jpg") or filename.endswith('.jpeg')
#     if not ext_flag:
#         continue
#     idx+=1
#     if idx%100 == 0:
#         print('processed {} images'.format(idx))
    
#     savename = filename.split('.jp')[0]+'.txt'
#     savepath = os.path.join(save_folder, savename)
#     fopen = open(savepath, 'w')
#     fopen.write(info_line)
#     fopen.close()

# print('file_num=========', idx)
# print(f'finished !')




# cluster_file = '/data2/ossdata/mz/dy_wb_xhs_alignfacefeat_ms1mv3_r50_pred_filter.txt'
# save_folder = '/data3/ossdata/mz/image_clsterinfo_ms1mv3_filter'
# cluster_file = '/data2/ossdata/mz/dy_wb_xhs_alignfacefeat_ms1mv3_r50_pred_filter_test.txt'
# save_folder = '/data3/ossdata/mz/image_clsterinfo_ms1mv3_filter_test'
# filename_list = '/data2/ossdata/mz/dy_wb_xhs_alignface.txt'

# cluster_file = '/data2/ossdata/mz/dy_wb_xhs_alignfacefeat_ms1mv3_r50_pred_filter2.txt'
# save_folder = '/data3/ossdata/mz/image_clsterinfo_ms1mv3_filter2'
# filename_list = '/data2/ossdata/mz/dy_wb_xhs_alignface_filter2.txt'

cluster_file = '/data2/ossdata/mz/dy_wb_xhs_alignfacefeat_ms1mv3_r50_pred_filter3.txt'
save_folder = '/data3/ossdata/mz/image_clsterinfo_ms1mv3_filter3'
filename_list = '/data2/ossdata/mz/dy_wb_xhs_alignface_filter3.txt'


filename_list_read = open(filename_list, 'r')
filename_lines = filename_list_read.readlines()
filename_list_read.close()

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

cluster_result_read = open(cluster_file, 'r')
cluster_result_lines = cluster_result_read.readlines()
cluster_result_read.close()

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
        print('process {} images'.format(line_num))

    # filename = filename.split(' ')[1]
    filename = filename.split('\n')[0]
    filename = filename.split('./')[1]
    # print('idx_str, filename========', idx_str, filename)
    savename = filename.split('.jp')[0]+'.txt'
    savepath = os.path.join(save_folder, savename)
    fopen = open(savepath, 'w')
    fopen.write(filename+' '+idx_str+'\n')
    fopen.close()
    

