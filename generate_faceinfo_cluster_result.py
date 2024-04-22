import os
import numpy as np
import shutil
import json


# {
#     "faceId": "dy_down_2022-01-30_7058909278043770144_28",
#     "personId": "2893147982341",
#     "source": "douyin",
#     "sourceMediaId": "123456789",
#     "originImageUrl": "https://s3.xn1a.stor.xn.sensoro.vip/aidata/image/dy_down_2016-10-25_7031377256194886924_1329.jpg",
#     "originImageNo": 13,
#     "smallImageInfo":"0,0,12,12,0.34,1,2,3,4,5,1,2,3,4,5"
# }



faceinfo_folder = '/data3/ossdata/mz/image_faceinfo'


# filename_list = '/data2/ossdata/mz/dy_wb_xhs_alignface.txt'
# cluster_folder = '/data3/ossdata/mz/image_clsterinfo_ms1mv3_filter'
# savefile_path = '/data3/ossdata/mz/image_clsterinfo_ms1mv3_filter_jsonfile.txt'

# filename_list = '/data2/ossdata/mz/dy_wb_xhs_alignface.txt'
# cluster_folder = '/data3/ossdata/mz/image_clsterinfo_ms1mv3_filter_test'
# savefile_path = '/data3/ossdata/mz/image_clsterinfo_ms1mv3_filter1_jsonfile.txt'

# filename_list = '/data2/ossdata/mz/dy_wb_xhs_alignface_filter2.txt'
# cluster_folder = '/data3/ossdata/mz/image_clsterinfo_ms1mv3_filter2'
# savefile_path = '/data3/ossdata/mz/image_clsterinfo_ms1mv3_filter2_jsonfile.txt'

filename_list = '/data2/ossdata/mz/dy_wb_xhs_alignface_filter3.txt'
cluster_folder = '/data3/ossdata/mz/image_clsterinfo_ms1mv3_filter3'
savefile_path = '/data3/ossdata/mz/image_clsterinfo_ms1mv3_filter3_jsonfile.txt'

filename_list_read = open(filename_list, 'r')
filename_lines = filename_list_read.readlines()
filename_list_read.close()

fopen = open(savefile_path, 'w')

idx=0
root_urlpath = 'https://s3.xn1a.stor.xn.sensoro.vip/aidata/image/'
for filename in filename_lines:
    filename = filename.split('\n')[0]
    filename = filename.split('./')[1]
    # print('filename========', filename)
    ext_flag = filename.endswith(".jpg") or filename.endswith('.jpeg')
    if not ext_flag:
        continue
    idx+=1
    if idx%100 == 0:
        print('processed {} images'.format(idx))

    faceId = filename.split('.jp')[0]
    source = filename.split('_')[0]
    sourceMediaId = filename.split('_')[3]
    if source == 'dy':
        source='douyin'
    if source == 'wb':
        source='weibo'
    if source == 'xhs':
        source='xiaohongshu'
    originImageUrl = root_urlpath+filename
    originImageNo = faceId.split('_')[-1]
    # print(source, sourceMediaId, originImageNo, originImageUrl)
    
    cluster_file = os.path.join(cluster_folder, faceId+'.txt')
    cluster_file_read = open(cluster_file, 'r')
    linestr = cluster_file_read.readline()
    # print(linestr)
    cluster_file_read.close()
    personId = linestr.split(' ')[1].split('\n')[0]
    # print(faceId, personId)

    faceinfo_file = os.path.join(faceinfo_folder, faceId+'.txt')
    faceinfo_file_read = open(faceinfo_file, 'r')
    linestr = faceinfo_file_read.readline()
    faceinfo_file_read.close()

    linestr = linestr.split('\n')[0]
    smallImageInfo = linestr[linestr.find(',')+1:]
    # print(smallImageInfo, linestr)

    line_json=json.dumps({"faceId": faceId, 
                            "personId": int(personId), 
                            "source": source,
                            "sourceMediaId": sourceMediaId,
                            "originImageUrl": originImageUrl,
                            "originImageNo": int(originImageNo),
                            "smallImageInfo": smallImageInfo})
    fopen.write(line_json+'\n')

fopen.close()

print('file_num=========', idx)
print(f'finished !')




    

