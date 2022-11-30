import matplotlib.pyplot as plt
from matplotlib import rcParams
import cv2
config = {
    "font.family":'serif',
    "font.size": 12,
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)

import numpy as np
plt.rcParams['savefig.dpi']=600
img_bgr_data =cv2.imread('result/images0.png')
plt.figure(figsize=(15,5))
# 坐标轴的刻度设置向内(in)或向外(out)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

#B
ax1=plt.subplot(131)
#ax1.set_xlim([0,255])#重点是对获取到的axes对象进行操作
ax1.hist(img_bgr_data[:,:,0].ravel(),bins=50,color='b')  #bins设置连续的边界值，即直方图的分布区间[0,10],[10,20]...
ax1.set_title('B通道', fontproperties='SimSun', size=14)
ax1.set_xlabel("亮度值", fontproperties='SimSun', size=12)
ax1.set_ylabel("像素个数",fontproperties='SimSun', size=12)  # 给纵坐标说明
plt.tight_layout()
#G
ax2 =plt.subplot(132)
#ax2.set_xlim([0,255])#重点是对获取到的axes对象进行操作
ax2.hist(img_bgr_data[:,:,1].ravel(),bins=50,color='g')
ax2.set_title('G通道', fontproperties='SimSun', size=14)
ax2.set_xlabel("亮度值", fontproperties='SimSun', size=12)
ax2.set_ylabel("像素个数",fontproperties='SimSun', size=12)  # 给纵坐标说明
#R
ax3 =plt.subplot(133)
#ax3.set_xlim([0,255])#重点是对获取到的axes对象进行操作
ax3.hist(img_bgr_data[:,:,2].ravel(),bins=50,color='r')
ax3.set_title('R通道', fontproperties='SimSun', size=14)
ax3.set_xlabel("亮度值", fontproperties='SimSun', size=12)
ax3.set_ylabel("像素个数",fontproperties='SimSun', size=12)  # 给纵坐标说明

plt.tight_layout()
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
#plt.legend()  # 显示不同颜色的意义
plt.savefig(r'111.png',dpi=600,format='png',transparent=True)
plt.show()

