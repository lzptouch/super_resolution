
import numpy as np
import matplotlib.pyplot as plt
from math import *
import numpy as np


  # 保存为.npy格式

a=np.load("net363.npy")
net363=a.tolist()
print(max(net363))


b=np.load("net363__.npy")
net363__=a.tolist()

print(max(net363__))
y_ticks = np.arange(30, 32.5, 0.5)
x_ticks = np.arange(0, 9, 1)
#对比范围和名称的区别
plt.xticks(x_ticks)
plt.yticks(y_ticks)
# 坐标轴名称
plt.title('PSNR with Different Number of Modules',fontsize=15)
plt.xlabel('Numbers of Modules',fontsize=15)
plt.ylabel('PSNR(dB)',fontsize=15)

# 准备绘制数据
x1 = range(1,len(a)+1)
y1 = net363

x2 = range(1,len(a)+1)
y2 = net363__[:len(a)+1]

plt.plot(x1, y1, "gold",   label="normal weight")
plt.plot(x2, y2, "black",   label="shared weight")
# 显示图例
plt.legend(loc="lower right")
# 保存图片
plt.savefig("Modules.jpg",dpi = 1024)
plt.savefig('img_test.eps', dpi=1024)
plt.show()