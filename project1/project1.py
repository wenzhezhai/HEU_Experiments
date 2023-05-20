# 导入所有需要的包并初始化
import os  # 导入操作系统模块，用于处理文件和目录
import torch  # 导入PyTorch模块，用于深度学习
import numpy as np  # 导入NumPy模块，用于科学计算
import seaborn as sns  # 导入Seaborn模块，用于数据可视化
import skimage.io as skio  # 导入skimage.io模块，用于图像输入输出
from skimage.draw import line_aa  # 从skimage.draw模块导入line_aa函数，用于绘制抗锯齿线条
from scipy import stats  # 从scipy模块导入stats子模块，用于统计分析
# 从torch.nn.functional模块导入interpolate函数，用于对张量进行插值
from torch.nn.functional import interpolate
from tqdm import tqdm, trange  # 从tqdm模块导入tqdm和trange函数，用于显示进度条
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
# %matplotlib inline  # 设置matplotlib在notebook中内嵌显示图像
img_dir = 'data'  # 定义一个变量img_dir，存储图像数据的目录名

### 1.Plot the histogram H(z) for the diﬀerence against the horizontal axis z ∈ [−31, +31].


# 初始化
# 定义集合A
nat_imgs = []  # 创建一个空列表，用于存储自然图像
for fname in os.listdir(img_dir):  # 遍历图像数据目录中的所有文件名
    if 'natural' in fname:  # 如果文件名中包含'natural'，说明是自然图像
        # 用skio.imread函数读取图像文件，并转换为灰度图
        img = skio.imread(os.path.join(img_dir, fname), as_gray=True)
        img = (img*32-0.5).astype(int)  # 对图像进行缩放和偏移，使其像素值在0到31之间，并转换为整数类型
        nat_imgs.append(img)  # 将图像添加到列表中

# 定义集合B
# 用skio.imread函数读取非自然图像文件，并转换为灰度图
non_nat_img = skio.imread(os.path.join(
    img_dir, 'non_nat_img.jpg'), as_gray=True)
# 对图像进行缩放和偏移，使其像素值在0到31之间，并转换为整数类型
non_nat_img = (non_nat_img*32-0.5).astype(int)

# 定义集合C
# 用np.random.randint函数生成一个随机整数矩阵，大小为64x64，元素值在0到31之间
syn_img = np.random.randint(low=0, high=32, size=(64, 64))

# 水平梯度


def compute_grad(img: np.array) -> np.array:  # 定义一个函数，输入参数为一个numpy数组，输出也是一个numpy数组
    return img[:, 1:] - img[:, :-1]  # 返回输入图像的水平梯度，即每个像素与其右边相邻像素的差值


# 定义一个函数，接受四个参数：自然图像列表，非自然图像，合成图像和是否显示直方图的布尔值
def vis_hist(nat_imgs, non_nat_img, syn_img, show_hist=True):
    # 对自然图像列表中的每个图像计算梯度，并将梯度展平为一维数组，然后将所有数组拼接成一个大数组
    z_A = np.concatenate([compute_grad(img).flatten() for img in nat_imgs])
    z_B = compute_grad(non_nat_img)  # 对非自然图像计算梯度
    z_C = compute_grad(syn_img)  # 对合成图像计算梯度

    # H(z)
    plt.subplot(2, 3, 1)  # 在一个2行3列的子图网格中，选择第一个位置
    plt.title('Natural')  # 设置子图的标题为“自然”
    plt.xlabel('z')  # 设置子图的x轴标签为“z”
    if show_hist:  # 如果show_hist为真
        plt.hist(z_A, bins=32, density=True)  # 绘制z_A的直方图，分成32个区间，归一化为密度函数
    else:  # 否则
        # 计算z_A的直方图和区间范围，分成32个区间，归一化为密度函数
        h, z_range = np.histogram(z_A, bins=32, density=True)
        z_range = (z_range[:-1] + z_range[1:]) / 2  # 计算每个区间的中点
        plt.plot(z_range, h)  # 绘制折线图，x轴为区间中点，y轴为直方图值

    plt.subplot(2, 3, 2)  # 在一个2行3列的子图网格中，选择第二个位置
    plt.title('Non-natural')  # 设置子图的标题为“非自然”
    plt.xlabel('z')  # 设置子图的x轴标签为“z”
    if show_hist:  # 如果show_hist为真
        plt.hist(z_B, bins=32, density=True)  # 绘制z_B的直方图，分成32个区间，归一化为密度函数
    else:  # 否则
        # 计算z_B的直方图和区间范围，分成32个区间，归一化为密度函数
        h, z_range = np.histogram(z_B, bins=32, density=True)
        z_range = (z_range[:-1] + z_range[1:]) / 2  # 计算每个区间的中点
        plt.plot(z_range, h)  # 绘制折线图，x轴为区间中点，y轴为直方图值

    plt.subplot(2, 3, 3)  # 在一个2行3列的子图网格中，选择第三个位置
    plt.title('Synthetic')  # 设置子图的标题为“合成”
    plt.xlabel('z')  # 设置子图的x轴标签为“z”
    if show_hist:  # 如果show_hist为真
        plt.hist(z_C, bins=32, density=True)  # 绘制z_C的直方图，分成32个区间，归一化为密度函数
    else:  # 否则
        # 计算z_C的直方图和区间范围，分成32个区间，归一化为密度函数
        h, z_range = np.histogram(z_C, bins=32, density=True)
        z_range = (z_range[:-1] + z_range[1:]) / 2  # 计算每个区间的中点
        plt.plot(z_range, h)  # 绘制折线图，x轴为区间中点，y轴为直方图值

    # log H(z)
    plt.subplot(2, 3, 4)  # 在一个2行3列的子图网格中，选择第四个位置
    plt.title('Natural')  # 设置子图的标题为“自然”
    plt.xlabel('z')  # 设置子图的x轴标签为“z”
    if show_hist:  # 如果show_hist为真
        # 绘制z_A的直方图，分成32个区间，归一化为密度函数，并取对数
        plt.hist(z_A, bins=32, density=True, log=True)
    else:  # 否则
        # 计算z_A的直方图和区间范围，分成32个区间，归一化为密度函数
        h, z_range = np.histogram(z_A, bins=32, density=True)
        z_range = (z_range[:-1] + z_range[1:]) / 2  # 计算每个区间的中点
        plt.plot(z_range, h)  # 绘制折线图，x轴为区间中点，y轴为直方图值
        plt.yscale('log')  # 设置y轴为对数刻度

    plt.subplot(2, 3, 5)  # 在一个2行3列的子图网格中，选择第五个位置
    plt.title('Non-natural')  # 设置子图的标题为“非自然”
    plt.xlabel('z')  # 设置子图的x轴标签为“z”
    if show_hist:  # 如果show_hist为真
        # 绘制z_B的直方图，分成32个区间，归一化为密度函数，并取对数
        plt.hist(z_B, bins=32, density=True, log=True)
    else:  # 否则
        # 计算z_B的直方图和区间范围，分成32个区间，归一化为密度函数
        h, z_range = np.histogram(z_B, bins=32, density=True)
        z_range = (z_range[:-1] + z_range[1:]) / 2  # 计算每个区间的中点
        plt.plot(z_range, h)  # 绘制折线图，x轴为区间中点，y轴为直方图值
        plt.yscale('log')  # 设置y轴为对数刻度

    plt.subplot(2, 3, 6)  # 在一个2行3列的子图网格中，选择第六个位置
    plt.title('Synthetic')  # 设置子图的标题为“合成”
    plt.xlabel('z')  # 设置子图的x轴标签为“z”
    if show_hist:  # 如果show_hist为真
        # 绘制z_C的直方图，分成32个区间，归一化为密度函数，并取对数
        plt.hist(z_C, bins=32, density=True, log=True)
    else:  # 否则
        # 计算z_C的直方图和区间范围，分成32个区间，归一化为密度函数
        h, z_range = np.histogram(z_C, bins=32, density=True)
        z_range = (z_range[:-1] + z_range[1:]) / 2  # 计算每个区间的中点
        plt.plot(z_range, h)  # 绘制折线图，x轴为区间中点，y轴为直方图值
        plt.yscale('log')  # 设置y轴为对数刻度
    plt.subplot(2, 3, 6)  # 在一个2行3列的子图网格中，选择第六个位置
    plt.title('Synthetic')  # 设置子图的标题为“合成”
    plt.xlabel('z')  # 设置子图的x轴标签为“z”
    if show_hist:  # 如果show_hist为真
        # 绘制z_C的直方图，分成32个区间，归一化为密度函数，并取对数
        plt.hist(z_C, bins=32, density=True, log=True)
    else:  # 否则
        # 计算z_C的直方图和区间范围，分成32个区间，归一化为密度函数
        h, z_range = np.histogram(z_C, bins=32, density=True)
        z_range = (z_range[:-1] + z_range[1:]) / 2  # 计算每个区间的中点
        plt.plot(z_range, h)  # 绘制折线图，x轴为区间中点，y轴为直方图值
        plt.yscale('log')  # 设置y轴为对数刻度

    plt.tight_layout()  # 调整子图的间距，使得标题和标签不重叠

    return z_A, z_B, z_C  # 返回三种类型图像的梯度数组


# 调用vis_hist函数，传入自然图像列表，非自然图像，合成图像和显示直方图的布尔值，并接收返回值
z_A, z_B, z_C = vis_hist(nat_imgs, non_nat_img, syn_img, show_hist=True)
plt.savefig('1_histogram.png')
###2.Compute the mean, variance, and kurtosis for this histogram
# 计算z_A的均值，标准差和峰度，并分别赋值给mean_A, std_A和kurt_A
mean_A = np.mean(z_A) # 均值是所有元素的平均值
std_A = np.std(z_A) # 标准差是所有元素与均值的差的平方的平均值的平方根
kurt_A = np.mean(((z_A-mean_A)/std_A)**4) # 峰度是所有元素与均值的差除以标准差的四次方的平均值

# 计算z_B的均值，标准差和峰度，并分别赋值给mean_B, std_B和kurt_B
mean_B = np.mean(z_B)
std_B = np.std(z_B)
kurt_B = np.mean(((z_B-mean_B)/std_B)**4)

# 计算z_C的均值，标准差和峰度，并分别赋值给mean_C, std_C和kurt_C
mean_C = np.mean(z_C)
std_C = np.std(z_C)
kurt_C = np.mean(((z_C-mean_C)/std_C)**4)

# 使用格式化字符串打印A, B, C三组数据的均值，标准差和峰度，保留五位小数
print(f'A: mean={mean_A:.5f} | std={std_A:.5f} | kurt={kurt_A:.5f}')
print(f'B: mean={mean_B:.5f} | std={std_B:.5f} | kurt={kurt_B:.5f}')
print(f'C: mean={mean_C:.5f} | std={std_C:.5f} | kurt={kurt_C:.5f}')



###3.Fit this histogram to a Generalized Gaussian distribution e z/σγ| | and plot the ﬁttedcurves super-imposed against the histogram.

# 使用gennorm.fit函数对z_A, z_B, z_C三组数据进行广义正态分布的拟合，得到三个参数：形状参数gamma，位置参数loc和尺度参数sigma
# 这里我们固定位置参数为0，只返回gamma和sigma，并分别赋值给gamma_A, sigma_A等
gamma_A, _, sigma_A = stats.gennorm.fit(z_A.flatten(), floc=0)
gamma_B, _, sigma_B = stats.gennorm.fit(z_B.flatten(), floc=0)
gamma_C, _, sigma_C = stats.gennorm.fit(z_C.flatten(), floc=0)

# 定义一个数组z_range，用于表示z的取值范围，从-31到31，步长为1
z_range = np.arange(-31, 31, 1)

# 使用subplot函数创建一个一行三列的子图布局
plt.subplot(1, 3, 1)
# 在第一个子图中，设置标题为Natural，横轴标签为z
plt.title('Natural')
plt.xlabel('z')
# 使用hist函数绘制z_A的直方图，设置bins为32，density为True表示归一化频数
plt.hist(z_A, bins=32, density=True)
# 使用plot函数绘制z_A的广义正态分布的概率密度函数（PDF），使用gennorm.pdf函数计算PDF的值，传入拟合得到的参数
plt.plot(z_range, stats.gennorm.pdf(z_range, beta=gamma_A, loc=0, scale=sigma_A))

# 在第二个子图中，设置标题为Non-natural，横轴标签为z
plt.subplot(1, 3, 2)
plt.title('Non-natural')
plt.xlabel('z')
# 使用hist函数绘制z_B的直方图，设置bins为32，density为True表示归一化频数
plt.hist(z_B, bins=32, density=True)
# 使用plot函数绘制z_B的广义正态分布的概率密度函数（PDF），使用gennorm.pdf函数计算PDF的值，传入拟合得到的参数
plt.plot(z_range, stats.gennorm.pdf(z_range, beta=gamma_B, loc=0, scale=sigma_B))

# 在第三个子图中，设置标题为Synthetic，横轴标签为z
plt.subplot(1, 3, 3)
plt.title('Synthetic')
plt.xlabel('z')
# 使用hist函数绘制z_C的直方图，设置bins为32，density为True表示归一化频数
plt.hist(z_C, bins=32, density=True)
# 使用plot函数绘制z_C的广义正态分布的概率密度函数（PDF），使用gennorm.pdf函数计算PDF的值，传入拟合得到的参数
plt.plot(z_range, stats.gennorm.pdf(z_range, beta=gamma_C, loc=0, scale=sigma_C))

# 使用tight_layout函数调整子图之间的间距
plt.tight_layout()
plt.savefig('2_gaussian.png')
plt.close()
# 使用格式化字符串打印三组数据拟合得到的形状参数gamma（取负数），保留五位小数
print(f'gamma_A: {-gamma_A:.5f} | gamma_B: {-gamma_B:.5f} | gamma_C: {-gamma_C:.5f}')

###4.Plot the Gaussian distribution using the mean and the variance in step (2), and super-impose this plot with the plots in step (1) above (i.e. plot the Gaussian and its log plot, this is easy to do in python with matplotlib).


# 使用subplot函数创建一个两行三列的子图布局
plt.subplot(2, 3, 1)
# 在第一个子图中，设置标题为Natural，横轴标签为z
plt.title('Natural')
plt.xlabel('z')
# 使用hist函数绘制z_A的直方图，设置bins为32，density为True表示归一化频数
plt.hist(z_A, bins=32, density=True)
# 使用plot函数绘制z_A的正态分布的概率密度函数（PDF），使用norm.pdf函数计算PDF的值，传入均值和标准差作为参数
plt.plot(z_range, stats.norm.pdf(z_range, loc=mean_A, scale=std_A))

# 在第二个子图中，设置标题为Non-natural，横轴标签为z
plt.subplot(2, 3, 2)
plt.title('Non-natural')
plt.xlabel('z')
# 使用hist函数绘制z_B的直方图，设置bins为32，density为True表示归一化频数
plt.hist(z_B, bins=32, density=True)
# 使用plot函数绘制z_B的正态分布的概率密度函数（PDF），使用norm.pdf函数计算PDF的值，传入均值和标准差作为参数
plt.plot(z_range, stats.norm.pdf(z_range, loc=mean_B, scale=std_B))

# 在第三个子图中，设置标题为Synthetic，横轴标签为z
plt.subplot(2, 3, 3)
plt.title('Synthetic')
plt.xlabel('z')
# 使用hist函数绘制z_C的直方图，设置bins为32，density为True表示归一化频数
plt.hist(z_C, bins=32, density=True)
# 使用plot函数绘制z_C的正态分布的概率密度函数（PDF），使用norm.pdf函数计算PDF的值，传入均值和标准差作为参数
plt.plot(z_range, stats.norm.pdf(z_range, loc=mean_C, scale=std_C))

# 在第四个子图中，设置标题为Natural，横轴标签为z
plt.subplot(2, 3, 4)
plt.title('Natural')
plt.xlabel('z')
# 使用hist函数绘制z_A的直方图，设置bins为32，density为True表示归一化频数，log为True表示使用对数坐标轴
plt.hist(z_A, bins=32, density=True, log=True)
# 使用plot函数绘制z_A的正态分布的概率密度函数（PDF），使用norm.pdf函数计算PDF的值，传入均值和标准差作为参数
plt.plot(z_range, stats.norm.pdf(z_range, loc=mean_A, scale=std_A))
# 使用yscale函数设置纵轴的刻度为对数刻度
plt.yscale('log')

# 在第五个子图中，设置标题为Non-natural，横轴标签为z
plt.subplot(2, 3, 5)
plt.title('Non-natural')
plt.xlabel('z')
# 使用hist函数绘制z_B的直方图，设置bins为32，density为True表示归一化频数，log为True表示使用对数坐标轴
plt.hist(z_B, bins=32, density=True, log=True)
# 使用plot函数绘制z_B的正态分布的概率密度函数（PDF），使用norm.pdf函数计算PDF的值，传入均值和标准差作为参数
plt.plot(z_range, stats.norm.pdf(z_range, loc=mean_B, scale=std_B))
# 使用yscale函数设置纵轴的刻度为对数刻度
plt.yscale('log')

plt.subplot(2, 3, 6)
plt.title('Synthetic')
plt.xlabel('z')
plt.hist(z_C, bins=32, density=True, log=True)
plt.plot(z_range, stats.norm.pdf(z_range, loc=mean_C, scale=std_C))
plt.yscale('log')

plt.tight_layout()
plt.savefig('3_scale.png')
plt.close()

###5. Down-sample your image by a 2 × 2 average (or simply sub-sample) the image. Plot the histogram and log histogram, and impose with the plots in step 1, to compare the diﬀerence. Repeat this down-sampling process 2-3 times.

# 使用interpolate函数对三组图像进行下采样，即降低图像的分辨率，减少图像的像素数
# interpolate函数要求输入的形状为[B, C, H, W]，其中B是批量大小，C是通道数，H是高度，W是宽度
# 因此我们需要使用torch.from_numpy函数将numpy数组转换为torch张量，并使用None和float函数调整形状和类型
# interpolate函数返回的结果也是torch张量，我们需要使用numpy和astype函数将其转换为numpy数组和整数类型

    # 原始分辨率
    # 原始分辨率
_ = vis_hist(nat_imgs, non_nat_img, syn_img, show_hist=False)
# 保存图片为hist_original.png
plt.savefig('4_hist_original.png')


# 下采样因子为2，即将图像的高度和宽度缩小一半
nat_imgs_down2 = [interpolate(torch.from_numpy(img)[None, None, :, :].float(), scale_factor=2)[0, 0].numpy().astype(int) for img in nat_imgs]
non_nat_img_down2 = interpolate(torch.from_numpy(non_nat_img)[None, None, :, :].float(), scale_factor=2)[0, 0].numpy().astype(int)
syn_img_down2 = interpolate(torch.from_numpy(syn_img)[None, None, :, :].float(), scale_factor=2)[0, 0].numpy().astype(int)
_ = vis_hist(nat_imgs_down2, non_nat_img_down2, syn_img_down2, show_hist=False)
# 保存图片为hist_down2.png
plt.savefig('5_hist_down2.png')


# 下采样因子为4，即将图像的高度和宽度缩小四分之一
nat_imgs_down4 = [interpolate(torch.from_numpy(img)[None, None, :, :].float(), scale_factor=2)[0, 0].numpy().astype(int) for img in nat_imgs_down2]
non_nat_img_down4 = interpolate(torch.from_numpy(non_nat_img_down2)[None, None, :, :].float(), scale_factor=2)[0, 0].numpy().astype(int)
syn_img_down4 = interpolate(torch.from_numpy(syn_img_down2)[None, None, :, :].float(), scale_factor=2)[0, 0].numpy().astype(int)
_ = vis_hist(nat_imgs_down4, non_nat_img_down4, syn_img_down4, show_hist=False)
# 保存图片为hist_down4.png
plt.savefig('6_hist_down4.png')


def verify_inverse_power_law(img):
    # 这个函数的目的是返回一个元组：（平移后的对数频谱，用于可视化曲线的数组）
    H, W = img.shape[:2]  # 获取图像的高度和宽度
    c_h = np.floor((H-1)/2).astype(int)   # 计算中心点的纵坐标
    c_w = np.floor((W-1)/2).astype(int)   # 计算中心点的横坐标

    FT_raw = np.abs(np.fft.fft2(img))   # 对图像进行傅里叶变换，并取复数的模
    A = np.zeros((2*c_h+1, 2*c_w+1))    # 创建一个全零的数组，大小和图像一样
    A[c_h::-1, c_w::-1] = FT_raw[:c_h+1, :c_w+1]   # 将傅里叶变换后的左上角部分复制到数组的中心
    A[c_h::-1, c_w:] = FT_raw[:c_h+1, :c_w+1]   # 将傅里叶变换后的右上角部分复制到数组的中心
    A[c_h:, c_w::-1] = FT_raw[:c_h+1, :c_w+1]   # 将傅里叶变换后的左下角部分复制到数组的中心
    A[c_h:, c_w:] = FT_raw[:c_h+1, :c_w+1]   # 将傅里叶变换后的右下角部分复制到数组的中心
    A = np.log(A)  # 对数组取对数，得到对数频谱

    r = np.indices((c_h+1, c_w+1)).transpose(1, 2, 0)  # 创建一个坐标网格，大小为图像中心点到左上角的距离
    r = np.linalg.norm(r, axis=-1, keepdims=False)  # 计算每个网格点到原点（即中心点）的距离
    r = np.log10(r + 1e-6)  # 对距离取对数，避免出现零值
    f_max = r.max()  # 获取最大距离值
    f_list = np.arange(0, f_max, 0.03)  # 创建一个等差数列，表示不同的频率区间
    a_list = []  # 创建一个空列表，用于存放每个频率区间对应的对数幅度值
    f_num = len(f_list)  # 获取频率区间的个数
    for i in range(f_num):  # 遍历每个频率区间
        idx_mask = np.logical_and(r>f_list[i], r<=(f_list[i+1] if i < f_num-1 else f_max))  # 创建一个布尔掩码，表示当前频率区间内的网格点
        if np.any(idx_mask):  # 如果当前频率区间内有任何网格点
            a_list.append(np.array([f_list[i], np.mean(A[c_h:, c_w:][idx_mask])]))  # 计算当前频率区间内对应的对数幅度值，并添加到列表中，同时记录当前频率值
    return A, np.stack(a_list, axis=0)  # 返回一个元组，包含平移后的对数频谱和用于可视化曲线的数组，数组的每一行是一个（频率，幅度）对

def solve_p2(img_dir):  # 这个函数的目的是对给定目录下的自然图像进行傅里叶变换，并绘制对数频谱和幅度曲线
    data_to_show = {}  # 创建一个空字典，用于存放每个图像的对数频谱和幅度曲线
    for fname in os.listdir(img_dir):  # 遍历目录下的所有文件名
        if 'natural' in fname:  # 如果文件名中包含'natural'，说明是自然图像
            img = skio.imread(os.path.join(img_dir, fname), as_gray=True)  # 读取图像，并转换为灰度图
            data_to_show.update({  # 将图像的文件名和对应的对数频谱和幅度曲线添加到字典中
                os.path.splitext(fname)[0]: verify_inverse_power_law(img)  # 调用之前定义的函数，返回一个元组
            })

    num_img = len(data_to_show)  # 获取自然图像的个数
    layout_side = np.ceil(np.sqrt(num_img)).astype(int)  # 计算绘制子图时每行需要的个数，向上取整
    maps = {}  # 创建一个空字典，用于存放每个图像的频谱（非对数）
    curves = {}  # 创建一个空字典，用于存放每个图像的幅度曲线
    i = 1  # 初始化子图的索引
    for k, v in data_to_show.items():  # 遍历每个图像的文件名和对应的对数频谱和幅度曲线
        A_map, A_list = v  # 解包元组，得到对数频谱和幅度曲线
        plt.subplot(layout_side+1, layout_side, i)  # 创建一个子图，位置为第i个
        plt.title(k)  # 设置子图的标题为文件名
        sns.heatmap(A_map, cmap='coolwarm', xticklabels=False, yticklabels=False)  # 绘制对数频谱的热力图，使用酷暖色调，不显示坐标轴标签
        maps.update({k: np.exp(A_map)}) # 将对数频谱转换为原始频谱，并添加到字典中
        curves.update({k: A_list})  # 将幅度曲线添加到字典中
        i += 1  # 更新子图的索引

    plt.tight_layout()  # 调整子图的布局，避免重叠或空隙
    plt.savefig('7_natural_scene.png')
    return maps, curves  # 返回两个字典，分别包含每个图像的频谱和幅度曲线

maps, curves = solve_p2(img_dir)  # 调用函数，传入目录名，得到两个字典，并赋值给变量maps和curves

colorlist = ['black', 'red', 'green', 'blue']  # 创建一个颜色列表，用于绘制不同的曲线
plt.figure()  # 创建一个新的图形
for i, k in enumerate(curves.keys()):  # 遍历每个图像的文件名和对应的索引
    plt.plot(curves[k][:, 0], curves[k][:, 1], color=colorlist[i], label=k)  # 绘制每个图像的幅度曲线，使用对应的颜色和标签
plt.title('Inverse power law')  # 设置图形的标题为'Inverse power law'
plt.xlabel('log f')  # 设置横坐标轴的标签为'log f'
plt.ylabel('log A')  # 设置纵坐标轴的标签为'log A'
plt.legend()  # 显示图例
plt.savefig('8_inverse_power_law.png')  # 保存图形为png文件


def power_integral(A_map, f0):  # 这个函数的目的是计算给定频谱在[f0, 2f0]区间内的功率积分
    A_map = A_map ** 2 # 将幅度转换为功率
    c_h, c_w = A_map.shape[:2]  # 获取频谱的高度和宽度
    c_h = (c_h - 1) // 2  # 计算中心点的纵坐标
    c_w = (c_w - 1) // 2  # 计算中心点的横坐标

    r = np.indices((c_h+1, c_w+1)).transpose(1, 2, 0)  # 创建一个坐标网格，大小为频谱中心点到左上角的距离
    r = np.linalg.norm(r, axis=-1, keepdims=False)  # 计算每个网格点到原点（即中心点）的距离
    idx_mask = np.logical_and(r>f0, r<=2*f0)  # 创建一个布尔掩码，表示[f0, 2f0]区间内的网格点
    if np.any(idx_mask):  # 如果该区间内有任何网格点
        return np.sum(A_map[c_h:, c_w:][idx_mask])  # 返回该区间内对应的功率值之和
    else:  # 否则
        raise ValueError('improper value for f_0')  # 抛出一个异常，说明f0的值不合适

f0_list = np.arange(1, 200, 5)  # 创建一个等差数列，表示不同的f0值
plt.figure()  # 创建一个新的图形
plt.clf()  # 清除当前的图形和轴
plt.title('Power invariance w.r.t. bandwidth')  # 设置图形的标题为'Power invariance w.r.t. bandwidth'
plt.xlabel('f_0')  # 设置横坐标轴的标签为'f_0'
plt.ylabel('Integral')  # 设置纵坐标轴的标签为'Integral'
plt.yticks(rotation=0)  # 设置纵坐标轴刻度的旋转角度为0
for i, k in enumerate(maps.keys()):  # 遍历每个图像的文件名和对应的索引
    inte = []  # 创建一个空列表，用于存放每个f0值对应的功率积分值
    for f0 in tqdm(f0_list):  # 遍历每个f0值，并显示进度条
        inte.append(power_integral(maps[k], f0))  # 调用之前定义的函数，计算每个f0值对应的功率积分值，并添加到列表中
    plt.plot(f0_list, inte, color=colorlist[i], label=k)  # 绘制每个图像的功率不变性曲线，使用对应的颜色和标签
plt.legend()  # 显示图例
plt.savefig('9_power_invariance.png')  # 保存图形为png文件

def GenLength(img_reso, size):  # 这个函数的目的是根据图像分辨率和线段数量，生成一些随机的线段长度
    alpha = np.random.random(size=size)  # 生成一些服从均匀分布的随机数，范围为[0, 1)
    if img_reso == 1024:  # 如果图像分辨率为1024
        return 8 / np.sqrt(2*alpha)  # 返回一个数组，每个元素是一个线段长度，服从指数分布
    elif img_reso == 512:  # 如果图像分辨率为512
        return 4 / np.sqrt(2*alpha)  # 返回一个数组，每个元素是一个线段长度，服从指数分布
    elif img_reso == 256:  # 如果图像分辨率为256
        return 2 / np.sqrt(2*alpha)  # 返回一个数组，每个元素是一个线段长度，服从指数分布
    else:  # 否则
        raise ValueError  # 抛出一个异常，说明图像分辨率不合法

def GenImg(N=5000, reso=1024):  # 这个函数的目的是生成一个随机的线段图像，可以指定线段数量和图像分辨率，默认为5000和1024
    # white background, black line segments
    coordinates = np.random.uniform(0, reso-1, size=(N, 2))  # 生成一些服从均匀分布的随机数，范围为[0, reso-1)，表示每个线段中心点的坐标
    orient = np.random.uniform(0, np.pi, size=N)  # 生成一些服从均匀分布的随机数，范围为[0, pi)，表示每个线段的方向角度
    length = GenLength(img_reso=reso, size=N)  # 调用之前定义的函数，生成一些随机的线段长度
    img = np.ones((reso, reso))  # 创建一个全白的图像，大小为reso x reso

    # draw lines
    for i in trange(N):  # 遍历每个线段，并显示进度条
        x1 = int(coordinates[i, 0] - length[i]/2*np.cos(orient[i]))  # 计算每个线段起点的横坐标
        y1 = int(coordinates[i, 1] - length[i]/2*np.sin(orient[i]))  # 计算每个线段起点的纵坐标
        x2 = int(coordinates[i, 0] + length[i]/2*np.cos(orient[i]))  # 计算每个线段终点的横坐标
        y2 = int(coordinates[i, 1] + length[i]/2*np.sin(orient[i]))  # 计算每个线段终点的纵坐标
        rr, cc, val = line_aa(r0=y1, c0=x1, r1=y2, c1=x2)  # 使用反走样算法，得到每个线段上所有像素点的行列索引和灰度值

        # truncation
        trun_mask_r = np.logical_and(rr>=0, rr<reso)  # 创建一个布尔掩码，表示行索引在有效范围内的像素点
        trun_mask_c = np.logical_and(cc>=0, cc<reso)  # 创建一个布尔掩码，表示列索引在有效范围内的像素点
        trun_mask = np.logical_and(trun_mask_r, trun_mask_c)  # 创建一个布尔掩码，表示在有效范围内的像素点

        rr = rr[trun_mask]  # 只保留在有效范围内的行索引
        cc = cc[trun_mask]  # 只保留在有效范围内的列索引
        val = 1- val[trun_mask] # 只保留在有效范围内的灰度值，并反转颜色（黑色为线段）

        img[rr, cc] = val   # 将图像上对应位置的像素值设置为灰度值

    return img   # 返回生成的图像

img_1024 = GenImg(N=5000, reso=1024)   # 调用函数，生成一个1024x1024的随机线段图像，并赋值给变量img_1024
img_512 = GenImg(N=1250, reso=512)   # 调用函数，生成一个512x512的随机线段图像，并赋值给变量img_512
img_256 = GenImg(N=313, reso=256)   # 调用函数，生成一个256x256的随机线段图像，并赋值给变量img_256

plt.subplot(1, 3, 1)   # 创建一个子图，位置为第一个
plt.title('1024x1024 sample')   # 设置子图的标题为'1024x1024 sample'
plt.imshow(img_1024, cmap='gray')   # 显示img_1024，并使用灰度色调

plt.subplot(1, 3, 2)   # 创建一个子图，位置为第二个
plt.title('512x512 sample')   # 设置子图的标题为'512x512 sample'
plt.imshow(img_512, cmap='gray')   # 显示img_512，并使用灰度色调

plt.subplot(1, 3, 3)   # 创建一个子图，位置为第三个
plt.title('256x256 sample')   # 设置子图的标题为'256x256 sample'
plt.imshow(img_256, cmap='gray')   # 显示img_256，并使用灰度色调

plt.tight_layout()   # 调整子图的布局，避免重叠或空隙
plt.savefig('10_1024x1024_sample.png')

for i in range(1, 3):
    r_start, c_start = np.random.randint(0, 1024-128+1, size=2)
    plt.subplot(3, 2, i)
    plt.title(f'Crop{i} on 1024')
    plt.imshow(img_1024[r_start: r_start+128, c_start: c_start+128], cmap='gray')

# crop on 512
for i in range(3, 5):
    r_start, c_start = np.random.randint(0, 512-128+1, size=2)
    plt.subplot(3, 2, i)
    plt.title(f'Crop{i-2} on 512')
    plt.imshow(img_512[r_start: r_start+128, c_start: c_start+128], cmap='gray')

# crop on 256
for i in range(5, 7):
    r_start, c_start = np.random.randint(0, 256-128+1, size=2)
    plt.subplot(3, 2, i)
    plt.title(f'Crop{i-4} on 256')
    plt.imshow(img_256[r_start: r_start+128, c_start: c_start+128], cmap='gray')

plt.tight_layout()


# crop on 1024
for i in range(1, 3):  # 遍历两次，分别为第一和第二个子图
    r_start, c_start = np.random.randint(0, 1024-128+1, size=2)  # 随机生成一个起始点的行列索引，范围为[0, 896]
    plt.subplot(3, 2, i)  # 创建一个子图，位置为第i个
    plt.title(f'Crop{i} on 1024')  # 设置子图的标题为'Crop{i} on 1024'
    plt.imshow(img_1024[r_start: r_start+128, c_start: c_start+128], cmap='gray')  # 显示从起始点开始，大小为128x128的区域，并使用灰度色调
    plt.savefig('11_1024_sample.png')  # 保存图像为png文件

# crop on 512
for i in range(3, 5):  # 遍历两次，分别为第三和第四个子图
    r_start, c_start = np.random.randint(0, 512-128+1, size=2)  # 随机生成一个起始点的行列索引，范围为[0, 384]
    plt.subplot(3, 2, i)  # 创建一个子图，位置为第i个
    plt.title(f'Crop{i-2} on 512')  # 设置子图的标题为'Crop{i-2} on 512'
    plt.imshow(img_512[r_start: r_start+128, c_start: c_start+128], cmap='gray')  # 显示从起始点开始，大小为128x128的区域，并使用灰度色调
    plt.savefig('12_512_sample.png')  # 保存图像为png文件
    
# crop on 256
for i in range(5, 7):  # 遍历两次，分别为第五和第六个子图
    r_start, c_start = np.random.randint(0, 256-128+1, size=2)  # 随机生成一个起始点的行列索引，范围为[0, 128]
    plt.subplot(3, 2, i)  # 创建一个子图，位置为第i个
    plt.title(f'Crop{i-4} on 256')  # 设置子图的标题为'Crop{i-4} on 256'
    plt.imshow(img_256[r_start: r_start+128, c_start: c_start+128], cmap='gray')  # 显示从起始点开始，大小为128x128的区域，并使用灰度色调
    plt.savefig('13_256_sample.png')  # 保存图像为png文件
plt.tight_layout()   # 调整子图的布局，避免重叠或空隙

