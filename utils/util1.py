import torch
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as Draw
import random

def narrow_image(im):
    """
    缩放图片成416*416
    :param im:  原图
    :return: 原图W、H、缩放比例、缩放后的416*416图片
    """
    img = Image.new(mode="RGB", size=(416, 416), color=(128, 128, 128))     #生成416*416的灰色图片
    W,H = im.size             #获取原始图片宽高
    max_side = max(W,H)           #最大边长
    min_side = min(W,H)           #最短边长
    scale = 416/max_side          #得到缩放比例
    Z = max_side-min_side         #得到最大边和最短边差值
    im.thumbnail((416,416))   #按照最大边长缩放图片为416*416
    if W > H :               #如果框大于高，填充高
        xy = (0,int((Z*scale)/2))
    else:                    #如果高大于宽，填充宽
        xy = (int((Z*scale)/2),0)
    img.paste(im, xy)       #把图片粘贴到灰色图区域，起始点为xy

    return W,H,scale,img

def enlarge_box(W,H,scale,box):
    """
    网络输出的框为416*416图片上的框，需反算到原图上
    :param W: 原图的W
    :param H: 原图的H
    :param scale: 缩放比例
    :param box: 网络输出的框
    :return: 反算到原图上的框
    """
    if box.shape[0] == 0:
        return torch.Tensor([])
    cent_x = box[:,1:2]     #中心点x
    cent_y = box[:,2:3]     #中心点y
    if W >= H :             #如果宽大于高，在y坐标上减去填充部分坐标
        cent_x = cent_x/scale
        cent_y = cent_y/scale-(W-H)/2
    else:                   #如果高大于宽，在x坐标上减去填充部分坐标
        cent_x = cent_x/scale-(H-W)/2
        cent_y = cent_y/scale
    w_h = box[:,3:5]       #网络输出宽高
    w_h = w_h/scale        #原图上的宽高
    box[:, 1:2] = cent_x
    box[:, 2:3] = cent_y
    box[:, 3:5] = w_h
    return box

def narrow_box(W,H,scale,box):
    """
    原图需reshape成416*416图片，
    实际标注坐标也会发生变化，
    这里会按比例缩放原图标签到缩放后的416*416图片中
    :param W: 原图W
    :param H: 原图H
    :param scale: 缩放比例
    :param box: 原图的标签框
    :return: 缩放后的标签框
    """
    cent_x = box[:, 1:2]      #中心点x
    cent_y = box[:, 2:3]      #中心点y
    if W > H:           #如果宽大于高，在y坐标上加上填充部分坐标
        cent_x = cent_x * scale
        cent_y = (cent_y + (W - H) / 2) * scale
    else:               #如果宽大于高，在x坐标上加上填充部分坐标
        cent_x = (cent_x + (H - W) / 2) * scale
        cent_y = cent_y * scale
    w_h = box[:, 3:5]        #原图宽高
    w_h = w_h * scale        #缩放后的宽高
    box[:, 1:2] = cent_x
    box[:, 2:3] = cent_y
    box[:, 3:5] = w_h
    return box

def draw(box,image):
    """
    画图函数：把框在原图上画出
    :param box: 实际框
    :param image: 原图
    :return: 画了框的图
    """

    fp = open(r'coco.names', "r")
    text = fp.read().split("\n")[:-1]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(text))]
    draw = Draw.ImageDraw(image)
    W,H = image.size
    for i in range(len(box)):       #逐个框画出
        cx = box[i][1]
        cy = box[i][2]
        w = box[i][3]
        h = box[i][4]

        num_class = int(box[i][5])       #分类标签
        cls = float('%.2f' % box[i][6])

        x1 = int(cx - w / 2)
        if x1 < 0:
            x1 = 0
        y1 = int(cy - h / 2)
        if y1 < 0:
            y1 = 0
        x2 = int(cx + w / 2)
        if x2 > W:
            x2 = W
        y2 = int(cy + h / 2)
        if y2 > H :
            y2 = H
        xy = (x1, y1, x2, y2)      #实际框

        xy_ = (x1, y1 - 15, x2, y1)     #信息框

        draw.rectangle(xy, fill=None, outline=tuple(colors[num_class]), width=3)       #实际框
        draw.rectangle(xy_, fill=tuple(colors[num_class]), outline=tuple(colors[num_class]), width=3)      #信息框
        draw.text(xy=(x1 + 2, y1 - 12), text=text[num_class]+ " " + str(cls), fill="black", font=None)      #目标信息
    return image
    # image.show()