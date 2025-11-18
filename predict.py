import cv2
import math
import torch
import numpy as np
from Utils import *
import cv_algorithms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt


def predict(model, image_location,op_image_name, path, device='cpu'):
    sig = nn.Sigmoid()
    image = cv2.imread(image_location)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #print('Shape of original image: ', image.shape)
    x,y = image.shape[:2]
    resize_w, resize_h = 384,384
    rs_image = cv2.resize(image,(resize_w, resize_h))
    r1, r2 = x/resize_w, y/resize_h

    ip_image = rs_image/255
    ip_image = ip_image.transpose((2,0,1))
    ip_image = torch.from_numpy(ip_image).float()
  
    with torch.no_grad():
        model.eval()
        model.to(device)
        x_tensor = ip_image.to(device).unsqueeze(0)
        pr_mask = model(x_tensor)
        pr_mask = pr_mask.squeeze().cpu()
        pr_mask = sig(pr_mask).round()
        pr = pr_mask.numpy()

        imgThresh = cv2.threshold(pr, 0, 1, cv2.THRESH_BINARY)[1]  
        guo_hall = cv_algorithms.guo_hall(imgThresh.astype(np.uint8))

    op_image, final_lines = draw_lines(np.copy(rs_image),guo_hall, r1, r2, op_image_name, path)
    if final_lines == 0:
      return
    guo_hall =  cv2.resize(guo_hall,(y,x))
    op_image = cv2.resize(op_image,(y,x))
    return final_lines
   
def draw_lines(image,mask, r1, r2, op_image_name, path):
    mask = mask*255
    mask = cv2.GaussianBlur(mask,(5,5),1)
    mask = cv2.Canny(mask.astype(np.uint8),100,255)
    lines = cv2.HoughLinesP(mask,1,np.pi / 180,threshold=50,
                        minLineLength=50,maxLineGap=250)
    if not np.any(lines):
      return 0,0
    lines = np.squeeze(lines, axis=1)
    m_lines = aggregate_lines(lines)
    final_lines = []
    for line in m_lines:

        x1, y1, x2, y2 = line.astype(int)
        m = (y2-y1)/(x2-x1+0.0001)
        slope = math.atan(m)
        lengt = np.sqrt((x2-x1)**2)+((y2-y1)**2)
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = str(int(x1*r1))+','+str(int(y1*r2))#+str(int(x2,y2))
        
    for l in m_lines:
      x,y, x11,y11 = l.astype(int)
      p21, p22 = (x,y), (x11,y11)
      p31,p32 = extend_line(image.shape, p21, p22)
      
      if p31[1] <= p32[1]:
        final_lines.append((p31[0]*r2, p31[1]*r1, p32[0]*r2, p32[1]*r1))

      elif p31[1]>=p32[1]:
        final_lines.append((p32[0]*r2, p32[1]*r1, p31[0]*r2, p31[1]*r1))

    sorter = lambda x: (x[0], x[1]) #(x[0], x[1])
    final_lines = sorted(final_lines, key=sorter)

    with open(path+op_image_name[:-4]+".txt", "w") as file:

      file.write(str(final_lines))

    return image,  final_lines 
