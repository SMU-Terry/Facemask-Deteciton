import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os,glob
import tqdm

# 人脸检测
def face_detect(img, img_height, img_width, face_detector):
    # 将图片转化为转为blob格式
    img_blob = cv.dnn.blobFromImage(img,1,(300,300),(100,177,123),swapRB=True)
    # 将blob格式图片输入到SSD模型中
    face_detector.setInput(img_blob)
    # 推理
    detections = face_detector.forward()
    # 人数
    person_count = detections.shape[2]
    for face_index in range(person_count):
        # 置信度
        confidence = detections[0,0,face_index,2]
        if confidence > 0.5:
            locations = detections[0,0,face_index,3:7] * np.array([img_width,img_height,img_width,img_height])
            l,t,r,b = locations.astype('int')
            return img[t:b,l:r],t,b,l,r
    return None,None,None,None,None


# 转为Blob
def img_blob(img):
    # 转为Blob
    img_blob = cv.dnn.blobFromImage(img,1,(100,100),(104,177,123),swapRB=True)
    # 压缩维度
    img_squeezed = np.squeeze(img_blob,axis=0)
    # img转置
    img_transpose = img_squeezed.T
    # 旋转
    img_rotate = cv.rotate(img_transpose,cv.ROTATE_90_CLOCKWISE)
    # 镜像
    img_flip = cv.flip(img_rotate,1)  
    # 去除负数，并归一化
    img_final = np.maximum(img_flip,0) / img_flip.max()  
    return img_final


if __name__ == '__main__':

    # 加载SSD模型
    face_detector = cv.dnn.readNetFromCaffe('./facemask_detection/weights/deploy.prototxt.txt', 
                    './facemask_detection/weights/res10_300x300_ssd_iter_140000.caffemodel')
    img_list = []
    label_list = []
    labels = os.listdir('./facemask_detection/images/')
    for label in labels:
        file_list = glob.glob('./facemask_detection/images/%s/*jpg'%(label))
        for img_file in tqdm.tqdm(file_list,desc='处理%s中 '%(label)):
            img = cv.imread(img_file)
            img_crop = face_detect(img, img.shape[0], img.shape[1], face_detector)
            if img_crop is not None:
                blob_img = img_blob(img_crop)
                img_list.append(blob_img)
                label_list.append(label)
    X = np.asarray(img_list)
    Y = np.asarray(label_list)
    np.savez('./facemask_detection/data/imageData.npz',X='img_list',Y='label_list')
