import cv2 as cv
import numpy as np
import tensorflow as tf
import image_preproccess
import time

class MaskDetection:
    """
    口罩检测：正常、未佩戴、不规范（漏鼻子）
    可运行在树莓派
    """
    def __init__(self):
        """
        构造函数
        """
        # 加载之前训练好的口罩检测模型
        self.model = tf.keras.models.load_model('./facemask_detection/data/face_mask_model.h5')
        print(self.model.summary())

        # 加载SSD模型
        self.face_detector = cv.dnn.readNetFromCaffe('./facemask_detection/weights/deploy.prototxt.txt', 
                        './facemask_detection/weights/res10_300x300_ssd_iter_140000.caffemodel')
        # 中文label图像
        self.resize_width = 100
        self.resize_height = 40
        self.overlay_height = self.resize_height
        self.overlay_width = self.resize_width
        self.zh_label_img_list = self.getPngList()
        # 颜色，BGR顺序，绿色、红色、黄色
        self.colors = [(0,255,0),(0,0,255),(0,255,255)]


    def getPngList(self):
        """
        获取PNG图像列表

        @return numpy array list
        """
        overlay_list = []
        for i in range(3):
            fileName = './facemask_detection/label_img/%s.png' % (i)
            overlay = cv.imread(fileName, cv.COLOR_BGR2RGB)
            # overlay = cv.resize(overlay, (0,0), fx=0.3, fy=0.3)
            overlay = cv.resize(overlay, (self.resize_width,self.resize_height))
            overlay_list.append(overlay)
        return overlay_list


    def facemask_detection(self):
        """
        口罩识别
        """
        cap = cv.VideoCapture(0)
        frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # 记录帧率时间
        start_time = time.time()
        while True:
            # 读取
            ret,frame = cap.read()
            # 翻转
            frame = cv.flip(frame,1)
            # # 识别前缩放
            # frame_resize = cv.resize(frame,(300,300))
            # 提取出人脸区域
            face_region,face_top,face_bottom,face_left,face_right = image_preproccess.face_detect(frame, frame_h, frame_w, self.face_detector)

            if face_region is not None:
                # 将人脸区域转化为blob格式
                blob_img = image_preproccess.img_blob(face_region)
                # 增加一个维度(batch_size dimension = 1)
                img_input = blob_img.reshape(1,100,100,3)
                result = self.model.predict(img_input)
                # softmax处理
                result = tf.nn.softmax(result[0]).numpy()
                # 最大值索引
                max_index = result.argmax()
                # 最大值
                max_value = result[max_index]
                # 中文标签
                overlay = self.zh_label_img_list[max_index]
                # 覆盖范围
                overlay_left,overlay_top = face_left,(face_top-self.overlay_height-20)
                overlay_right,overlay_bottom = (face_left+self.overlay_width),(overlay_top+self.overlay_height)
                # 判断边界
                if overlay_top>0 and overlay_right<frame_w:
                    overlay_copy = cv.addWeighted(frame[overlay_top:overlay_bottom,overlay_left:overlay_right],
                                                1, overlay, 20 ,0)
                    frame[overlay_top:overlay_bottom,overlay_left:overlay_right] = overlay_copy
                    cv.putText(frame, str(round(max_value*100,2))+'%', (overlay_right+20,overlay_top+40),
                                cv.FONT_ITALIC, 0.8, self.colors[max_index], 2)

                # 人脸框
                cv.rectangle(frame,(face_left,face_top), (face_right,face_bottom), self.colors[max_index], 5)

            # 显示帧率
            now = time.time()
            fps = 1 / (now-start_time)
            start_time = now
            cv.putText(frame, "FPS:  " + str(round(fps,2)), (10, 15), cv.FONT_ITALIC, 0.6, (0, 255, 0), 2)

            cv.imshow('demo', frame)
            if cv.waitKey(10) == 27:
                break
        
        cap.release()
        cv.destroyAllWindows()
            

if __name__ == '__main__':
    mask_detection = MaskDetection()
    mask_detection.facemask_detection()