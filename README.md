# Facemask-Deteciton
This project can detecte whether you ware facemask in standard way. It can detecte three modes(1. 'standardly wearing', 2. 'unstandly wearing', 3. 'no wearing') 
该项目可以检测人物是否规范佩戴口罩，一共由三种模式(正常、未佩戴、不规范(漏鼻子)) 

# image_preproccess.py
firstly, this script will crop your face from the picture, and then it will transfer your face region into blob formate for better training in the next step. After proccessing all images, these images will be stored in numpy zip formate.

# train_model.py
It will train the by using LeNet CNN.

<img width="1866" alt="cnn" src="https://user-images.githubusercontent.com/64240681/167637184-2cdd8245-ceaa-488e-96f7-e7bef691a6e2.png">

# test_model.py
After you've done all above operations. Run this script to test your model. Or if you don't want to train the model(haven't finished above two steps), I have already saved my training model in the data file, you can use it dirctly run test_model.py. BTW, you need to use tensorflow==2.2.0 so that you can load my training model, cause i used tensorflow==2.2.0 to save the model, it needs to keep the version consistent.

![Q5%70@Z Z_1~HM3YDQLW2PP](https://user-images.githubusercontent.com/64240681/167639216-058703f9-cefe-4ec9-a20f-bd78c68634ea.png)

![@AJ$CRP`VF 2 {486 LK9E](https://user-images.githubusercontent.com/64240681/167639352-ab3ba990-66cc-4490-975d-51f54bf18d03.png)

![TU`GZ33ZK08~A@0QYLA3SLI](https://user-images.githubusercontent.com/64240681/167639368-3e268a7f-21d8-4f67-a110-0d52aba44b60.png)


# PS
Here is the download link for images data:
(链接：https://pan.baidu.com/s/1qQRRmUxE5h6mROSIjYTO8g 提取码：qb36)
