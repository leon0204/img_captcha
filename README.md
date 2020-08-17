# img_captcha
使用 tensor  做图片验证码 

#### coding tree
```
├── model
│   └── model.tar.gz
├── README.md
├── requirements.txt
├── src
│   ├── app.py 
│   ├── predict.py  # requests flask
│   ├── runFlask.sh # deploy flask
│   └── train_model.py  # train_model 
├── test-data
│   └── test-data.tar.gz
└── train-data
    └── train-data.tar.gz
```

#### 运行步骤



```
# 解压 train-data 和 test-data 到对应目录.（删除原目录）
tar xzvf test-data.tar.gz  
tar xzvf train-data.tar.gz

```

```
#安装 pip 依赖

pip install -r requirement.txt
```
```
# 训练模型
python3 train_model.py
# 自定义训练集大小和训练轮次
BATCH_SIZE = 700
EPOCHS = 50

或者直接使用我预训练的 
tar xzvf model.tar.gz

```

```
#部署flask服务，提供验证码服务
bash runFlask.sh
```


#### 测试
```
curl 127.0.0.1:9000/ping
# -> pong

#测试 test-daata
curl -X POST -F image=@56497.jpg 'http://localhost:9000/predict'

# 或者 python3 predict.py
```




#### 效果
![效果图](https://github.com/leon0204/img_captcha/blob/master/effect.png)



