# Convolutional-Neural-Networks-for-Sentence-Classification
Tensorflow implement of Convolutional Neural Networks for Sentence Classification(Kim, 2014)

2014년 EMNLP에 소개된 김윤 박사님의 CNN 모델 전체 재구현 버전입니다.

텐서플로우를 사용하였고, 모든 variation, 7개 dataset에 대하여 모두 실험 가능합니다.

The whole implementation of Yoon Kim's CNN Classifier(EMNLP 2014).

Tensorflow framework used, and you can experiment all variation for 7 dataset in the paper.

###Requirements

Python(>=3.5) \
Tensorflow-gpu(>=1.12.0)

### Data Preprocessing

Each raw dataset is in each dataset directory.

First, you need word2vec pre-trained vector file.
(Download in here : <a href = "https://translate.googleusercontent.com/translate_c?depth=1&hl=ko&prev=search&rurl=translate.google.com&sl=en&sp=nmt4&u=https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit%3Fusp%3Dsharing&xid=17259,15700022,15700186,15700190,15700253,15700256,15700259&usg=ALkJrhjmxptCcqmVqDCE9FCEOt3FKsoCfg"> google pre trained vector </a>)
You should adjust filepath in [datasetname]input.py.

Second, run [datasetname]input.py.
Then [datasetname]_original_data.bin will be made in the directory.

### Training and Testing

* Before run : adjust tensorflow running environment to yours. 

Run [datasetname]train.py.

The dataset is divided train, dev, test set and program tests accuracy for each epochs or steps.
Early stopping is adapted, so you should adjust proper condition for stopping in the code.


Model's variation and all the other experiment settings are in config.py.

※ some of the filepath may wrong...just fix it for run. :)





