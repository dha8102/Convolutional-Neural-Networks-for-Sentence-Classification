# Convolutional-Neural-Networks-for-Sentence-Classification
Tensorflow implement of <a href="https://arxiv.org/abs/1408.5882">Convolutional Neural Networks for Sentence Classification(Kim, 2014)</a>

2014년 EMNLP에 소개된 김윤 박사님의 CNN 모델 전체 재구현 버전입니다.

텐서플로우를 사용하였고, 모든 variation, 7개 dataset에 대하여 모두 실험 가능합니다.

The whole implementation of Yoon Kim's CNN Classifier(EMNLP 2014).

Tensorflow framework used, and you can experiment all variation for 7 dataset in the paper.

### Requirements

Python(>=3.5) \
Tensorflow-gpu(>=1.12.0)

### Data Preprocessing

Each raw dataset is in each dataset directory.

First, you need word2vec pre-trained vector file.
(Download in here : <a href = "https://translate.googleusercontent.com/translate_c?depth=1&hl=ko&prev=search&rurl=translate.google.com&sl=en&sp=nmt4&u=https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit%3Fusp%3Dsharing&xid=17259,15700022,15700186,15700190,15700253,15700256,15700259&usg=ALkJrhjmxptCcqmVqDCE9FCEOt3FKsoCfg"> google pre trained vector </a>)\
You should adjust filepath in [datasetname]input.py.

Second, run [datasetname]input.py.\
Then [datasetname]_original_data.bin will be made in the directory.

### Training and Testing

* Before run : adjust tensorflow running environment to yours. 

Run [datasetname]train.py.

The dataset is divided train, dev, test set and program tests accuracy for each epochs or steps.

Early stopping is adapted, so you should adjust proper condition for stopping in the code.


Model's variation and all the other experiment settings are in config.py.

※ some of the filepath may wrong...just fix it for run. :)

### Result

<img src = "https://lh3.googleusercontent.com/cA4WvVzf5GV5cXsBFcka4YLY1Be8sau3rBx-TVdK6cSxlt-mHeDUwSoxAGxo2MqrZdFuH1oxYMkw82X5jKpr9en6PfRglCjdBpoVSxwNedwyB24u2fH4zH3wsySuhn5girP3xwhDzikL-Sv8vrL6eayTD5N7UnzdV3LTf80weIFehs1kpz0MTD3w5vSfGnEEFJW0Yfy6RxogsEYKrr9WbwAsIhO9pAHYVkdLC-82EQsWZeIw0BpXv7lAK7JwptiRAIZGar0K1t5ey94I7zY7OPm973fdwtycjbFArxovqF0vwwoQsfyI09rvDGj6uULqdCojEF9Ao3cvBuCKBl8x3dysoDvWeudpf7sq3EssZgiaU6QyxuqY3myO3J-9RnUt__BXejsL6ONA7fU0QKt4RWGEOorGBt6zXltDdyNWwUwc5EeIWGDoKmKD7Nt0a3-b2rtCSKWLZVJr8u4uqplXFIs9sIZ7qkFSQ5nCTtP8ilh0lBorZVJTRsCEwxwJYdMwFpz0iJsCYNtRaHzVJQHiUoNQ7rb9ZhoTA38ZuctbCSNt-2N_hasz73r4YEm2SPbWWODZaAie6NEfERLkC8YrxvfHMAiJasSP0D0p13d6XdhQa3-WTzgbqsQuT2RkywkDACv-_2a0esEFykmEIexaHhaxA3sP204=w1280-h421-no">




