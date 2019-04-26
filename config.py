####### model configuration #######

def embedding_dim():
    return 300


def filter_size():
    return [3, 4, 5]


def num_filters():
    return 100


def dropout_prob():
    return 0.5


def l2_lambda():
    return 0.001


def batch_size():
    return 50


def evaluate_every():
    return 500


def dev_ratio():
    return 0.1



def cv_num():
    return 3


def model_variation():
    # 0:CNN-rand, 1:CNN-static, 2:CNN-non-static, 3:CNN-multichannel
    return 3

def dataset_name():
    # MR, SST1, SST2, Subj, TREC, CR, MPQA
    return "CR"
