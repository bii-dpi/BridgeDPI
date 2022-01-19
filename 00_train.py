import warnings

from utils import *
from DL_ClassifierModel import *
from progressbar import progressbar


warnings.filterwarnings("ignore")


DIRECTION = "btd"
CUDA_NUM = 0
EPOCHS = 100


def train_model(direction, seed, cuda_num, dataClass):
    model = DTI_Bridge(seed=seed,
                       cSize=dataClass.pContFeat.shape[1],
                       device=torch.device(f"cuda:{cuda_num}"))

    model.train(dataClass, seed, epochs=EPOCHS,
                savePath=f"models/{direction}_{seed}")


dataClass = DataClass_normal(direction=DIRECTION)

for seed in [123456789]:
    train_model(DIRECTION, seed, CUDA_NUM, dataClass)

