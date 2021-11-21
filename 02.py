from utils import *
from DL_ClassifierModel import *


def train_model(direction):
    #If you need train the embeddings, you can run " ***dataClass.vectorize(amSize=16, atSize=16)*** ".
    dataClass = DataClass_normal(dataPath=f"../get_data/Bridge/data/{direction}_training",
                                 pSeqMaxLen=2644, dSeqMaxLen=188,
                                 sep=' ')
    dataClass.vectorize()

    model = DTI_Bridge(outSize=128,
                       cHiddenSizeList=[1024],
                       fHiddenSizeList=[1024,256],
                       fSize=1024, cSize=dataClass.pContFeat.shape[1],
                       gcnHiddenSizeList=[128,128], fcHiddenSizeList=[128], nodeNum=64,
                       hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))

    model.train(dataClass, trainSize=512, batchSize=512, epoch=1,
                lr1=0.001, stopRounds=-1, earlyStop=30,
                savePath=f'models/direction', metrics="AUC", report=["ACC", "AUC", "LOSS"],
                preheat=0)

    return model

model = train_model("dztbz")

from sklearn.metrics import average_precision_score as aupr

#If you need test the embeddings, you can run " ***dataClass.vectorize(amSize=16, atSize=16)*** ".
dataClass = DataClass_normal(dataPath=f"../get_data/Bridge/data/dztbz_testing",
                             pSeqMaxLen=2644, dSeqMaxLen=188,
                             sep=' ')

model.to_eval_mode()
y_est, y = model.calculate_y_logit(dataClass)

print(aupr(y, y_est))



