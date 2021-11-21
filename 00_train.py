from utils import *
from DL_ClassifierModel import *


def train_model(direction):
    #If you need train the embeddings, you can run " ***dataClass.vectorize(amSize=16, atSize=16)*** ".
    dataClass = DataClass_normal(dataPath=f"../get_data/Bridge/data/{direction}_training",
                                 pSeqMaxLen=2644, dSeqMaxLen=188,
                                 sep=' ')

    model = DTI_Bridge(outSize=128,
                       cHiddenSizeList=[1024],
                       fHiddenSizeList=[1024,256],
                       fSize=1024, cSize=dataClass.pContFeat.shape[1],
                       gcnHiddenSizeList=[128,128], fcHiddenSizeList=[128], nodeNum=64,
                       hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))

    model.train(dataClass, trainSize=512, batchSize=512, epoch=100,
                lr1=0.001, stopRounds=-1, earlyStop=30,
                savePath=f'models/direction', metrics="AUC", report=["ACC", "AUC", "LOSS"],
                preheat=0)

train_model("dztbz")

"""
Also, if you want to train the E2E, E2E/go models, you just need to instance another model class (DTI_E2E, DTI_E2E_nogo, see DL_ClassifierModel.py for more details).

## 3. How to do prediction
```python
model = DTI_Bridge(...)
model.load(path="xxx.pkl", map_location="cpu", dataClass=dataClass)
model.to_eval_mode()
Ypre,Y = model.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(batchSize=128, type='test', device=torch.device('cpu')))
```
>**path** is your model saved path, which is a ".pkl" file.
"""

