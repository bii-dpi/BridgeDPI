from utils import *
from DL_ClassifierModel import *
from sklearn.metrics import average_precision_score as aupr


def test_model(direction):
    #If you need test the embeddings, you can run " ***dataClass.vectorize(amSize=16, atSize=16)*** ".
    dataClass = DataClass_normal(dataPath=f"../get_data/Bridge/data/{direction}_testing",
                                 pSeqMaxLen=2644, dSeqMaxLen=188,
                                 sep=' ')

    model = DTI_Bridge(outSize=128,
                       cHiddenSizeList=[1024],
                       fHiddenSizeList=[1024,256],
                       fSize=1024, cSize=dataClass.pContFeat.shape[1],
                       gcnHiddenSizeList=[128,128], fcHiddenSizeList=[128], nodeNum=64,
                       hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))
    model.load(path=f"models/{direction}.pkl", map_location="cuda",
               dataClass=dataClass)

    model.to_eval_mode()
    y_est, y = model.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(batchSize=128,
                                                                                        type='test',
                                                                                        device=torch.device('cuda')))
    print(aupr(y, y_est))


test_model("dztbz")

