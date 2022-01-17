import numpy as np
import pandas as pd
import torch,time,os,pickle,random
from torch import nn as nn
from nnLayer import *
from metrics import *
from collections import Counter,Iterable
from sklearn.model_selection import StratifiedKFold,KFold
from torch.backends import cudnn
from tqdm import tqdm
from Others import *


class BaseClassifier:
    def __init__(self):
        pass


    def calculate_y_logit(self, X, XLen):
        pass


    def train(self, dataClass, seed, savePath,
              trainSize=512, batchSize=512,
              epochs=100, earlyStop=100, saveRounds=1,
              lr=0.001, weightDecay=0.001,
              isHigherBetter=True, metrics="AUPR", report=["ACC", "AUC",
                                                           "AUPR", "F1", "LOSS",
                                                           "recall_1",
"recall_5", "recall_10", "recall_25", "recall_50"]):
        # XXX: Why are these two different things? No real reason for this.
        assert batchSize%trainSize==0
        self.stepCounter = 0
        self.stepUpdate = batchSize//trainSize

        # Create the logging files.
        with open(f"results/{dataClass.direction}_training_perf_{seed}.csv", "w") as f:
            f.write("Epoch," + ",".join(report) + "\n")

        with open(f"results/{dataClass.direction}_testing_perf_{seed}.csv", "w") as f:
            f.write("Epoch," + ",".join(report) + "\n")

        # Create the performance-calculator.
        metrictor = Metrictor()

        # Create the Adam optimizer with L2 regularization and LR scheduler.
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.moduleList.parameters()),
                                     lr=lr, weight_decay=weightDecay)
        schedulerRLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max" if isHigherBetter else "min",
                                                                  factor=0.5, patience=4, verbose=True)

        # Get the training data as an iterable.
        trainStream = dataClass.random_batch_data_stream(seed=seed,
                                                         batchSize=trainSize,
                                                         type="train",
                                                         device=self.device)
        itersPerEpoch = (dataClass.trainSampleNum+trainSize-1)//trainSize

        # Keep track of current epoch performance, best performance, and the
        # number of epochs since this best performance.
        mtc,bestMtc,stopSteps = 0.0,0.0,0

        # Get the validation data.
        if dataClass.validSampleNum>0:
            validStream = dataClass.random_batch_data_stream(seed=seed,
                                                             batchSize=trainSize,
                                                             type="valid",
                                                             device=self.device)

        for e in range(epochs):
            for i in range(itersPerEpoch):
                self.to_train_mode()
                X,Y = next(trainStream)
                if X["res"]:
                    loss = self._train_step(X,Y, optimizer)

            if dataClass.validSampleNum>0 and (e+1)%saveRounds==0:
                self.to_eval_mode()
                print(f"========== Epoch:{e+1:5d} ==========")
                Y_pre,Y = self.calculate_y_prob_by_iterator(
                            dataClass.one_epoch_batch_data_stream(trainSize,
                                                                  type="train", mode="predict",
                                                                  device=self.device))
                metrictor.set_data(Y_pre, Y)
                print(f"[Total Train]",end="")
                curr_results = metrictor(report)
                with open(f"results/{dataClass.direction}_training_perf_{seed}.csv", "a") as f:
                    f.write(f"{e+1}," + ",".join([str(curr_results[met]) for met in report]) + "\n")

                print(f"[Total Valid]",end="")
                Y_pre,Y = self.calculate_y_prob_by_iterator(
                            dataClass.one_epoch_batch_data_stream(trainSize,
                                                                  type="valid", mode="predict",
                                                                  device=self.device))
                metrictor.set_data(Y_pre, Y)
                res = metrictor(report)
                with open(f"results/{dataClass.direction}_testing_perf_{seed}.csv", "a") as f:
                    f.write(f"{e+1}," + ",".join([str(res[met]) for met in report]) + "\n")

                mtc = res[metrics]
                schedulerRLR.step(mtc)
                print("=================================")

                if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                    print(f"Got a better Model with val {metrics}: {mtc:.3f}")
                    bestMtc = mtc
                    self.save("%s.pkl"%savePath, e+1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps>=earlyStop:
                        print(f"The val {metrics} has not improved for more than {earlyStop} steps in epoch {e+1}, stop training.")
                        break


    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()


    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {"epochs":epochs, "bestMtc":bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            #stateDict["trainIdList"],stateDict["validIdList"],stateDict["testIdList"] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            if "am2id" in stateDict:
                stateDict["am2id"],stateDict["id2am"] = dataClass.am2id,dataClass.id2am
            if "go2id" in stateDict:
                stateDict["go2id"],stateDict["id2go"] = dataClass.go2id,dataClass.id2go
            if "at2id" in stateDict:
                stateDict["at2id"],stateDict["id2at"] = dataClass.at2id,dataClass.id2at
        torch.save(stateDict, path)
        print("Model saved in '%s'."%path)


    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            # if "trainIdList" in parameters:
            #     dataClass.trainIdList = parameters["trainIdList"]
            # if "validIdList" in parameters:
            #     dataClass.validIdList = parameters["validIdList"]
            # if "testIdList" in parameters:
            #     dataClass.testIdList = parameters["testIdList"]
            if "am2id" in parameters:
                dataClass.am2id,dataClass.id2am = parameters["am2id"],parameters["id2am"]
            if "go2id" in parameters:
                dataClass.go2id,dataClass.id2go = parameters["go2id"],parameters["id2go"]
            if "at2id" in parameters:
                dataClass.at2id,dataClass.id2at = parameters["at2id"],parameters["id2at"]
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters["epochs"], parameters["bestMtc"]))


    def calculate_y_prob(self, X, mode):
        Y_pre = self.calculate_y_logit(X, mode)["y_logit"]
        return torch.sigmoid(Y_pre)


    def calculate_loss(self, X, Y):
        out = self.calculate_y_logit(X, "predict")
        Y_logit = out["y_logit"]

        addLoss = 0.0
        if "loss" in out: addLoss += out["loss"]
        return self.criterion(Y_logit, Y) + addLoss


    def calculate_indicator_by_iterator(self, dataStream, classNum, report):
        metrictor = Metrictor(classNum)
        Y_prob_pre,Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y)
        return metrictor(report)


    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X, mode="predict").cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.concatenate(YArr).astype("int32"),np.concatenate(Y_preArr).astype("float32")
        return Y_preArr, YArr


    def to_train_mode(self):
        for module in self.moduleList:
            module.train()


    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()


    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        if self.stepCounter<self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y)/self.stepUpdate
        loss.backward()

        if p:
            nn.utils.clip_grad_norm_(self.moduleList.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
        return loss*self.stepUpdate


class DTI_Bridge(BaseClassifier):
    def __init__(self, seed, cSize, device,
                 outSize=128, cHiddenSizeList=[1024], fHiddenSizeList=[1024,256],
                 fSize=1024,
                 gcnHiddenSizeList=[128,128], fcHiddenSizeList=[128], nodeNum=64,
                 resnet=True,
                 hdnDropout=0.5, fcDropout=0.5,
                 useFeatures = {"kmers":True,"pSeq":True,"FP":True,"dSeq":True},
                 maskDTI=False):
        torch.manual_seed(seed)

        self.nodeEmbedding = TextEmbedding(torch.tensor(np.random.normal(size=(max(nodeNum,0),outSize)), dtype=torch.float32), dropout=hdnDropout, name="nodeEmbedding").to(device)

        self.amEmbedding = TextEmbedding(torch.eye(24), dropout=hdnDropout, freeze=True, name="amEmbedding").to(device)
        self.pCNN = TextCNN(24, 64, [25], ln=True, name="pCNN").to(device)
        self.pFcLinear = MLP(64, outSize, dropout=hdnDropout, bnEveryLayer=True, dpEveryLayer=True, outBn=True, outAct=True, outDp=True, name="pFcLinear").to(device)

        self.dCNN = TextCNN(75, 64, [7], ln=True, name="dCNN").to(device)
        self.dFcLinear = MLP(64, outSize, dropout=hdnDropout, bnEveryLayer=True, dpEveryLayer=True, outBn=True, outAct=True, outDp=True, name="dFcLinear").to(device)

        self.fFcLinear = MLP(fSize, outSize, fHiddenSizeList, outAct=True, name="fFcLinear", dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True).to(device)
        self.cFcLinear = MLP(cSize, outSize, cHiddenSizeList, outAct=True, name="cFcLinear", dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True).to(device)

        self.nodeGCN = GCN(outSize, outSize, gcnHiddenSizeList, name="nodeGCN", dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True, resnet=resnet).to(device)

        self.fcLinear = MLP(outSize, 1, fcHiddenSizeList, dropout=fcDropout, bnEveryLayer=True, dpEveryLayer=True).to(device)

        self.criterion = nn.BCEWithLogitsLoss()

        self.embModuleList = nn.ModuleList([])
        self.finetunedEmbList = nn.ModuleList([])
        self.moduleList = nn.ModuleList([self.nodeEmbedding,self.cFcLinear,self.fFcLinear,self.nodeGCN,self.fcLinear,
                                         self.amEmbedding, self.pCNN, self.pFcLinear, self.dCNN, self.dFcLinear])
        self.device = device
        self.resnet = resnet
        self.nodeNum = nodeNum
        self.hdnDropout = hdnDropout
        self.useFeatures = useFeatures
        self.maskDTI = maskDTI

    def calculate_y_logit(self, X, mode="train"):
        Xam = (self.cFcLinear(X["aminoCtr"]).unsqueeze(1) if self.useFeatures["kmers"] else 0) + \
              (self.pFcLinear(self.pCNN(self.amEmbedding(X["aminoSeq"]))).unsqueeze(1) if self.useFeatures["pSeq"] else 0) # => batchSize × 1 × outSize
        Xat = (self.fFcLinear(X["atomFin"]).unsqueeze(1) if self.useFeatures["FP"] else 0) + \
              (self.dFcLinear(self.dCNN(X["atomFea"])).unsqueeze(1) if self.useFeatures["dSeq"] else 0) # => batchSize × 1 × outSize

        if self.nodeNum>0:
            node = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(self.nodeEmbedding.embedding.weight)).repeat(len(Xat), 1, 1)
            node = torch.cat([Xam, Xat, node], dim=1) # => batchSize × nodeNum × outSize
            nodeDist = torch.sqrt(torch.sum(node**2,dim=2,keepdim=True)+1e-8)# => batchSize × nodeNum × 1

            cosNode = torch.matmul(node,node.transpose(1,2)) / (nodeDist*nodeDist.transpose(1,2)+1e-8) # => batchSize × nodeNum × nodeNum
            #cosNode = cosNode*0.5 + 0.5
            cosNode = F.relu(cosNode) # => batchSize × nodeNum × nodeNum
            cosNode[:,range(node.shape[1]),range(node.shape[1])] = 1 # => batchSize × nodeNum × nodeNum
            if self.maskDTI: cosNode[:,0,1] = cosNode[:,1,0] = 0
            D = torch.eye(node.shape[1], dtype=torch.float32, device=self.device).repeat(len(Xam),1,1) # => batchSize × nodeNum × nodeNum
            D[:,range(node.shape[1]),range(node.shape[1])] = 1/(torch.sum(cosNode,dim=2)**0.5)
            pL = torch.matmul(torch.matmul(D,cosNode),D) # => batchSize × batchnodeNum × nodeNumSize
            node_gcned = self.nodeGCN(node, pL) # => batchSize × nodeNum × outSize

            node_embed = node_gcned[:,0,:]*node_gcned[:,1,:] # => batchSize × outSize
        else:
            node_embed = (Xam*Xat).squeeze(dim=1) # => batchSize × outSize
        #if self.resnet:
        #    node_gcned += torch.cat([Xam[:,0,:],Xat[:,0,:]],dim=1)
        return {"y_logit":self.fcLinear(node_embed).squeeze(dim=1)}#, "loss":1*l2}

