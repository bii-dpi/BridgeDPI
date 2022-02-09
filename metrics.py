import warnings

import numpy as np

from collections import Counter
from sklearn import metrics as skmetrics


warnings.filterwarnings("ignore")


def lgb_MaF(preds, dtrain):
    Y = np.array(dtrain.get_label(), dtype=np.int32)
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'macro_f1', float(F1(preds.shape[0], Y_pre, Y, 'macro')), True


def lgb_precision(preds, dtrain):
    Y = dtrain.get_label()
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'precision', float(Counter(Y==Y_pre)[True]/len(Y)), True


id2lab = [[-1,-1]]*20
for a in range(1,11):
    for s in [1,2]:
        id2lab[a-1+(s-1)*10] = [a,s]


class Metrictor:
    def __init__(self):
        self._reporter_ = {"ACC":self.ACC, "AUC":self.AUC,
                           "Precision":self.Precision, "Recall":self.Recall,
                           "AUPR":self.AUPR, "F1":self.F1, "LOSS":self.LOSS,
                           "recall_1": self.recall_1, "recall_5": self.recall_5,
                           "recall_10": self.recall_10,
                           "recall_25": self.recall_25, "recall_50": self.recall_50,
                           "LogAUC": self.logAUC, "EF_1": self.ef_1,
                           "EF_5": self.ef_5, "EF_10": self.ef_10,
                           "EF_25": self.ef_25, "EF_50": self.ef_50}


    def __call__(self, report, end='\n'):
        res = {}
        for mtc in report:
            v = self._reporter_[mtc]()
            print(f" {mtc}={v:6.3f}", end=';')
            res[mtc] = v
        print(end=end)
        return res



    def set_data(self, Y_prob_pre, Y, threshold=0.5):
        self.Y = Y.astype('int')
        if len(Y_prob_pre.shape)>1:
            self.Y_prob_pre = Y_prob_pre[:,1]
            self.Y_pre = Y_prob_pre.argmax(axis=-1)
        else:
            self.Y_prob_pre = Y_prob_pre
            self.Y_pre = (Y_prob_pre>threshold).astype('int')



    @staticmethod
    def table_show(resList, report, rowName='CV'):
        lineLen = len(report)*8 + 6
        print("="*(lineLen//2-6) + "FINAL RESULT" + " ="*(lineLen//2-6))
        print(f"{'-':^6}" + "".join([f"{i:>8}" for i in report]))
        for i,res in enumerate(resList):
            print(f"{rowName+'_'+str(i+1):^6}" + "".join([f"{res[j]:>8.3f}" for j in report]))
        print(f"{'MEAN':^6}" + "".join([f"{np.mean([res[i] for res in resList]):>8.3f}" for i in report]))
        print("======" + "========"*len(report))


    def each_class_indictor_show(self, id2lab):
        print('Waiting for finishing...')


    def ACC(self):
        return ACC(self.Y_pre, self.Y)


    def AUC(self):
        return AUC(self.Y_prob_pre,self.Y)


    def logAUC(self):
        return LogAUC(self.Y_prob_pre,self.Y)


    def Precision(self):
        return Precision(self.Y_pre, self.Y)


    def Recall(self):
        return Recall(self.Y_pre, self.Y)


    def AUPR(self):
        return AUPR(self.Y_prob_pre, self.Y)


    def F1(self):
        return F1(self.Y_pre, self.Y)


    def LOSS(self):
        return LOSS(self.Y_prob_pre,self.Y)


    def recall_1(self):
        return RECALL_X(self.Y, self.Y_prob_pre, 0.01)


    def recall_5(self):
        return RECALL_X(self.Y, self.Y_prob_pre, 0.05)


    def recall_10(self):
        return RECALL_X(self.Y, self.Y_prob_pre, 0.1)


    def recall_25(self):
        return RECALL_X(self.Y, self.Y_prob_pre, 0.25)


    def recall_50(self):
        return RECALL_X(self.Y, self.Y_prob_pre, 0.50)


    def ef_1(self):
        return EF(self.Y, self.Y_prob_pre, 0.01)


    def ef_5(self):
        return EF(self.Y, self.Y_prob_pre, 0.05)


    def ef_10(self):
        return EF(self.Y, self.Y_prob_pre, 0.1)


    def ef_25(self):
        return EF(self.Y, self.Y_prob_pre, 0.25)


    def ef_50(self):
        return EF(self.Y, self.Y_prob_pre, 0.50)


def ACC(Y_pre, Y):
    return (Y_pre==Y).sum() / len(Y)


def AUC(Y_prob_pre, Y):
    return skmetrics.roc_auc_score(Y, Y_prob_pre)


def Precision(Y_pre, Y):
    return skmetrics.precision_score(Y, Y_pre)


def Recall(Y_pre, Y):
    return skmetrics.recall_score(Y, Y_pre)


def EF(y, Y_pre, prec_val, sorted_=False):
    if not sorted_:
        sorted_indices = np.argsort(Y_pre)[::-1]
        y = y[sorted_indices]
        Y_pre = Y_pre[sorted_indices]

    num_pos = np.sum(y)
    len_to_take = int(len(y) * prec_val)

    return np.sum(y[:len_to_take]) / num_pos


def LogAUC(Y_pre, y):
    sorted_indices = np.argsort(Y_pre)[::-1]
    y = y[sorted_indices]
    Y_pre = Y_pre[sorted_indices]
    num_pos = np.sum(y)

    prec_vals = np.arange(1, 101) / 1000
    recalls = []
    for prec_val in prec_vals:
        recalls.append(EF(y, Y_pre, prec_val, True))

    return np.trapz(y=recalls, x=np.log10(prec_vals))


def RECALL_X(Y, Y_pre, x):
    precisions, recalls, _ = \
        skmetrics.precision_recall_curve(Y, Y_pre)
    precisions = np.abs(np.array(precisions) - x)

    return recalls[np.argmin(precisions)]


def AUPR(Y_pre, Y):
    return skmetrics.average_precision_score(Y, Y_pre)


def F1(Y_pre, Y):
    return skmetrics.f1_score(Y, Y_pre)


def LOSS(Y_prob_pre, Y):
    Y_prob_pre,Y = Y_prob_pre.reshape(-1),Y.reshape(-1)
    Y_prob_pre[Y_prob_pre>0.99] -= 1e-3
    Y_prob_pre[Y_prob_pre<0.01] += 1e-3
    return -np.mean(Y*np.log(Y_prob_pre) + (1-Y)*np.log(1-Y_prob_pre))

