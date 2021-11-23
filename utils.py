import os, logging, pickle, torch, gc, deepchem

import numpy as np
import pandas as pd

from tqdm import tqdm
from rdkit.Chem import AllChem
from collections import Counter
from gensim.models import Word2Vec
from rdkit import Chem, DataStructs
from deepchem.feat import graph_features
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from deepchem.models.graph_models import GraphConvModel
from sklearn.feature_extraction.text import CountVectorizer


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class DataClass_normal:
    def __init__(self, direction, sep=" ",
                 pSeqMaxLen=2644, dSeqMaxLen=188):
        """Load data."""

        print("Loading the raw data.")

        self.direction = direction
        dataPath = f"../get_data/Bridge/data/{direction}"

        # ID mappers.
        ## Map protein sequence to order ID, and vice versa.
        self.p2id, self.id2p = {}, []
        ## The same for ligands.
        self.d2id, self.id2d = {}, []
        pCnt, dCnt = 0, 0

        # Uniques data.
        ## Flat list of all protein sequences.
        pSeqData = []
        ## List-of-lists of ligand SMILES strings (each string a list).
        dSeqData = []
        ## Flat list of all ligand Mol objects.
        dMolData = []
        ## List-of-lists of ligand per-atom graph features.
        ## https://github.com/deepchem/deepchem/blob/master/deepchem/feat/graph_features.py
        dFeaData = []
        ## List of all Morgan fingerprints of the ligands.
        dFinData = []

        # Individual example data: list of [pID, lID, label].
        eSeqData = []

        print("1. Loading features...")
        path = os.path.join(f"{dataPath}_training")
        with open(path, "r") as f:
            curr_lines = f.readlines()

        for line in tqdm(curr_lines):
            drug, protein, lab = line.strip("\n").strip().split(sep)

            if protein not in self.p2id:
                pSeqData.append(protein)

                self.p2id[protein] = pCnt
                self.id2p.append(protein)
                pCnt += 1

            if drug not in self.d2id:
                mol = Chem.MolFromSmiles(drug)

                dSeqData.append([a.GetSymbol() for a in mol.GetAtoms()])
                dMolData.append(mol)
                dFeaData.append([graph_features.atom_features(a) for a in mol.GetAtoms()])

                tmp = np.ones((1,))
                DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol, 2, nBits=1024),
                                                tmp)
                dFinData.append(tmp)

                self.d2id[drug] = dCnt
                self.id2d.append(drug)
                dCnt += 1

            eSeqData.append([self.p2id[protein], self.d2id[drug], lab])

        path = os.path.join(f"{dataPath}_testing")
        with open(path, "r") as f:
            curr_lines = f.readlines()

        for line in tqdm(curr_lines):
            drug, protein, lab = line.strip("\n").strip().split(sep)

            if protein not in self.p2id:
                pSeqData.append(protein)

                self.p2id[protein] = pCnt
                self.id2p.append(protein)
                pCnt += 1

            if drug not in self.d2id:
                mol = Chem.MolFromSmiles(drug)

                dSeqData.append([a.GetSymbol() for a in mol.GetAtoms()])
                dMolData.append(mol)
                dFeaData.append([graph_features.atom_features(a) for a in mol.GetAtoms()])

                tmp = np.ones((1,))
                DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol, 2, nBits=1024),
                                                tmp)
                dFinData.append(tmp)

                self.d2id[drug] = dCnt
                self.id2d.append(drug)
                dCnt += 1

            eSeqData.append([self.p2id[protein], self.d2id[drug], lab])

        # Create constituent maps.
        print("2. Creating amino acid and atom maps to tokenize data...")
        self.am2id, self.id2am = {"<UNK>":0, "<EOS>":1}, ["<UNK>", "<EOS>"]
        amCnt = 2
        for pSeq in tqdm(pSeqData):
            for am in pSeq:
                if am not in self.am2id:
                    self.am2id[am] = amCnt
                    self.id2am.append(am)
                    amCnt += 1
        self.amNum = amCnt

        self.at2id, self.id2at = {"<UNK>":0, "<EOS>":1}, ["<UNK>", "<EOS>"]
        atCnt = 2
        for dSeq in tqdm(dSeqData):
            for at in dSeq:
                if at not in self.at2id:
                    self.at2id[at] = atCnt
                    self.id2at.append(at)
                    atCnt += 1
        self.atNum = atCnt

        # Tokenize protein data with padding at the end (1"s).
        pSeqTokenized = []
        pSeqLen = []
        for pSeq in tqdm(pSeqData):
            pSeq = [self.am2id[am] for am in pSeq]
            pSeqLen.append(min(len(pSeq), pSeqMaxLen))
            pSeqTokenized.append(pSeq[:pSeqMaxLen] + [1] * max(pSeqMaxLen-len(pSeq),0))

        # Tokenize ligand data with padding at the end (1"s).
        dSeqTokenized = []
        dSeqLen = []
        for dSeq in tqdm(dSeqData):
            atoms = [self.at2id[i] for i in dSeq]
            dSeqLen.append(min(len(dSeq), dSeqMaxLen))
            dSeqTokenized.append(atoms[:dSeqMaxLen] + [1]*max(dSeqMaxLen-len(atoms),0))

        # Split the examples by ID into training and validation.
        def get_len(dataPath, suffix):
            with open(f"{dataPath}_{suffix}", "r") as f:
                return len(f.readlines())


        self.testIdList = []
        self.testSampleNum = len(self.testIdList)
        self.trainSampleNum = get_len(dataPath, "training")
        self.validSampleNum = get_len(dataPath, "testing")

        self.trainIdList = list(range(self.trainSampleNum))
        self.validIdList = list(range(self.trainSampleNum,
                                      self.trainSampleNum +
                                      self.validSampleNum))

        # Assign attributes from previously-computed variables.
        self.pSeqMaxLen, self.dSeqMaxLen = pSeqMaxLen, dSeqMaxLen
        self.pSeqData = pSeqData
        self.dSeqData, self.dMolData = dSeqData, dMolData
        self.pSeqLen, self.dSeqLen = np.array(pSeqLen, dtype=np.int32), np.array(dSeqLen, dtype=np.int32)
        self.pSeqTokenized = np.array(pSeqTokenized, dtype=np.int32)

        print("3. Creating k-mer features for proteins...")
        ctr = CountVectorizer(ngram_range=(1, 3), analyzer="char")
        pContFeat = ctr.fit_transform(["".join(i) for i in self.pSeqData]).toarray().astype("float32")
        k1, k2, k3 = [len(i)==1 for i in ctr.get_feature_names()], \
                     [len(i)==2 for i in ctr.get_feature_names()], \
                     [len(i)==3 for i in ctr.get_feature_names()]

        # TODO: make this respect train/valid split. XXX: No. Why? No label info
        # is used here.
        pContFeat[:,k1] = (pContFeat[:,k1] - pContFeat[:,k1].mean(axis=1).reshape(-1,1))/(pContFeat[:,k1].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat[:,k2] = (pContFeat[:,k2] - pContFeat[:,k2].mean(axis=1).reshape(-1,1))/(pContFeat[:,k2].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat[:,k3] = (pContFeat[:,k3] - pContFeat[:,k3].mean(axis=1).reshape(-1,1))/(pContFeat[:,k3].std(axis=1).reshape(-1,1)+1e-8)
        mean, std = pContFeat.mean(axis=0), pContFeat.std(axis=0)+1e-8
        pContFeat = (pContFeat - mean) / std
        self.pContFeatVectorizer = {"transformer":ctr,
                                    "mean":mean, "std":std}
        self.pContFeat = pContFeat

        # Assign attributes from previously-computed variables.
        self.dSeqTokenized = np.array(dSeqTokenized, dtype=np.int32)
        self.dGraphFeat = np.array([i[:dSeqMaxLen]+[[0]*75]*(dSeqMaxLen-len(i)) for i in dFeaData], dtype=np.int8)
        self.dFinprFeat = np.array(dFinData, dtype=np.float32)

        self.eSeqData = np.array(eSeqData, dtype=np.int32)
        self.vector = {}

        print("Finished.")


    def random_batch_data_stream(self, seed, batchSize=32, type="train", sampleType="PWRL", device=torch.device("cpu"), log=False):
        np.random.seed(seed)

        if sampleType=="PWRL":
            pass
        elif sampleType=="CEL":
            if type=="train":
                idList = list(self.trainIdList)
            elif type=="valid":
                idList = list(self.validIdList)
            else:
                idList = list(self.testIdList)
            while True:
                np.random.shuffle(idList)
                for i in range((len(idList)+batchSize-1)//batchSize):
                    samples = idList[i*batchSize:(i+1)*batchSize]
                    edges = self.eSeqData[samples]
                    pTokenizedNames,dTokenizedNames = [i[0] for i in edges],[i[1] for i in edges]
                    yield {
                            "res":True, \
                            "aminoSeq":torch.tensor(self.pSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
                            "aminoCtr":torch.tensor(self.pContFeat[pTokenizedNames], dtype=torch.float32).to(device), \
                            "pSeqLen":torch.tensor(self.pSeqLen[pTokenizedNames], dtype=torch.int32).to(device), \
                            "atomFea":torch.tensor(self.dGraphFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                            "atomFin":torch.tensor(self.dFinprFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                            "atomSeq":torch.tensor(self.dSeqTokenized[dTokenizedNames], dtype=torch.long).to(device), \
                            "dSeqLen":torch.tensor(self.dSeqLen[dTokenizedNames], dtype=torch.int32).to(device), \
                          }, torch.tensor([i[2] for i in edges], dtype=torch.float32).to(device)


    def one_epoch_batch_data_stream(self, batchSize=32, type="valid", mode="predict", device=torch.device("cpu")):
        if mode=="train":
            pass

        elif mode=="predict":
            if type=="train":
                idList = list(self.trainIdList)
            elif type=="valid":
                idList = list(self.validIdList)
            else:
                idList = list(self.testIdList)
            for i in range((len(idList)+batchSize-1)//batchSize):
                samples = idList[i*batchSize:(i+1)*batchSize]
                edges = self.eSeqData[samples]
                pTokenizedNames,dTokenizedNames = [i[0] for i in edges],[i[1] for i in edges]

                yield {
                        "res":True, \
                        "aminoSeq":torch.tensor(self.pSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
                        "aminoCtr":torch.tensor(self.pContFeat[pTokenizedNames], dtype=torch.float32).to(device), \
                        "pSeqLen":torch.tensor(self.pSeqLen[pTokenizedNames], dtype=torch.int32).to(device), \
                        "atomFea":torch.tensor(self.dGraphFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                        "atomFin":torch.tensor(self.dFinprFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                        "atomSeq":torch.tensor(self.dSeqTokenized[dTokenizedNames], dtype=torch.long).to(device), \
                        "dSeqLen":torch.tensor(self.dSeqLen[dTokenizedNames], dtype=torch.int32).to(device), \
                        }, torch.tensor([i[2] for i in edges], dtype=torch.float32).to(device)

