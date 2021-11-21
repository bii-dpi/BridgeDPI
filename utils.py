import os, logging, pickle, random, torch, gc, deepchem

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


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class DataClass_normal:
    def __init__(self, dataPath, sep=' ',
                 pSeqMaxLen=1024, dSeqMaxLen=128,
                 kmers=-1,
                 validSize=0.2):
        """Load data."""

        print("Loading the raw data.")

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
        path = os.path.join(dataPath)
        with open(path, 'r') as f:
            while True:
                line = f.readline()
                if line == '':
                    break

                drug, protein, lab = line.strip().split(sep)

                if protein not in self.p2id:
                    pSeqData.append(protein)

                    self.p2id[protein] = pCnt
                    self.id2p.append(protein)
                    pCnt += 1

                if drug not in self.d2id:
                    mol = Chem.MolFromSmiles(drug)
                    if mol is None: continue

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

        # Tokenize protein data with padding at the end (1's).
        pSeqTokenized = []
        pSeqLen = []
        for pSeq in tqdm(pSeqData):
            pSeq = [self.am2id[am] for am in pSeq]
            pSeqLen.append(min(len(pSeq), pSeqMaxLen))
            pSeqTokenized.append(pSeq[:pSeqMaxLen] + [1] * max(pSeqMaxLen-len(pSeq),0))

        # Tokenize ligand data with padding at the end (1's).
        dSeqTokenized = []
        dSeqLen = []
        for dSeq in tqdm(dSeqData):
            atoms = [self.at2id[i] for i in dSeq]
            dSeqLen.append(min(len(dSeq), dSeqMaxLen))
            dSeqTokenized.append(atoms[:dSeqMaxLen] + [1]*max(dSeqMaxLen-len(atoms),0))

        # Split the examples by ID into training and validation.
        self.trainIdList, self.validIdList = train_test_split(range(len(eSeqData)), test_size=validSize)
        self.testIdList = []
        self.trainSampleNum, self.validSampleNum, self.testSampleNum = len(self.trainIdList), len(self.validIdList), len(self.testIdList)

        # Assign attributes from previously-computed variables.
        self.pSeqMaxLen, self.dSeqMaxLen = pSeqMaxLen, dSeqMaxLen
        self.pSeqData = pSeqData
        self.dSeqData, self.dMolData = dSeqData, dMolData
        self.pSeqLen, self.dSeqLen = np.array(pSeqLen, dtype=np.int32), np.array(dSeqLen, dtype=np.int32)
        self.pSeqTokenized = np.array(pSeqTokenized, dtype=np.int32)

        print("3. Creating k-mer features for proteins...")
        ctr = CountVectorizer(ngram_range=(1, 3), analyzer='char')
        pContFeat = ctr.fit_transform([''.join(i) for i in self.pSeqData]).toarray().astype('float32')
        k1, k2, k3 = [len(i)==1 for i in ctr.get_feature_names()], \
                     [len(i)==2 for i in ctr.get_feature_names()], \
                     [len(i)==3 for i in ctr.get_feature_names()]

        pContFeat[:,k1] = (pContFeat[:,k1] - pContFeat[:,k1].mean(axis=1).reshape(-1,1))/(pContFeat[:,k1].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat[:,k2] = (pContFeat[:,k2] - pContFeat[:,k2].mean(axis=1).reshape(-1,1))/(pContFeat[:,k2].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat[:,k3] = (pContFeat[:,k3] - pContFeat[:,k3].mean(axis=1).reshape(-1,1))/(pContFeat[:,k3].std(axis=1).reshape(-1,1)+1e-8)
        mean, std = pContFeat.mean(axis=0), pContFeat.std(axis=0)+1e-8
        pContFeat = (pContFeat - mean) / std
        self.pContFeatVectorizer = {'transformer':ctr,
                                    'mean':mean, 'std':std}
        self.pContFeat = pContFeat

        # Assign attributes from previously-computed variables.
        self.dSeqTokenized = np.array(dSeqTokenized, dtype=np.int32)
        self.dGraphFeat = np.array([i[:dSeqMaxLen]+[[0]*75]*(dSeqMaxLen-len(i)) for i in dFeaData], dtype=np.int8)
        self.dFinprFeat = np.array(dFinData, dtype=np.float32)

        self.eSeqData = np.array(eSeqData, dtype=np.int32)
        self.vector = {}

        print("Finished.")


    def describe(self):
        pass


    def change_seed(self, seed, validSize=0.2):
        """Get new train/validate split."""
        self.trainIdList, self.validIdList = train_test_split(range(len(self.eSeqData)),
                                                              test_size=validSize, random_state=seed)
        self.testIdList = []
        self.trainSampleNum, self.validSampleNum, self.testSampleNum = len(self.trainIdList), \
                                                                       len(self.validIdList), \
                                                                       len(self.testIdList)


    def vectorize(self, amSize=16, goSize=16, atSize=16, window=25, sg=1, kmers=-1,
                        workers=16, loadCache=True, pos=False, suf=''):
        print("Getting embeddings.")

        # Load pre-trained embeddings, if possible.
        path = f'cache/char2vec_am{amSize}_go{goSize}_at{atSize}.pkl'
        if os.path.exists(path) and loadCache:
            with open(path, 'rb') as f:
                self.vector['embedding'] = pickle.load(f)
            print(f'Loaded cache from {path}.')
            return

        # Train (new) embedding vectors.
        self.vector['embedding'] = {}

        print("1. Training amino acid char embeddings...")
        amDoc = [pSeq+'<EOS>' for pSeq in self.pSeqData]
        model = Word2Vec(amDoc, min_count=0, window=window, vector_size=amSize,
workers=workers, sg=sg, epochs=10)
        char2vec = np.zeros((self.amNum, amSize), dtype=np.float32)
        for i in range(self.amNum):
            if self.id2am[i] in model.wv:
                char2vec[i] = model.wv[self.id2am[i]]
            else:
                print(f'{self.id2am[i]} not in vocab, randomly initialize it...')
                char2vec[i] = np.random.random((1,amSize))
        self.vector['embedding']['amino'] = char2vec

        print("2. Training atom char embeddings...")
        atDoc = ["".join(dSeq)+'<EOS>' for dSeq in self.dSeqData]
        model = Word2Vec(atDoc, min_count=0, window=window, vector_size=atSize,
workers=workers, sg=sg, epochs=10)
        char2vec = np.zeros((self.atNum, atSize), dtype=np.float32)
        for i in range(self.atNum):
            if self.id2at[i] in model.wv:
                char2vec[i] = model.wv[self.id2at[i]]
            else:
                print(f"{self.id2at[i]} not in vocab, randomly initialize it...")
                char2vec[i] = np.random.random((1,atSize))
        self.vector['embedding']['atom'] = char2vec

        with open(path, 'wb') as f:
            pickle.dump(self.vector['embedding'], f, protocol=4)

        print("Finished.")


    def random_batch_data_stream(self, batchSize=32, type='train', sampleType='PWRL', device=torch.device('cpu'), log=False):
        if sampleType=='PWRL':
            pass
        elif sampleType=='CEL':
            if type=='train':
                idList = list(self.trainIdList)
            elif type=='valid':
                idList = list(self.validIdList)
            else:
                idList = list(self.testIdList)
            while True:
                random.shuffle(idList)
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


    def one_epoch_batch_data_stream(self, batchSize=32, type='valid', mode='predict', device=torch.device('cpu')):
        if mode=='train':
            pass

        elif mode=='predict':
            if type=='train':
                idList = list(self.trainIdList)
            elif type=='valid':
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

    def new_one_epoch_batch_data_stream(self, dataPath, batchSize=32, mode='predict', sep=' ', device=torch.device('cpu'), cache=None):
        # Open files and load data
        print('Loading the raw data...')
        p2id,id2p = {},[]
        d2id,id2d = {},[]
        pCnt,dCnt = 0,0
        pSeqData,pPSSMData,gSeqData,dMolData,dSeqData,dFeaData,dFinData = [],[],[],[],[],[],[]
        eSeqData = []

        if cache is not None and os.path.exists(cache):
            data = np.load(cache)
            pSeqTokenized = data['pSeqTokenized']
            pContFeat = data['pContFeat']
            pSeqLen = data['pSeqLen']
            dGraphFeat = data['dGraphFeat']
            dFinprFeat = data['dFinprFeat']
            dSeqTokenized = data['dSeqTokenized']
            dSeqLen = data['dSeqLen']
            eSeqData = data['eSeqData']
        else:
            path = os.path.join(dataPath)
            with open(path, 'r') as f:
                while True:
                    line = f.readline()
                    if line=='':
                        break
                    drug,protein,lab = line.strip().split(sep)

                    if protein not in p2id:
                        pSeqData.append( protein )
                        p2id[protein] = pCnt
                        id2p.append(protein)
                        pCnt += 1
                    if drug not in d2id:
                        mol = Chem.MolFromSmiles(drug)
                        if mol is None: continue
                        dSeqData.append( [a.GetSymbol() for a in mol.GetAtoms()] )
                        dMolData.append( mol )
                        dFeaData.append( [graph_features.atom_features(a) for a in mol.GetAtoms()] )

                        tmp = np.ones((1,))
                        DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol,2,nBits=1024), tmp)

                        dFinData.append( tmp )

                        d2id[drug] = dCnt
                        id2d.append(drug)
                        dCnt += 1

                    eSeqData.append( [p2id[protein], d2id[drug], lab] )

            print('Tokenizing the data...')
            # Tokenized protein data
            pSeqTokenized = []
            pSeqLen = []
            for pSeq in tqdm(pSeqData):
                pSeq = [self.am2id[am] for am in pSeq]
                pSeqLen.append( min(len(pSeq),self.pSeqMaxLen) )
                pSeqTokenized.append( pSeq[:self.pSeqMaxLen] + [1]*max(self.pSeqMaxLen-len(pSeq),0) )

            # Tokenized drug data
            dSeqTokenized = []
            dSeqLen = []
            for dSeq in tqdm(dSeqData):
                atoms = [self.at2id[i] for i in dSeq if i in self.at2id]
                dSeqLen.append( min(len(dSeq),self.dSeqMaxLen) )
                dSeqTokenized.append( atoms[:self.dSeqMaxLen] + [1]*max(self.dSeqMaxLen-len(atoms),0) )

            # Finish
            print('Other solving...')
            del dSeqData,dMolData
            gc.collect()

            pSeqLen,dSeqLen = np.array(pSeqLen, dtype=np.int32),np.array(dSeqLen, dtype=np.int32)
            pSeqTokenized = np.array(pSeqTokenized,dtype=np.int32)

            ctr = self.pContFeatVectorizer['transformer']
            pContFeat = ctr.transform([''.join(i) for i in pSeqData]).toarray().astype('float32')
            k1,k2,k3 = [len(i)==1 for i in ctr.get_feature_names()],[len(i)==2 for i in ctr.get_feature_names()],[len(i)==3 for i in ctr.get_feature_names()]

            pContFeat[:,k1] = (pContFeat[:,k1] - pContFeat[:,k1].mean(axis=1).reshape(-1,1))/(pContFeat[:,k1].std(axis=1).reshape(-1,1)+1e-8)
            pContFeat[:,k2] = (pContFeat[:,k2] - pContFeat[:,k2].mean(axis=1).reshape(-1,1))/(pContFeat[:,k2].std(axis=1).reshape(-1,1)+1e-8)
            pContFeat[:,k3] = (pContFeat[:,k3] - pContFeat[:,k3].mean(axis=1).reshape(-1,1))/(pContFeat[:,k3].std(axis=1).reshape(-1,1)+1e-8)
            pContFeat = (pContFeat-self.pContFeatVectorizer['mean']) / self.pContFeatVectorizer['std']

            dSeqTokenized = np.array(dSeqTokenized,dtype=np.int32)
            dGraphFeat = np.array([i[:self.dSeqMaxLen]+[[0]*75]*(self.dSeqMaxLen-len(i)) for i in dFeaData], dtype=np.int8)
            dFinprFeat = np.array(dFinData, dtype=np.float32)
            eSeqData = np.array(eSeqData, dtype=np.int32)

            if cache is not None:
                np.savez(cache, pSeqTokenized=pSeqTokenized, pContFeat=pContFeat, pSeqLen=pSeqLen,
                                dGraphFeat=dGraphFeat, dFinprFeat=dFinprFeat, dSeqTokenized=dSeqTokenized, dSeqLen=dSeqLen,
                                eSeqData=eSeqData)

        print('Predicting...')
        if mode=='train':
            pass
        elif mode=='predict':
            idList = list(range(len(eSeqData)))
            for i in tqdm(range((len(idList)+batchSize-1)//batchSize)):
                samples = idList[i*batchSize:(i+1)*batchSize]
                edges = eSeqData[samples]
                pTokenizedNames,dTokenizedNames = [i[0] for i in edges],[i[1] for i in edges]

                yield {
                        "res":True, \
                        "aminoSeq":torch.tensor(pSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
                        "aminoCtr":torch.tensor(pContFeat[pTokenizedNames], dtype=torch.float32).to(device), \
                        "pSeqLen":torch.tensor(pSeqLen[pTokenizedNames], dtype=torch.int32).to(device), \
                        "atomFea":torch.tensor(dGraphFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                        "atomFin":torch.tensor(dFinprFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                        "atomSeq":torch.tensor(dSeqTokenized[dTokenizedNames], dtype=torch.long).to(device), \
                        "dSeqLen":torch.tensor(dSeqLen[dTokenizedNames], dtype=torch.int32).to(device), \
                        }, torch.tensor([i[2] for i in edges], dtype=torch.float32).to(device)

    def single_data_stream(self, drug, protein, pSeqMaxLen=None, dSeqMaxLen=None, mode='predict', device=torch.device('cpu')):
        if pSeqMaxLen is None:
            pSeqMaxLen = self.pSeqMaxLen
        if dSeqMaxLen is None:
            dSeqMaxLen = self.dSeqMaxLen
        # Presolve the data
        print('Presolving the data...')
        pSeqData,dMolData,dSeqData,dFeaData,dFinData = [],[],[],[],[]

        pSeqData.append( protein )

        mol = Chem.MolFromSmiles(drug)
        dSeqData.append( [a.GetSymbol() for a in mol.GetAtoms()] )
        dMolData.append( mol )
        dFeaData.append( [graph_features.atom_features(a) for a in mol.GetAtoms()] )
        tmp = np.ones((1,))
        DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol,2,nBits=1024), tmp)
        dFinData.append( tmp )

        print('Tokenizing the data...')
        # Tokenized protein data
        pSeqTokenized = []
        pSeqLen = []
        for pSeq in tqdm(pSeqData):
            pSeq = [self.am2id[am] for am in pSeq]
            pSeqLen.append( min(len(pSeq),pSeqMaxLen) )
            pSeqTokenized.append( pSeq[:pSeqMaxLen] + [1]*max(pSeqMaxLen-len(pSeq),0) )

        # Tokenized drug data
        dSeqTokenized = []
        dSeqLen = []
        for dSeq in tqdm(dSeqData):
            atoms = [self.at2id[i] for i in dSeq if i in self.at2id]
            dSeqLen.append( min(len(dSeq),dSeqMaxLen) )
            dSeqTokenized.append( atoms[:dSeqMaxLen] + [1]*max(dSeqMaxLen-len(atoms),0) )

        # Finish
        print('Other solving...')
        del dSeqData,dMolData
        gc.collect()

        pSeqLen,dSeqLen = np.array(pSeqLen, dtype=np.int32),np.array(dSeqLen, dtype=np.int32)
        pSeqTokenized = np.array(pSeqTokenized,dtype=np.int32)

        ctr = self.pContFeatVectorizer['transformer']
        pContFeat = ctr.transform([''.join(i) for i in pSeqData]).toarray().astype('float32')
        k1,k2,k3 = [len(i)==1 for i in ctr.get_feature_names()],[len(i)==2 for i in ctr.get_feature_names()],[len(i)==3 for i in ctr.get_feature_names()]

        pContFeat[:,k1] = (pContFeat[:,k1] - pContFeat[:,k1].mean(axis=1).reshape(-1,1))/(pContFeat[:,k1].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat[:,k2] = (pContFeat[:,k2] - pContFeat[:,k2].mean(axis=1).reshape(-1,1))/(pContFeat[:,k2].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat[:,k3] = (pContFeat[:,k3] - pContFeat[:,k3].mean(axis=1).reshape(-1,1))/(pContFeat[:,k3].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat = (pContFeat-self.pContFeatVectorizer['mean']) / self.pContFeatVectorizer['std']

        dSeqTokenized = np.array(dSeqTokenized,dtype=np.int32)
        dGraphFeat = np.array([i[:dSeqMaxLen]+[[0]*75]*(dSeqMaxLen-len(i)) for i in dFeaData], dtype=np.int8)
        dFinprFeat = np.array(dFinData, dtype=np.float32)

        print('Predicting...')
        if mode=='train':
            pass
        elif mode=='predict':
            yield {
                    "res":True, \
                    "aminoSeq":torch.tensor(pSeqTokenized, dtype=torch.long).to(device), \
                    "aminoCtr":torch.tensor(pContFeat, dtype=torch.float32).to(device), \
                    "pSeqLen":torch.tensor(pSeqLen, dtype=torch.int32).to(device), \
                    "atomFea":torch.tensor(dGraphFeat, dtype=torch.float32).to(device), \
                    "atomFin":torch.tensor(dFinprFeat, dtype=torch.float32).to(device), \
                    "atomSeq":torch.tensor(dSeqTokenized, dtype=torch.long).to(device), \
                    "dSeqLen":torch.tensor(dSeqLen, dtype=torch.int32).to(device), \
                    }, torch.tensor([-1], dtype=torch.float32).to(device)

