#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import time
import os
from datetime import datetime,timedelta,date
from datetime import datetime as dt
from sklearn.externals import joblib
from sklearn import svm
from category_encoders import TargetEncoder
from sqlalchemy import create_engine
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.max_seq_items = None

def Clinet_Karma_Inference():
    start = datetime.now()
    print('开始时间:', start)
    print('输入数据集位置与文件名')
    data_location = input()
    
    #engine = create_engine('mssql+pymssql://username:password@tpecogcm2:1433/BIWork',echo=True)
    #df = pd.read_sql('SELECT * FROM table_name1',engine)
    
    
    df = pd.read_csv(data_location,names=['DDwFD', 'contractsn', 'DATE', 'DuringMBA', 'RH', 'AH', 're', 'fbCNT',
       'nCR', 'nMR', 'nTR', 'nconcompla', 'nconcompli', 'nmatcompla',
       'nmatcompli', 'nteccompla', 'accLIKE', 'proLIKE', 'accDLIKE',
       'proDLIKE', 'FavorCNT', 'BlockCNT', 'AddFavorCNT', 'AddBlockCNT',
       'star', 'isIntCNT', 'LVdiffCNT', 'MGMLead', 'MGMRes', 'MGMDemo',
       'MGMDeal', 'conUUCCNT', 'conSOUCNT', 'conSERCNT', 'conTWBCNT',
       'conOTHCNT', 'conLigCNT', 'conMedCNT', 'conTanCNT', 'conDarCNT',
       'conEx01CNT', 'conEx03CNT', 'conEx12CNT', 'conEx24CNT', 'conage30CNT',
       'conage40CNT', 'conage50CNT', 'conage60CNT', 'conage61CNT', 'conMCNT',
       'conFCNT', 'helpCNT', 'helpCNTmax', 'help3mCNT', 'help3mCNTmax',
       'helpITCNT', 'helpITCNTmax', 'helpIT3mCNT', 'helpIT3mCNTmax',
       'helpIiCNT', 'helpIiCNTmax', 'helpIi3mCNT', 'helpIi3mCNTmax',
       'helpMaCNT', 'helpMaCNTmax', 'helpMa3mCNT', 'helpMa3mCNTmax',
       'helpTeCNT', 'helpTeCNTmax', 'helpTe3mCNT', 'helpTe3mCNTmax',
       'helpconCNT', 'helpconCNTmax', 'helpcon3mCNT', 'helpcon3mCNTmax',
       'ALL_RH', 'ALL_AH', 'ALL_re', 'ALL_fbCNT', 'ALL_nCR', 'ALL_nMR',
       'ALL_nTR', 'ALL_nconcompla', 'ALL_nconcompli', 'ALL_nmatcompla',
       'ALL_nmatcompli', 'ALL_nteccompla', 'ALL_accLIKE', 'ALL_proLIKE',
       'ALL_accDLIKE', 'ALL_proDLIKE', 'ALL_FavorCNT', 'ALL_BlockCNT',
       'ALL_AddFavorCNT', 'ALL_AddBlockCNT', 'ALL_star', 'ALL_isIntCNT',
       'ALL_LVdiffCNT', 'ALL_MGMLead', 'ALL_MGMRes', 'ALL_MGMDemo',
       'ALL_MGMDeal', 'ALL_conUUCCNT', 'ALL_conSOUCNT', 'ALL_conSERCNT',
       'ALL_conTWBCNT', 'ALL_conOTHCNT', 'ALL_conLigCNT', 'ALL_conMedCNT',
       'ALL_conTanCNT', 'ALL_conDarCNT', 'ALL_conEx01CNT', 'ALL_conEx03CNT',
       'ALL_conEx12CNT', 'ALL_conEx24CNT', 'ALL_conage30CNT',
       'ALL_conage40CNT', 'ALL_conage50CNT', 'ALL_conage60CNT',
       'ALL_conage61CNT', 'ALL_conMCNT', 'ALL_conFCNT', 'ALL_helpCNT',
       'ALL_helpCNTmax', 'ALL_help3mCNT', 'ALL_help3mCNTmax', 'ALL_helpITCNT',
       'ALL_helpITCNTmax', 'ALL_helpIT3mCNT', 'ALL_helpIT3mCNTmax',
       'ALL_helpIiCNT', 'ALL_helpIiCNTmax', 'ALL_helpIi3mCNT',
       'ALL_helpIi3mCNTmax', 'ALL_helpMaCNT', 'ALL_helpMaCNTmax',
       'ALL_helpMa3mCNT', 'ALL_helpMa3mCNTmax', 'ALL_helpTeCNT',
       'ALL_helpTeCNTmax', 'ALL_helpTe3mCNT', 'ALL_helpTe3mCNTmax',
       'ALL_helpconCNT', 'ALL_helpconCNTmax', 'ALL_helpcon3mCNT',
       'ALL_helpcon3mCNTmax', 'L1D_RH', 'L1D_AH', 'L1D_re', 'L1D_fbCNT',
       'L1D_nCR', 'L1D_nMR', 'L1D_nTR', 'L1D_nconcompla', 'L1D_nconcompli',
       'L1D_nmatcompla', 'L1D_nmatcompli', 'L1D_nteccompla', 'L1D_accLIKE',
       'L1D_proLIKE', 'L1D_accDLIKE', 'L1D_proDLIKE', 'L1D_FavorCNT',
       'L1D_BlockCNT', 'L1D_AddFavorCNT', 'L1D_AddBlockCNT', 'L1D_star',
       'L1D_isIntCNT', 'L1D_LVdiffCNT', 'L1D_MGMLead', 'L1D_MGMRes',
       'L1D_MGMDemo', 'L1D_MGMDeal', 'L1D_conUUCCNT', 'L1D_conSOUCNT',
       'L1D_conSERCNT', 'L1D_conTWBCNT', 'L1D_conOTHCNT', 'L1D_conLigCNT',
       'L1D_conMedCNT', 'L1D_conTanCNT', 'L1D_conDarCNT', 'L1D_conEx01CNT',
       'L1D_conEx03CNT', 'L1D_conEx12CNT', 'L1D_conEx24CNT', 'L1D_conage30CNT',
       'L1D_conage40CNT', 'L1D_conage50CNT', 'L1D_conage60CNT',
       'L1D_conage61CNT', 'L1D_conMCNT', 'L1D_conFCNT', 'L1D_helpCNT',
       'L1D_helpCNTmax', 'L1D_help3mCNT', 'L1D_help3mCNTmax', 'L1D_helpITCNT',
       'L1D_helpITCNTmax', 'L1D_helpIT3mCNT', 'L1D_helpIT3mCNTmax',
       'L1D_helpIiCNT', 'L1D_helpIiCNTmax', 'L1D_helpIi3mCNT',
       'L1D_helpIi3mCNTmax', 'L1D_helpMaCNT', 'L1D_helpMaCNTmax',
       'L1D_helpMa3mCNT', 'L1D_helpMa3mCNTmax', 'L1D_helpTeCNT',
       'L1D_helpTeCNTmax', 'L1D_helpTe3mCNT', 'L1D_helpTe3mCNTmax',
       'L1D_helpconCNT', 'L1D_helpconCNTmax', 'L1D_helpcon3mCNT',
       'L1D_helpcon3mCNTmax', 'L3D_RH', 'L3D_AH', 'L3D_re', 'L3D_fbCNT',
       'L3D_nCR', 'L3D_nMR', 'L3D_nTR', 'L3D_nconcompla', 'L3D_nconcompli',
       'L3D_nmatcompla', 'L3D_nmatcompli', 'L3D_nteccompla', 'L3D_accLIKE',
       'L3D_proLIKE', 'L3D_accDLIKE', 'L3D_proDLIKE', 'L3D_FavorCNT',
       'L3D_BlockCNT', 'L3D_AddFavorCNT', 'L3D_AddBlockCNT', 'L3D_star',
       'L3D_isIntCNT', 'L3D_LVdiffCNT', 'L3D_MGMLead', 'L3D_MGMRes',
       'L3D_MGMDemo', 'L3D_MGMDeal', 'L3D_conUUCCNT', 'L3D_conSOUCNT',
       'L3D_conSERCNT', 'L3D_conTWBCNT', 'L3D_conOTHCNT', 'L3D_conLigCNT',
       'L3D_conMedCNT', 'L3D_conTanCNT', 'L3D_conDarCNT', 'L3D_conEx01CNT',
       'L3D_conEx03CNT', 'L3D_conEx12CNT', 'L3D_conEx24CNT', 'L3D_conage30CNT',
       'L3D_conage40CNT', 'L3D_conage50CNT', 'L3D_conage60CNT',
       'L3D_conage61CNT', 'L3D_conMCNT', 'L3D_conFCNT', 'L3D_helpCNT',
       'L3D_helpCNTmax', 'L3D_help3mCNT', 'L3D_help3mCNTmax', 'L3D_helpITCNT',
       'L3D_helpITCNTmax', 'L3D_helpIT3mCNT', 'L3D_helpIT3mCNTmax',
       'L3D_helpIiCNT', 'L3D_helpIiCNTmax', 'L3D_helpIi3mCNT',
       'L3D_helpIi3mCNTmax', 'L3D_helpMaCNT', 'L3D_helpMaCNTmax',
       'L3D_helpMa3mCNT', 'L3D_helpMa3mCNTmax', 'L3D_helpTeCNT',
       'L3D_helpTeCNTmax', 'L3D_helpTe3mCNT', 'L3D_helpTe3mCNTmax',
       'L3D_helpconCNT', 'L3D_helpconCNTmax', 'L3D_helpcon3mCNT',
       'L3D_helpcon3mCNTmax', 'L7D_RH', 'L7D_AH', 'L7D_re', 'L7D_fbCNT',
       'L7D_nCR', 'L7D_nMR', 'L7D_nTR', 'L7D_nconcompla', 'L7D_nconcompli',
       'L7D_nmatcompla', 'L7D_nmatcompli', 'L7D_nteccompla', 'L7D_accLIKE',
       'L7D_proLIKE', 'L7D_accDLIKE', 'L7D_proDLIKE', 'L7D_FavorCNT',
       'L7D_BlockCNT', 'L7D_AddFavorCNT', 'L7D_AddBlockCNT', 'L7D_star',
       'L7D_isIntCNT', 'L7D_LVdiffCNT', 'L7D_MGMLead', 'L7D_MGMRes',
       'L7D_MGMDemo', 'L7D_MGMDeal', 'L7D_conUUCCNT', 'L7D_conSOUCNT',
       'L7D_conSERCNT', 'L7D_conTWBCNT', 'L7D_conOTHCNT', 'L7D_conLigCNT',
       'L7D_conMedCNT', 'L7D_conTanCNT', 'L7D_conDarCNT', 'L7D_conEx01CNT',
       'L7D_conEx03CNT', 'L7D_conEx12CNT', 'L7D_conEx24CNT', 'L7D_conage30CNT',
       'L7D_conage40CNT', 'L7D_conage50CNT', 'L7D_conage60CNT',
       'L7D_conage61CNT', 'L7D_conMCNT', 'L7D_conFCNT', 'L7D_helpCNT',
       'L7D_helpCNTmax', 'L7D_help3mCNT', 'L7D_help3mCNTmax', 'L7D_helpITCNT',
       'L7D_helpITCNTmax', 'L7D_helpIT3mCNT', 'L7D_helpIT3mCNTmax',
       'L7D_helpIiCNT', 'L7D_helpIiCNTmax', 'L7D_helpIi3mCNT',
       'L7D_helpIi3mCNTmax', 'L7D_helpMaCNT', 'L7D_helpMaCNTmax',
       'L7D_helpMa3mCNT', 'L7D_helpMa3mCNTmax', 'L7D_helpTeCNT',
       'L7D_helpTeCNTmax', 'L7D_helpTe3mCNT', 'L7D_helpTe3mCNTmax',
       'L7D_helpconCNT', 'L7D_helpconCNTmax', 'L7D_helpcon3mCNT',
       'L7D_helpcon3mCNTmax', 'L14D_RH', 'L14D_AH', 'L14D_re', 'L14D_fbCNT',
       'L14D_nCR', 'L14D_nMR', 'L14D_nTR', 'L14D_nconcompla',
       'L14D_nconcompli', 'L14D_nmatcompla', 'L14D_nmatcompli',
       'L14D_nteccompla', 'L14D_accLIKE', 'L14D_proLIKE', 'L14D_accDLIKE',
       'L14D_proDLIKE', 'L14D_FavorCNT', 'L14D_BlockCNT', 'L14D_AddFavorCNT',
       'L14D_AddBlockCNT', 'L14D_star', 'L14D_isIntCNT', 'L14D_LVdiffCNT',
       'L14D_MGMLead', 'L14D_MGMRes', 'L14D_MGMDemo', 'L14D_MGMDeal',
       'L14D_conUUCCNT', 'L14D_conSOUCNT', 'L14D_conSERCNT', 'L14D_conTWBCNT',
       'L14D_conOTHCNT', 'L14D_conLigCNT', 'L14D_conMedCNT', 'L14D_conTanCNT',
       'L14D_conDarCNT', 'L14D_conEx01CNT', 'L14D_conEx03CNT',
       'L14D_conEx12CNT', 'L14D_conEx24CNT', 'L14D_conage30CNT',
       'L14D_conage40CNT', 'L14D_conage50CNT', 'L14D_conage60CNT',
       'L14D_conage61CNT', 'L14D_conMCNT', 'L14D_conFCNT', 'L14D_helpCNT',
       'L14D_helpCNTmax', 'L14D_help3mCNT', 'L14D_help3mCNTmax',
       'L14D_helpITCNT', 'L14D_helpITCNTmax', 'L14D_helpIT3mCNT',
       'L14D_helpIT3mCNTmax', 'L14D_helpIiCNT', 'L14D_helpIiCNTmax',
       'L14D_helpIi3mCNT', 'L14D_helpIi3mCNTmax', 'L14D_helpMaCNT',
       'L14D_helpMaCNTmax', 'L14D_helpMa3mCNT', 'L14D_helpMa3mCNTmax',
       'L14D_helpTeCNT', 'L14D_helpTeCNTmax', 'L14D_helpTe3mCNT',
       'L14D_helpTe3mCNTmax', 'L14D_helpconCNT', 'L14D_helpconCNTmax',
       'L14D_helpcon3mCNT', 'L14D_helpcon3mCNTmax', 'FDsellingdate',
       'product_sdate', 'mb', 'mb_STV', 'mbdate', 'WarrantyPeriod', 'mbaCNT',
       'mbadate_First', 'mbadate_New', 'MBA_technical', 'MBA_customer',
       'MBA_Scheduling', 'MBA_Class', 'MBA_Service'])
    #df = pd.read_csv('D:/karma5.0/testing.csv',names=df_head.columns.tolist())
    #df_head = pd.read_csv('D:/karma5.0/head.csv')
    #encode_location = input()
    #df_encoding =pd.read_csv(encode_location)
    df_encoding =pd.read_csv('./Model_package/Target_Encoder_Table/Encode_table.csv')
    print('Loading...')
    
    y = df_encoding['mb']
    X = df_encoding.drop(['mb'],axis=1)
    te = TargetEncoder(cols=['DuringMBA', 'RH', 'AH', 're', 'fbCNT',
           'nCR', 'nMR', 'nTR', 'nconcompla', 'nmatcompla','nteccompla', 
            'isIntCNT','ALL_RH', 'ALL_AH', 'ALL_re', 'ALL_fbCNT', 'ALL_nCR', 'ALL_nMR',
           'ALL_nTR', 'ALL_nconcompla', 'ALL_nmatcompla', 'ALL_nteccompla',
           'ALL_accLIKE', 'ALL_proLIKE', 'ALL_accDLIKE', 'ALL_proDLIKE',
           'ALL_FavorCNT', 'ALL_BlockCNT', 'ALL_AddFavorCNT', 'ALL_AddBlockCNT',
           'ALL_star', 'ALL_isIntCNT',
           'ALL_conUUCCNT','ALL_conSOUCNT', 'ALL_conSERCNT', 'ALL_conTWBCNT', 'ALL_conOTHCNT',
           'ALL_conLigCNT', 'ALL_conMedCNT', 'ALL_conTanCNT', 'ALL_conDarCNT',
           'ALL_conEx01CNT', 'ALL_conEx03CNT', 'ALL_conEx12CNT', 'ALL_conEx24CNT',
           'ALL_conage30CNT', 'ALL_conage40CNT', 'ALL_conage50CNT',
           'ALL_conage60CNT', 'ALL_conage61CNT', 'ALL_conMCNT', 'ALL_conFCNT',
           'ALL_helpCNT','ALL_help3mCNT',
            'mb_STV', 'mbaCNT',
           'MBA_total', 'mbadiff','mbaFtoSelltime', 'mbaNtoSelltime',
           'ALL_MGM_total','ALL_Like_total', 'ALL_DLike_total','ALL_help_total', 'ALL_help_max_total','ALL3m_help_max_total']).fit(X,y)
    
    
    print('Data Preprocessing...')
    #前處理
    df['MBA_technical'] = df['MBA_technical'].fillna(0)
    df['MBA_customer'] = df['MBA_customer'].fillna(0)
    df['MBA_Scheduling'] = df['MBA_Scheduling'].fillna(0)
    df['MBA_Class'] = df['MBA_Class'].fillna(0)
    df['MBA_Service'] = df['MBA_Service'].fillna(0)
    df['MBA_total']=df['MBA_Class']+df['MBA_Scheduling']+df['MBA_Service']+df['MBA_customer']+df['MBA_technical']
    df['mbadate_First']=pd.to_datetime(df['mbadate_First']).dt.date
    df['mbadate_New']=pd.to_datetime(df['mbadate_New']).dt.date
    df['FDsellingdate']=pd.to_datetime(df['FDsellingdate']).dt.date
    df['mbdate']=pd.to_datetime(df['mbdate']).dt.date
    df['DATE']=pd.to_datetime(df['DATE']).dt.date
    df['mbadiff'] = df['mbadate_New']-df['mbadate_First']
    df['mbtime'] = df['mbdate']-df['FDsellingdate']
    df['mbaFtoSelltime'] = df['mbadate_First']-df['FDsellingdate']
    df['mbaNtoSelltime'] = df['mbadate_New']-df['FDsellingdate']
    df['mbFdiff'] = df['mbdate']-df['mbadate_First']
    df['mbNdiff'] = df['mbdate']-df['mbadate_New']
    #'MBA_technical','MBA_customer','MBA_Scheduling','MBA_Class','MBA_Service',
    df = df.drop(['ALL_nconcompli','ALL_nmatcompli','mbadate_First','mbadate_New','FDsellingdate','WarrantyPeriod','product_sdate'],axis=1)
    df['mbadiff'] = (df['mbadiff'] / np.timedelta64(1, 'D')).astype(float)
    df['mbaFtoSelltime'] = (df['mbaFtoSelltime'] / np.timedelta64(1, 'D')).astype(float)
    df['mbaNtoSelltime'] = (df['mbaNtoSelltime'] / np.timedelta64(1, 'D')).astype(float)
    #df = df[df['mbaFtoSelltime']>=0.0]
    #df = df[df['mbaNtoSelltime']>=0.0]
    df['RH'] = df['RH'].fillna(0)
    df['AH'] = df['AH'].fillna(0)
    df['re'] = df['re'].fillna(0)
    df['fbCNT'] = df['fbCNT'].fillna(0)
    df['nCR'] = df['nCR'].fillna(0)
    df['nMR'] = df['nMR'].fillna(0)
    df['nTR'] = df['nTR'].fillna(0)
    df['nconcompla'] = df['nconcompla'].fillna(0)
    df['nmatcompla'] = df['nmatcompla'].fillna(0)
    df['nteccompla'] = df['nteccompla'].fillna(0)
    df['star'] = df['star'].fillna(1)
    #df = df[df['star']<=5.0]
    df['ALL_star'] = df['ALL_star'].fillna(1)
    #df = df[df['ALL_star']<=10.0]
    df['isIntCNT'] = df['isIntCNT'].fillna(0)
    df['ALL_RH'] = df['ALL_RH'].fillna(0)
    df['ALL_AH'] = df['ALL_AH'].fillna(0)
    df['ALL_re'] = df['ALL_re'].fillna(0)
    df['ALL_fbCNT'] = df['ALL_fbCNT'].fillna(0)
    df['ALL_nCR'] = df['ALL_nCR'].fillna(1)
    df['ALL_nMR'] = df['ALL_nMR'].fillna(1)
    df['ALL_nTR'] = df['ALL_nTR'].fillna(1)
    df['ALL_nconcompla'] = df['ALL_nconcompla'].fillna(1)
    df['ALL_nmatcompla'] = df['ALL_nmatcompla'].fillna(1)
    df['ALL_nteccompla'] = df['ALL_nteccompla'].fillna(1)
    df['ALL_isIntCNT'] = df['ALL_isIntCNT'].fillna(0)
    #df['MGM_total'] = df['MGMLead']+df['MGMRes']+df['MGMDemo']+df['MGMDeal']
    df['ALL_MGM_total']= df['ALL_MGMLead']+df['ALL_MGMRes']+df['ALL_MGMDemo']+df['ALL_MGMDeal']
    #df['Like_total'] = df['accLIKE']+df['proLIKE']
    df['ALL_Like_total'] = df['ALL_accLIKE']+df['ALL_proLIKE']
    #df['DLike_total'] = df['accDLIKE']+df['proDLIKE']
    df['ALL_DLike_total'] = df['ALL_accDLIKE']+df['ALL_proDLIKE']
    #df['help_total'] = df['helpITCNT']+df['helpIiCNT']+df['helpTeCNT']+df['helpMaCNT']+df['helpconCNT']
    df['ALL_help_total'] = df['ALL_helpITCNT']+df['ALL_helpIiCNT']+df['ALL_helpTeCNT']+df['ALL_helpMaCNT']+df['ALL_helpconCNT']
    #df['help_max_total'] = df['helpITCNTmax']+df['helpIiCNTmax']+df['helpTeCNTmax']+df['helpMaCNTmax']+df['helpconCNTmax']
    df['ALL_help_max_total'] = df['ALL_helpITCNTmax']+df['ALL_helpIiCNTmax']+df['ALL_helpTeCNTmax']+df['ALL_helpMaCNTmax']+df['ALL_helpconCNTmax']
    #df['3mhelp_max_total'] = df['helpIT3mCNTmax']+df['helpIi3mCNTmax']+df['helpTe3mCNTmax']+df['helpMa3mCNTmax']+df['helpcon3mCNTmax']
    df['ALL3m_help_max_total'] = df['ALL_helpIT3mCNTmax']+df['ALL_helpIi3mCNTmax']+df['ALL_helpTe3mCNTmax']+df['ALL_helpMa3mCNTmax']+df['ALL_helpcon3mCNTmax']
    df = df.fillna(0)

    print('Model Inference...')
    record = df[['DDwFD', 'contractsn', 'DATE', 'DuringMBA', 'RH', 'AH', 're', 'fbCNT',
           'nCR', 'nMR', 'nTR', 'nconcompla', 'nmatcompla','nteccompla', 
            'isIntCNT','ALL_RH', 'ALL_AH', 'ALL_re', 'ALL_fbCNT', 'ALL_nCR', 'ALL_nMR',
           'ALL_nTR', 'ALL_nconcompla', 'ALL_nmatcompla', 'ALL_nteccompla',
           'ALL_accLIKE', 'ALL_proLIKE', 'ALL_accDLIKE', 'ALL_proDLIKE',
           'ALL_FavorCNT', 'ALL_BlockCNT', 'ALL_AddFavorCNT', 'ALL_AddBlockCNT',
           'ALL_star', 'ALL_isIntCNT',
           'ALL_conUUCCNT','ALL_conSOUCNT', 'ALL_conSERCNT', 'ALL_conTWBCNT', 'ALL_conOTHCNT',
           'ALL_conLigCNT', 'ALL_conMedCNT', 'ALL_conTanCNT', 'ALL_conDarCNT',
           'ALL_conEx01CNT', 'ALL_conEx03CNT', 'ALL_conEx12CNT', 'ALL_conEx24CNT',
           'ALL_conage30CNT', 'ALL_conage40CNT', 'ALL_conage50CNT',
           'ALL_conage60CNT', 'ALL_conage61CNT', 'ALL_conMCNT', 'ALL_conFCNT',
           'ALL_helpCNT','ALL_help3mCNT',
           'mb_STV', 'mbaCNT',
           'MBA_total', 'mbadiff','mbaFtoSelltime', 'mbaNtoSelltime', 'MBA_technical', 'MBA_customer','MBA_Scheduling', 'MBA_Class', 'MBA_Service',
           'ALL_MGM_total','ALL_Like_total', 'ALL_DLike_total','ALL_help_total', 'ALL_help_max_total','ALL3m_help_max_total']]
    
    
    
    print('Loding Model...')
    #模型分三個區段,分別輸入
    ocs1 = joblib.load('./Model_package/0to20/OCS_First.pkl')
    ocs2 = joblib.load('./Model_package/21to40/OCS_Second.pkl')
    ocs3 = joblib.load('./Model_package/41to60/OCS_third.pkl')
    
    #設定df時間區段
    tr1 = record['DDwFD']<=20
    tr2 = record['DDwFD']>20
    tr3 = record['DDwFD']<41
    tr4 = record['DDwFD']>=41

    #if 0 ~20
    df1 = record[tr1]
    if df1.shape[0] !=0:
        X1 = te.transform(df1[['DuringMBA', 'RH', 'AH', 're', 'fbCNT',
               'nCR', 'nMR', 'nTR', 'nconcompla', 'nmatcompla','nteccompla', 
                'isIntCNT','ALL_RH', 'ALL_AH', 'ALL_re', 'ALL_fbCNT', 'ALL_nCR', 'ALL_nMR',
               'ALL_nTR', 'ALL_nconcompla', 'ALL_nmatcompla', 'ALL_nteccompla',
               'ALL_accLIKE', 'ALL_proLIKE', 'ALL_accDLIKE', 'ALL_proDLIKE',
               'ALL_FavorCNT', 'ALL_BlockCNT', 'ALL_AddFavorCNT', 'ALL_AddBlockCNT',
               'ALL_star', 'ALL_isIntCNT',
               'ALL_conUUCCNT','ALL_conSOUCNT', 'ALL_conSERCNT', 'ALL_conTWBCNT', 'ALL_conOTHCNT',
               'ALL_conLigCNT', 'ALL_conMedCNT', 'ALL_conTanCNT', 'ALL_conDarCNT',
               'ALL_conEx01CNT', 'ALL_conEx03CNT', 'ALL_conEx12CNT', 'ALL_conEx24CNT',
               'ALL_conage30CNT', 'ALL_conage40CNT', 'ALL_conage50CNT',
               'ALL_conage60CNT', 'ALL_conage61CNT', 'ALL_conMCNT', 'ALL_conFCNT',
               'ALL_helpCNT','ALL_help3mCNT',
               'mb_STV', 'mbaCNT',
               'MBA_total', 'mbadiff','mbaFtoSelltime', 'mbaNtoSelltime',
               'ALL_MGM_total','ALL_Like_total', 'ALL_DLike_total','ALL_help_total', 'ALL_help_max_total','ALL3m_help_max_total']])
        df1['target'] = ocs1.predict(X1)    

    #if 21 ~40
    df2 = record[tr2 & tr3]

    if df2.shape[0] !=0:
        X2 = te.transform(df2[['DuringMBA', 'RH', 'AH', 're', 'fbCNT',
               'nCR', 'nMR', 'nTR', 'nconcompla', 'nmatcompla','nteccompla', 
                'isIntCNT','ALL_RH', 'ALL_AH', 'ALL_re', 'ALL_fbCNT', 'ALL_nCR', 'ALL_nMR',
               'ALL_nTR', 'ALL_nconcompla', 'ALL_nmatcompla', 'ALL_nteccompla',
               'ALL_accLIKE', 'ALL_proLIKE', 'ALL_accDLIKE', 'ALL_proDLIKE',
               'ALL_FavorCNT', 'ALL_BlockCNT', 'ALL_AddFavorCNT', 'ALL_AddBlockCNT',
               'ALL_star', 'ALL_isIntCNT',
               'ALL_conUUCCNT','ALL_conSOUCNT', 'ALL_conSERCNT', 'ALL_conTWBCNT', 'ALL_conOTHCNT',
               'ALL_conLigCNT', 'ALL_conMedCNT', 'ALL_conTanCNT', 'ALL_conDarCNT',
               'ALL_conEx01CNT', 'ALL_conEx03CNT', 'ALL_conEx12CNT', 'ALL_conEx24CNT',
               'ALL_conage30CNT', 'ALL_conage40CNT', 'ALL_conage50CNT',
               'ALL_conage60CNT', 'ALL_conage61CNT', 'ALL_conMCNT', 'ALL_conFCNT',
               'ALL_helpCNT','ALL_help3mCNT',
               'mb_STV', 'mbaCNT',
               'MBA_total', 'mbadiff','mbaFtoSelltime', 'mbaNtoSelltime',
               'ALL_MGM_total','ALL_Like_total', 'ALL_DLike_total','ALL_help_total', 'ALL_help_max_total','ALL3m_help_max_total']])
        df2['target'] = ocs2.predict(X2)

    #if 41 ~60
    df3 = record[tr4]
    if df3.shape[0] !=0:
        X3 = te.transform(df3[['DuringMBA', 'RH', 'AH', 're', 'fbCNT',
               'nCR', 'nMR', 'nTR', 'nconcompla', 'nmatcompla','nteccompla', 
                'isIntCNT','ALL_RH', 'ALL_AH', 'ALL_re', 'ALL_fbCNT', 'ALL_nCR', 'ALL_nMR',
               'ALL_nTR', 'ALL_nconcompla', 'ALL_nmatcompla', 'ALL_nteccompla',
               'ALL_accLIKE', 'ALL_proLIKE', 'ALL_accDLIKE', 'ALL_proDLIKE',
               'ALL_FavorCNT', 'ALL_BlockCNT', 'ALL_AddFavorCNT', 'ALL_AddBlockCNT',
               'ALL_star', 'ALL_isIntCNT',
               'ALL_conUUCCNT','ALL_conSOUCNT', 'ALL_conSERCNT', 'ALL_conTWBCNT', 'ALL_conOTHCNT',
               'ALL_conLigCNT', 'ALL_conMedCNT', 'ALL_conTanCNT', 'ALL_conDarCNT',
               'ALL_conEx01CNT', 'ALL_conEx03CNT', 'ALL_conEx12CNT', 'ALL_conEx24CNT',
               'ALL_conage30CNT', 'ALL_conage40CNT', 'ALL_conage50CNT',
               'ALL_conage60CNT', 'ALL_conage61CNT', 'ALL_conMCNT', 'ALL_conFCNT',
               'ALL_helpCNT','ALL_help3mCNT',
               'mb_STV', 'mbaCNT',
               'MBA_total', 'mbadiff','mbaFtoSelltime', 'mbaNtoSelltime',
               'ALL_MGM_total','ALL_Like_total', 'ALL_DLike_total','ALL_help_total', 'ALL_help_max_total','ALL3m_help_max_total']])
        df3['target'] = ocs3.predict(X3)
   
    print('Inference Complete')
       
    summary = pd.concat([df1,df2,df3],ignore_index=True,sort=False)
    result = summary[['contractsn','DuringMBA', 'RH', 'AH', 're', 'fbCNT',
                      'nCR', 'nMR', 'nTR', 'nconcompla', 'nmatcompla','nteccompla', 
                      'isIntCNT','mb_STV', 'mbaCNT','MBA_technical', 'MBA_customer','MBA_Scheduling', 'MBA_Class', 'MBA_Service','MBA_total','target']]
    
    result['target'][result['target']==1]=0
    result['target'][result['target']==-1]=1
    
    answer = result[['contractsn','DuringMBA', 'RH', 'AH', 're', 'fbCNT',
                     'mb_STV','MBA_total','MBA_technical', 'MBA_customer','MBA_Scheduling', 'MBA_Class', 'MBA_Service','target']][result['target']==1]
    
    risk = result[['contractsn','MBA_technical', 'MBA_customer','MBA_Scheduling', 'MBA_Class', 'MBA_Service','mb_STV']][result['target']==1]
    
    L = result['contractsn'][result['target']==1].tolist()
    df_list = pd.DataFrame(L,columns=['Risk Contract sn'])
    final = datetime.now()
    print('完成时间:', final)
    t = final - start
    print('所需时间:', t)
    
    
    
    print('结果输出至 Client_Karma_P2_py/Results/')
    
    tt = date.today()
    
    path = './Results/'+str(tt)
    
    if not os.path.isdir(path):
        os.mkdir(path)
        
    result.to_csv('./Results/'+str(tt)+'/'+'Detai_Result.csv',index=False)
    answer.to_csv('./Results/'+str(tt)+'/'+'Risk_List(Detail).csv',index=False)
    risk.to_csv('./Results/'+str(tt)+'/'+'Risk_List(MBA_Information).csv',index=False)
    df_list.to_csv('./Results/'+str(tt)+'/'+'Risk_Contractsn.csv',index=False)
    #result.to_sql('table_name2',engine)
    
    
    
if __name__ == '__main__':
        
    Clinet_Karma_Inference()

