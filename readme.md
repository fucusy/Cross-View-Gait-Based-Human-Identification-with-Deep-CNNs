# Introduction
Code for *2016 TPAMI(IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE) A Comprehensive Study on Cross-View Gait Based Human Identification with Deep CNNs*

# Data
## Prepare the dataset
get gait dataset from `http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitLP.html`,
use the Version 1, OULP-C1V1, `http://www.am.sanken.osaka-u.ac.jp/BiometricDB/doc/OULP_Doc01a_SubsetStatistics_OULP-C1V1.pdf`,
download the dataset into path: `~/data/`, rename the directory to `OULP_C1V1_Pack`,
you can see directories `OULP-C1V1_NormalizedSilhouette(88x128)` and `OULP-C1V1_SubjectIDList(FormatVersion1.0)` in `~/data/OULP_C1V1_Pack`.

## Prepare training, validation and test dataset
protocol, described in paper: *Cross-View Gait Recognition Using View-Dependent Discriminative Analysis*.
In this experiments, training set described in the paper was split into traning and validation part, in `OULP_setting/list_train.txt`
and `OULP_setting/list_val.txt`

1. change current directory to here
1. move OULP_setting directory to `~/data/OULP_setting`, run command line: `cp -r OULP_setting ~/data/OULP_setting`
2. run command line:`python preprocess_script/oulp_prepare.py ~/data/gait-oulp-c1v1`

## Running the code

Command line example to run the code

	th main.lua -datapath ~/data/gait-oulp-c1v1 -mode train  -modelname wuzifeng -gpu -gpudevice 1  -dropout 0.5 -learningrate 1e-3 -momentum 0.9 -calprecision 2 -calval 1 -batchsize 64 -iteration 2000000 >> main.lua.log

# Result

you will see the validation average precision up to 92.50.

you will see results in main.lua.log which like below:

    {
      iteration : 2000000
      seed : 1
      loadmodel : ""
      datapart : "test"
      batchsize : 64
      debug : false
      gpu : true
      gpudevice : 1
      modelname : "wuzifeng"
      calval : 1
      momentum : 0.9
      datapath : "/home/chenqiang/data/gait-oulp-c1v1"
      gradclip : 5
      dropout : 0.5
      learningrate : 0.001
      calprecision : 2
      mode : "train"
    }
    2017-05-23 15:53:51[INFO] load data from /home/chenqiang/data/gait-oulp-c1v1/oulp_train_data.txt, /home/chenqiang/data/gait-oulp-c1v1/oulp_val_data.txt, /home/chenqiang/data/gait-oulp-c1v1/oulp_test_data.txt	
    2017-05-23 15:53:51[INFO] train data instances 06848, uniq  0856	
    2017-05-23 15:53:51[INFO] train data instances 00800, uniq  0100	
    2017-05-23 15:53:51[INFO] train data instances 07648, uniq  0956	
    nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ParallelTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
          |      (1): nn.SpatialConvolutionMM(1 -> 16, 7x7)
          |      (2): nn.ReLU
          |      (3): nn.SpatialCrossMapLRN
          |      (4): nn.SpatialMaxPooling(2x2, 2,2)
          |      (5): nn.SpatialConvolutionMM(16 -> 64, 7x7)
          |      (6): nn.ReLU
          |      (7): nn.SpatialCrossMapLRN
          |      (8): nn.SpatialMaxPooling(2x2, 2,2)
          |    }
           `-> (2): nn.Sequential {
                 [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
                 (1): nn.SpatialConvolutionMM(1 -> 16, 7x7)
                 (2): nn.ReLU
                 (3): nn.SpatialCrossMapLRN
                 (4): nn.SpatialMaxPooling(2x2, 2,2)
                 (5): nn.SpatialConvolutionMM(16 -> 64, 7x7)
                 (6): nn.ReLU
                 (7): nn.SpatialCrossMapLRN
                 (8): nn.SpatialMaxPooling(2x2, 2,2)
               }
           ... -> output
      }
      (2): nn.Sequential {
        [input -> (1) -> (2) -> output]
        (1): nn.CSubTable
        (2): nn.Abs
      }
      (3): nn.Sequential {
        [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
        (1): nn.SpatialConvolutionMM(64 -> 256, 7x7)
        (2): nn.Reshape(1x112896)
        (3): nn.Dropout(0.5, busy)
        (4): nn.Linear(112896 -> 2)
        (5): nn.LogSoftMax
      }
    }
    2017-05-23 15:53:52[INFO] Number of parameters:1079906	
    2017-05-23 15:54:02[INFO] 00001th/2000000 Val Error 0.693167	
    2017-05-23 15:54:12[INFO] 00001th/2000000 Tes Error 0.693317	
    2017-05-23 15:54:21[INFO] 00001th/2000000 Tra Error 0.693182, 29	
    2017-05-23 15:54:31[INFO] 00065th/2000000 Val Error 0.693064	
    2017-05-23 15:54:40[INFO] 00065th/2000000 Tes Error 0.693260	
    2017-05-23 15:54:49[INFO] 00065th/2000000 Tra Error 0.693897, 27	


# Test

select the model file in trainedNet as a argument of -loadmodel when run main.lua, then you can see the test result in the redirected file.
the best average recognition precision you can get is  88.29.

	th main.lua -datapath ~/data/gait-oulp-c1v1 -mode evaluate -datapart test  -modelname wuzifeng -gpu -gpudevice 1 -loadmodel ./trainedNets/wuzifeng_tra_0.6666_i7745.t7 >> main.lua.result.log