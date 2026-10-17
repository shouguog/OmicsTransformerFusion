""" Example for MOSGAT classification
"""
from train_test_withname import train_test


if __name__ == "__main__":
    data_folders = ['KIPANperm01','LGGperm01','ROSMAPperm01','BRCAperm01']
    for data_folder in data_folders:
        testonly = False #Shouguo Gao True
        view_list = [1,2,3]
        num_epoch_pretrain = 500
        num_epoch = 1000
        lr_e_pretrain = 1e-3
        lr_e = 5e-4
        lr_c = 1e-4
    
        if data_folder == 'ROSMAPperm01':
            num_class = 2
        if data_folder == 'LGGperm01':
            num_class = 2
        if data_folder == 'KIPANperm01':
            num_class = 3
        if data_folder == 'BRCAperm01':
           num_class = 5
        models = ['GIN', 'GAT', 'SAGE']
        for modelname in models:
            train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, testonly,
               num_epoch_pretrain, num_epoch, encoder=modelname,
               LOGFILE="resultsperm01/" + data_folder + "_" + modelname + ".log",modelfolder = "modelperm01")
            #train_test(data_folder, view_list, num_class,
            #   lr_e_pretrain, lr_e, testonly,
            #   num_epoch_pretrain, num_epoch, encoder="GAT", LOGFILE="results/" + data_folder + "_GAT.log")
            #train_test(data_folder, view_list, num_class,
            #   lr_e_pretrain, lr_e, testonly,
            #   num_epoch_pretrain, num_epoch, encoder="SAGE", LOGFILE="results/" + data_folder + "_SAGE.log")
print("Hello World Finished")
