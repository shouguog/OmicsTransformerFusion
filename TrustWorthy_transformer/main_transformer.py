from torch import true_divide
from train_test_transformer import train

if __name__ == "__main__":    
    testonly = False #False #Shouguo True
    modelpath = './model_transformer/'
    #print("#####BRCA#####")
    #data_folder = 'BRCA'
    #train(data_folder, modelpath, testonly)

    #print("#####ROSMAP#####")
    #data_folder = 'ROSMAP'
    #train(data_folder, modelpath, testonly)

    print("#####KIPAN#####")
    data_folder = 'KIPAN'
    train(data_folder, modelpath, testonly)

