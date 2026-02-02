from sklearn.ensemble import RandomForestClassifier
from src.models.custom_base_estimators import VeroResNet, NeuralNetworkFitter, CNNFitterInput, ImageRFFitter, ImageRFFitterInput
from src.models.lutech_models import IRBoostSH
import torch
import numpy as np


def training_function(X_mods,y_train,fold,params):

    epochs = params.get("epochs",50)
    mb_size_train = params.get("mb_size_train",20)
    n_iteration = params.get("n_iteration",10)
    frezze_layer = params.get("frezze_layer",2)
    mod = params.get("mod",None)

    if params.get("id") == 6:

        rf = RandomForestClassifier(10, min_samples_split=60)

        rf_fitter_input = ImageRFFitterInput(model = RandomForestClassifier(10, min_samples_split=60), temp_file=f"image_emb_fold{fold}.pkl")

        rf_trainer = ImageRFFitter(rf_fitter_input)

        base_estimators={'clinical': rf, 'images': rf_trainer}

    else:

        #Clinical model
        rf = RandomForestClassifier(10, min_samples_split=60)

        #Images
        net = VeroResNet(num_classes=3)

        cnn_fitter_input = CNNFitterInput(
            model=net,
            loss_function= torch.nn.CrossEntropyLoss,
            optimizer= torch.optim.Adam,
            learning_rate= 1e-4,
            epochs= epochs,
            mb_size= mb_size_train,
            #log = True,
            freeze_level = frezze_layer,
            #logdir= r"C:\Users\giuseppe.lamanna\Desktop\progetti\fair-lab-local\fair-lab\Veronet_algorithm\experiments"
            )

        
        nn_trainer = NeuralNetworkFitter(cnn_fitter_input)

        base_estimators={'clinical': rf, 'images': nn_trainer}

    print({
            "epochs_cnn":epochs,
            "boost_iteration":n_iteration,
            "freeze_layer":frezze_layer,
            
        })

    ir_boost = IRBoostSH(base_estimators=base_estimators, n_iter=n_iteration, learning_rate=1.)

    ir_boost.fit(X_mods, y_train, mod)

    return ir_boost



