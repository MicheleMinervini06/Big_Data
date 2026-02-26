from sklearn.ensemble import RandomForestClassifier
from src.models.custom_base_estimators import VeroResNet, NeuralNetworkFitter, CNNFitterInput, ImageRFFitter, ImageRFFitterInput
from src.models.lutech_models import IRBoostSH
import torch
import numpy as np
import pandas as pd


def get_clinical_cost_matrix():
    """
    Get the default clinical cost matrix for CN/MCI/AD classification.
    
    Returns:
        cost_matrix: 3x3 numpy array where cost_matrix[i,j] is the cost of
                    predicting class j when the true class is i
    """
    # Default clinical cost matrix for CN/MCI/AD
    # Rows = true class (CN=0, MCI=1, AD=2)
    # Cols = predicted class (CN=0, MCI=1, AD=2)
    cost_matrix = np.array([
        [0.0, 0.3, 0.9],  # True CN: CN→AD most expensive (0.9)
        [0.5, 0.0, 0.7],  # True MCI: moderate costs
        [1.0, 0.8, 0.0]   # True AD: AD→CN CRITICAL (1.0)
    ])
    
    return cost_matrix


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

    # Check if cost-sensitive training is enabled
    train_cost_sensitive = params.get('train_cost_sensitive', False)
    cost_matrix = None
    
    if train_cost_sensitive:
        print("\n[COST-SENSITIVE TRAINING] Enabled - minimizing clinical costs during boosting...")
        cost_matrix = get_clinical_cost_matrix()
        print("Clinical Cost Matrix:")
        print("     CN   MCI   AD")
        print(f"CN  {cost_matrix[0, 0]:.1f}  {cost_matrix[0, 1]:.1f}  {cost_matrix[0, 2]:.1f}")
        print(f"MCI {cost_matrix[1, 0]:.1f}  {cost_matrix[1, 1]:.1f}  {cost_matrix[1, 2]:.1f}")
        print(f"AD  {cost_matrix[2, 0]:.1f}  {cost_matrix[2, 1]:.1f}  {cost_matrix[2, 2]:.1f}\n")
    
    ir_boost.fit(X_mods, y_train, mod, sample_weights=None, cost_matrix=cost_matrix)

    return ir_boost



