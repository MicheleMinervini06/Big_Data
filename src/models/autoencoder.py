
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim

### Autoencoder = una rete neurale che comprime i dati di input in uno spazio più piccolo (bottleneck) e cerca di ricostruire l’input a partire da quella rappresentazione compressa 


## Define the Autoencoder architecture
class Autoencoder(nn.Module):
     def __init__(self, input_dim, hidden_dim1, bottleneck_dim, sigmoid: bool = False, noise_factor = 0):
         super(Autoencoder, self).__init__()
         

         self.noise_factor = noise_factor # noise factor for a Denoising AE

         # Encoder part (serie di Linear + ReLU che portano i dati al bottleneck)
         self.encoder = nn.Sequential(
             nn.Linear(input_dim, hidden_dim1),
             nn.ReLU(),
             nn.Linear(hidden_dim1, bottleneck_dim),
             nn.ReLU(),
         )

         # Decoder part (ricostruisce i dati partendo dal bottleneck)
         layers = [
             nn.Linear(bottleneck_dim, hidden_dim1),
             nn.ReLU(),
             nn.Linear(hidden_dim1, input_dim),
         ]

         # Add either Sigmoid or ReLU based on the flag
         if sigmoid:
             layers.append(nn.Sigmoid())

         # Define the decoder as a Sequential model
         self.decoder = nn.Sequential(*layers)


     def forward(self, x):
         if self.noise_factor != 0: # Introduce noise
             x = self._add_noise(x)
         # Encode input to latent space
         encoded = self.encoder(x)
         # Decode latent space back to input space
         decoded = self.decoder(encoded)
         return decoded


     def _add_noise(self, x): # serve per fare Denoising Autoencoder (il modello impara a ricostruire l’input originale nonostante il rumore)
         noisy_data = x.clone()
         for i in range(x.shape[1]):
             range_col = x[:, i].max() - x[:, i].min()
             noise = torch.normal(0, self.noise_factor * range_col, size=x[:, i].size())
             noisy_data[:, i] += noise
         return noisy_data


     def fit(self,data: pd.DataFrame,epochs=100):
         self.train()

         criterion = nn.MSELoss()
         optimizer = optim.Adam(self.parameters(),lr=1e-2)

         x = self._dataframe_to_tensor(data)

         for epoch in range(epochs):
             
             optimizer.zero_grad()

             outputs = self.forward(x)
             loss = criterion(outputs,x)

             loss.backward()
             optimizer.step()

             print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
     def evaluate(self, test_data: pd.DataFrame):
        #Valuta il modello sul test set e restituisce la loss MSE
        self.eval()  # Metti il modello in modalità di valutazione
        criterion = nn.MSELoss()
        x_test = self._dataframe_to_tensor(test_data)
        with torch.no_grad():  # Disabilita il calcolo del gradiente per risparmiare memoria
            outputs = self.forward(x_test)
            loss = criterion(outputs, x_test)
            print('TestLoss: {loss.item():.4f}')
 

     def _dataframe_to_tensor(self, data: pd.DataFrame):
         # Assuming you want to convert all columns to a tensor
         # Convert the dataframe to a numpy array and then to a tensor
         data_array = data.to_numpy()
         return torch.tensor(data_array, dtype=torch.float32)

     def encode(self, x):
        # Passa i dati solo attraverso l'encoder per ottenere l'embedding (bottleneck)
        self.eval()  # Imposta il modello in modalità valutazione
        with torch.no_grad():  # Disabilita il calcolo del gradiente per velocizzare
            x_tensor = self._dataframe_to_tensor(x) if isinstance(x, pd.DataFrame) else x
            return self.encoder(x_tensor)


# Il Bottleneck rappresenta la "compressione" massima dei dati
class Bottleneck(nn.Module): 
    def __init__(self, hidden_dim1, bottleneck_dim):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim1, bottleneck_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.bottleneck(x)
