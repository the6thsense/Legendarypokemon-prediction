#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from  torch.utils.data import DataLoader as  dataloader
import sklearn as skl
import sklearn.preprocessing as ppr


# In[21]:


import pandas as pd
import random


# In[22]:


import numpy as np


# In[23]:


pokemon = pandas.read_csv( "/home/sanchit/Desktop/starting_ML/pokemon_data.csv")


# In[24]:


pokemon.head()


# In[25]:


pokemon_names = pokemon["Name"].copy()


# In[26]:


pokemon.drop(["Name", "#"],1, inplace = True) #removing unwanted columns


# In[27]:


pokemon.corr()["Legendary"].sort_values(ascending = False)


# ##### since we can see a strong correlation in sp. attack as well as with attack 
# ##### we will add both to get a new predictor that is sum of 2
# 

# In[28]:


pokemon["total_attack"] = pokemon["Sp. Atk"] + pokemon["Attack"]


# In[29]:


pokemon.corr()


# ###### so now  we drop sp. Atk and Attack rows in our data

# In[30]:


pokemon.drop(["Sp. Atk", "Attack"], 1, inplace = True)


# In[31]:


#checking linearity

pandas.plotting.scatter_matrix(pokemon.drop("Legendary", 1), figsize = (20,20))


# ###### sp. def and defence have certain relation 

# In[32]:


# adding to get total defence


# In[33]:


pokemon["total defence"] = pokemon["Sp. Def"] + pokemon["Defense"]


# In[34]:


pokemon.drop(["Sp. Def", "Defense"],1,inplace =True )


# In[35]:


pokemon.head()


# In[36]:


#replacing type 1 and type 2 with label encoder
pokemon.info()


# In[37]:


print(pokemon["Type 2"].unique().__len__())
pokemon["Type 1"]


# In[38]:


#so we use label enoder to fit on type 2 data and use that fit to transfroms both type 2 and type 1

x = ppr.LabelEncoder()
x.fit_transform(pokemon[ "Type 2"])

#UNABLE TO REPLACE NAN


# In[39]:


pokemon["Type 2"].replace( np.NaN, "no_power", inplace= True)


# In[40]:


pokemon ["Type 2"]  = pd.DataFrame(x.fit_transform(pokemon["Type 2"]))


# In[41]:


pokemon ["Type 1"]  = pd.DataFrame(x.transform(pokemon["Type 1"]))


# In[42]:


pokemon.head()


# In[43]:


label = pd.get_dummies(pokemon["Legendary"],prefix = "Legen")


# In[44]:


pokemon.drop( "Legendary",1, inplace = True)


# In[45]:


pokemon.head()


# In[46]:


pd.plotting.scatter_matrix(pokemon, figsize = (20,20))


# In[47]:


scale = ppr.RobustScaler()
columns = pokemon.columns
scaled_pokemon = pd.DataFrame(scale.fit_transform( pokemon), columns = columns )


# In[48]:


scaled_pokemon.head()


# In[49]:


pd.plotting.scatter_matrix(pokemon, figsize = (20,20))


# In[50]:


from sklearn.model_selection import StratifiedShuffleSplit as sss


# In[51]:


scaled_pokemon.__len__()


# In[52]:


k = sss( n_splits = 1 , test_size = 0.2, train_size = 0.8 )

train_idx , test_idx = list(tuple(k.split( scaled_pokemon,label))[0][0]),list(tuple(k.split( scaled_pokemon,label))[0][1])


# In[53]:


pokemon_tensor = torch.utils.data.TensorDataset( torch.tensor( np.array(scaled_pokemon) ), torch.tensor(np.array(label)))


# In[54]:


def split_data( datasets, train_idx, test_idxsamplers = torch.utils.data.SubsetRandomSampler , batchsize = 200 ):
    
    test_smp, train_smp = samplers(test_idx),samplers(train_idx)
    
    # returns  train_loader, valid_loader, test_loader 
    
    train_loader,  test_loader = dataloader(datasets, sampler = train_smp, batch_size = batchsize ),dataloader(datasets, sampler = test_smp,batch_size = batchsize )
    return  train_loader, test_loader 


# In[55]:


train_loader, test_loader = split_data( pokemon_tensor, train_idx, test_idx)


# In[302]:


def deep_cross_validation(model, tensor_data, k = 5, sampler = torch.utils.data.SubsetRandomSampler, batchsize = 50):
    length = len(tensor_data)
    indices = list(range( length))
    np.random.shuffle(indices)
    k = int(length/k)
    
    k_folds = [ indices[ i : i +k] for i in range(0,length,k)]
    
    avg_acc = []
    for fold in k_folds:
        temp = k_folds
        temp.remove(fold)
        train_idx =  temp
        test_idx = fold
        
        train_loader, test_loader = split_data( tensor_data, train_idx, test_idx, batchsize = batchsize)
        
        losses =  model.train( train_loader, test_loader, 100, predictions = False)
        
        correct, total, wrong = model.predict(test_loader)
        
        avg_acc.append(correct/total)
    return np.array(avg_acc).mean()
    


# In[334]:


from torch import nn

class work( nn.Module ):
    
    def __init__( self, lr_  ):
        
        super().__init__()
        self.model = nn.Sequential( nn.Linear( 7, 60),
                                   nn.Dropout( 0.3),
                             nn.ELU(),
                                   nn.Linear(  60,2),
                             nn.Softmax()  )
        self.lr = lr_
        self.optimizer =  torch.optim.Adam( self.model.parameters(), lr = self.lr )

        self.criterion = nn.NLLLoss()
        
        self.model[1].weights = nn.init.xavier_uniform_(torch.zeros(60,7), gain=1.0)
    def forward(self,data):
        pass
       
    def train( self, train_loader, test_loader, epochs, predictions = True):
        losses = []
        train_accuracy = []
        test_accuracy = []
        for epoch in range(epochs):
            for train_data, labels in train_loader:
                
                self.optimizer.zero_grad()

                out = self.model( train_data.float() )
                
                target = labels.max(1)[1]
                loss = self.criterion( out, target)
                
                loss.backward()
                self.optimizer.step()
                losses.append( loss.item() )
            if predictions:
                train_correct, train_total, train_wrong = self.predict(train_loader)
                test_correct, test_total, test_wrong = self.predict( test_loader )

                train_accuracy.append(train_correct/train_total)
                test_accuracy.append(test_correct/test_total)
                
        if predictions:  
            return losses, train_accuracy, test_accuracy
        else:
            return losses
                    
    def predict( self,dataloader ):
        correct = 0
        total = 0
        wrong = {}
        with torch.no_grad():
            for data, labels in dataloader:
                
                _, out = torch.max( self.model( data.float()), dim = 1)
                _,label = torch.max( labels, dim =1 )
                x = out.eq( label )
                correct += x.sum()
                total += len(label)
                for i in range(len(x)):
                    if x[i]== 0:
                        wrong[(label[i])] = data[i]
                    
            return correct.item(),total, wrong

        


# In[337]:


a = work( 0.01 )
losses, train_acc, test_acc =  a.train( train_loader, test_loader, 100)

x ,b , c = a.predict( test_loader)
xt,bt,ct=  a.predict( train_loader)

print("maximum_test_accuracy_of_fully_trained_model = ", np.array(test_acc).max())

print("avg_cross_validation_accuracy =",               deep_cross_validation(a, pokemon_tensor, k = 5, sampler =                 torch.utils.data.SubsetRandomSampler,batchsize = 50))

print("test_acc =", x/b, "train_accuracy = ",xt/bt)


# In[338]:


fig = plt.subplots( 1,3, figsize = (15,5))
fig[1][0].plot(losses)
fig[1][1].plot(train_acc)
fig[1][2].plot( test_acc)


# ##  test accuracy  - 96.875%
# ## train accuracy - 96.5625%
# 
# ## Cross Val Accuracy - 95.417

# In[ ]:




