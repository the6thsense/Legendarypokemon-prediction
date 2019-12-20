#!/usr/bin/env python
# coding: utf-8

# In[334]:


import pandas
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from  torch.utils.data import DataLoader as  dataloader
import sklearn as skl
import sklearn.preprocessing as ppr


# In[335]:


import pandas as pd
import random


# In[336]:


import numpy as np


# In[337]:


pokemon = pandas.read_csv( "/home/sanchit/Desktop/starting_ML/pokemon_data.csv")


# In[338]:


pokemon.head()


# In[339]:


pokemon_names = pokemon["Name"].copy()


# In[340]:


pokemon.drop(["Name", "#"],1, inplace = True) #removing unwanted columns


# In[341]:


pokemon.corr()["Legendary"].sort_values(ascending = False)


# ##### since we can see a strong correlation in sp. attack as well as with attack 
# ##### we will add both to get a new predictor that is sum of 2
# 

# In[342]:


pokemon["total_attack"] = pokemon["Sp. Atk"] + pokemon["Attack"]


# In[343]:


pokemon.corr()


# ###### so now  we drop sp. Atk and Attack rows in our data

# In[344]:


pokemon.drop(["Sp. Atk", "Attack"], 1, inplace = True)


# In[345]:


#checking linearity

pandas.plotting.scatter_matrix(pokemon.drop("Legendary", 1), figsize = (20,20))


# ###### sp. def and defence have certain relation 

# In[346]:


# adding to get total defence


# In[347]:


pokemon["total defence"] = pokemon["Sp. Def"] + pokemon["Defense"]


# In[348]:


pokemon.drop(["Sp. Def", "Defense"],1,inplace =True )


# In[349]:


pokemon.head()


# In[350]:


#replacing type 1 and type 2 with label encoder
pokemon.info()


# In[351]:


print(pokemon["Type 2"].unique().__len__())
pokemon["Type 1"]


# In[352]:


#so we use label enoder to fit on type 2 data and use that fit to transfroms both type 2 and type 1

x = ppr.LabelEncoder()
x.fit_transform(pokemon[ "Type 2"])

#UNABLE TO REPLACE NAN


# In[353]:


pokemon["Type 2"].replace( np.NaN, "aaa", inplace= True)


# In[354]:


pokemon ["Type 2"]  = pd.DataFrame(x.fit_transform(pokemon["Type 2"]))


# In[355]:


pokemon ["Type 1"]  = pd.DataFrame(x.transform(pokemon["Type 1"]))


# In[356]:


pokemon.head()


# In[357]:


label = pd.get_dummies(pokemon["Legendary"],prefix = "Legen")


# In[358]:


pokemon.drop( "Legendary",1, inplace = True)


# In[359]:


pokemon.head()


# In[360]:


pd.plotting.scatter_matrix(pokemon, figsize = (20,20))


# In[362]:


scale = ppr.RobustScaler()
columns = pokemon.columns
scaled_pokemon = pd.DataFrame(scale.fit_transform( pokemon), columns = columns )


# In[383]:


scaled_pokemon.head()


# In[364]:


pd.plotting.scatter_matrix(pokemon, figsize = (20,20))


# In[365]:


from sklearn.model_selection import StratifiedShuffleSplit as sss


# In[366]:


scaled_pokemon.__len__()


# In[381]:


k = sss( n_splits = 1 , test_size = 0.2, train_size = 0.8 )

train_idx , test_idx = list(tuple(k.split( scaled_pokemon,label))[0][0]),list(tuple(k.split( scaled_pokemon,label))[0][1])


# In[384]:


pokemon_tensor = torch.utils.data.TensorDataset( torch.tensor( np.array(scaled_pokemon) ), torch.tensor(np.array(label)))


# In[927]:


def split_data( datasets, train_idx, test_idx, samplers = torch.utils.data.SubsetRandomSampler , batchsize = 200 ):
    
    test_smp, train_smp = samplers(test_idx),samplers(train_idx)
    
    # returns  train_loader, valid_loader, test_loader 
    
    train_loader,  test_loader = dataloader(datasets, sampler = train_smp, batch_size = batchsize ),dataloader(datasets, sampler = test_smp,batch_size = batchsize )
    return  train_loader, test_loader 


# In[928]:


train_loader, test_loader = split_data( pokemon_tensor, train_idx, test_idx)


# In[457]:


from torch import nn

class work( nn.Module ):
    
    def __init__( self, lr_  ):
        
        super().__init__()
        self.model = nn.Sequential( nn.Linear( 7, 6),
                             nn.Tanh(),
                                   nn.Linear(  6,2),
                             nn.Softmax()  )
        self.lr = lr_
        self.optimizer =  torch.optim.Adam( self.model.parameters(), lr = self.lr )

        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self,data):
        pass
       
    def train( self, train_loader, epochs):
        losses = []
        correct0 = 0
        for epoch in range(epochs):
            self.optimizer =  torch.optim.Adam( self.model.parameters(), lr = self.lr)
            for train_data, labels in train_loader:
                self.optimizer.zero_grad()

                out = self.model( train_data.float() )
                
                target = labels.max(1)[1]
                loss = self.criterion( out, target)
                
                
                loss.backward()
                self.optimizer.step()
                losses.append( loss.item() )
            
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
                
                for i in range(len(x)):
                    if x[i]== 0:
                        wrong[(label[i]) = data[i]
                total += len( label)
                
            return correct,total, wrong

        


# In[932]:


a = work( 0.1 )
plt.figure(figsize = (10,10))
plt.plot(a.train( train_loader, 80))
prediction = a.predict( test_loader)
prediction[1],prediction[0]


# In[933]:


correct_results = prediction[0]
total = prediction[1]
wrong_ones = prediction[2]
test_accuracy = (float(correct_results)/float(total))*100
print(test_accuracy)


# In[936]:


(float(a.predict(train_loader)[0])*100/float(a.predict(train_loader)[1]))


# In[845]:


pd.DataFrame(wrong_ones).transpose()


# ##  test accuracy  - 98.125%
# ## train accuracy - 97.8125%
