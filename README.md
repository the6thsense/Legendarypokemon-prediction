# Legendarypokemon-prediction
The only way to identify a Legendary Pokémon is through statements from official media, such as the game or anime. I am trying to apply deep learning to predict legendary data of a small data set to an astronishing accuracy of  nearly 98.125 percent. 

#### data transformations followed:

1. this data is about pokemon that gives values of various attributes related to each of them. These attributes include 
  a) type 1
  b) type 2
  c) generation of pokemon
  d) sp. attack 
  e) sp. defense
  f) defense
  e) attack
  g) HP
  h) also contains name of pokemon
  i) and it's legendarity

But how is this data related to legendarity of the pokemon. scrolling across net I arrived to various conlusions about various attibutes as follows:

types : a Type can take 18 different values. To deal with this categorical data I label encoded it using the sklearn library's labelencoder function present in its preprocessing library.

Generation: latest present on data is 6th and other are from 1-5.
sp. attack, sp. def, defense, defense,attack : They are ranked relatively in numbers 

HP: Base Health Points ,so HP is kind of like a measure of your Pokémon's stamina and health.

now there are 2 classes of legendary that is False and True, thus converting them to dummy values of 0 and 1 using pandas' Get_dummies function.

when we finally have a clean data .. now let us analyse it.

so i checked for correlation matric of the data. soon I realised that strongest correlations with legendarity were found with sp. attack and sp. Defense. i added both the features namely sp. atk and attack to get total attacking potential of hte pokemons and again finding the correlation we achieved a variable having highest correlation. This was not the case with sp. def and defense but I added them also as they were showing a significant linearity among themselves clearly visible through scatter matrix between them.  

Next thing about the dataset was removing the High leverage points that may prove to be outliers when we train out neural network. So i used sk learn's Robust scaler which scales data without being affected by high leverage points using quartile range The centering and scaling statistics of this scaler are based on percentiles and are therefore not influenced by a few number of very large marginal outliers.

now feeding this to neural network we get along the best accuracy of 98.125 percent test accuracy. that is really appreciable against Towards datascience' article on this data sets which finds the accuracy to be 96 percent using Random Forest Algorithm.

