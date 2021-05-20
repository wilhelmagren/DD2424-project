# DD2424 Deep Learning in Data Science
This is the official git repository for the (group 70) project work in the course DD2424.

## Static Evaluation of Chess Positions with Convolutional Neural Networks
The field of deep learning has made a huge impact on today's state-of-the-art chess engines. 
Limited look-ahead and Monte Carlo tree search approaches to minimax has greatly increased chess engines' 
abilities to evaluate positions through optimized search algorithms. Human chess grandmasters have a 
great ability to look ahead at every move, much like the chess engines, but what they also possess is 
an ability of *static evaluation*. The aforementioned chess engines heavily rely on looking at 
future moves in order to evaluate current positions, but how well can you create a model for 
static evaluation? This report presents an approach to static evaluation by utilizing a 
Convolutional Neural Network (CNN) with no prior knowledge of the rules of chess.

The problem of static evaluation can be defined as a regression problem, but an easier starting 
point is to create a model suited for classification. This reasoning behind this approach was that it 
would be easier to determine the correctness of the proposed model. Both binary- and categorical 
classification was tested in order to determine the capabilities and correctness of the model. 
An initial model was experimented on but results indicated that it was too complex and demanded 
both too much time and data to train. The second model proposed was less complex, had a more 
stable training/evaluation procedure, and was much faster to train. This model achieved a 
testing accuracy of 83.80\% on the binary classification task, and a 67.66\% testing accuracy on 
the categorical classification task (7 classes). The second proposed model was deemed suitable 
and we moved on to tackling the initial problem of static evaluation, using a regression model.

The regression model performs well, although it might be slightly underfit on certain positions 
of chess due to an imbalance in the generated data. The model is in general bad at noticing the 
features of tactics in positions, but does a pretty good job at giving a fair centipawn evalution. 
The model was not expected to give an accurate centipawn score compared to the state-of-the-art 
chess engines whenever there are tactics present in the position. This is due to the nature of 
the static evalution, and the fact that our model both can't and won't look into future positions; 
which is something that tactical positions require in order to give an accurate evaluation.  

### Final model results, CNN vs Stockfish 13
![] (images/README/rm_comp_1.png)


**Authors:** Eric Bröndum, Christoffer Torgilsman, Wilhelm Ågren