# AMNLP_CoReference_Project
In this project, we implemented an Encoder-Decoder Models for the task of coreference resolution, focusing on “news articles”. While current models are based on structured prediction, we implemented a generative sequence-to-sequence (seq2seq) model.

## Data Augmentation
**The original format** for coreference clusters is basically an array of clusters, where clusters represent Entities, and each cluster includes token indices for specific mentions referring to the same entity, which is not suitable labels for the seq2seq problem.

>'Coming next, President Bush celebrated his sixtieth birthday this week. And thirty one years ago on Meet The Press then president Gerald Ford talked about being sixty and serving as president. Coming up right here on Meet The Press.'

In this example, there are 2 coreference entities with the following mention clusters: 

>[ [‘President Bush’ , ’his’] , [‘Meet The Press’ , ‘Meet The Press’] ]


Which translates to this word indecis (neglecting the tokenization stage):

>[ [ (2 , 3) , (5 , 5) ] , [ (16 , 18) , (36 , 38) ] ]

**The new format** We used was based on data augmentation to transform the coreference clusters format into plain text format. This augmentation allowed us to create a labeled sequence, which we used to train a generative model. This language model is learned by using cross entropy loss in a next-token prediction manner. On the training process itself, we will elaborate in the next section.

>'Coming next,<start><id_1> President Bush<id_1><end> celebrated<start><id_1> his<id_1><end> sixtieth birthday this week. And thirty one years ago on<start><id_2> Meet The Press<id_2><end> then president Gerald Ford talked about being sixty and serving as president. Coming up right here on<start><id_2> Meet The Press<id_2><end>.'

Our augmentation method handles mentions by adding a pair of tokens:  <start> <id> before a mention should come, where <id> represents the entity to which the mention is related to, and it also corresponds to the order of Entity appearance in the sentence (first entity is <id_1> second is <id_2> and so on..). and finally after the mention ends we add another pair of tokens: <id> <end>. Also in cases where one token is related to multiple mentions: we join those pairs of tokens with a special <or> token. Other than that, it preserves all sentence tokens.
Using this augmentation method we could express the labels without restrictions, and also reverse engineer the clusters form of the sequence , for model evaluation.
  
## Model Configurations
We used the architecture of Transformer Encoder-Decoder for the learning model, as it gives good results for numerous seq2seq tasks. The two configurations we experimented with were T5 and BART.
  
## Results
  
| metric        | T5 no pretrain  | T5 pretrain  | BART no pretrain  | BART pretrain |
| ------------- | :--------------:| :-----------:|:-----------------:|:-------------:|
| B3 F1         | 0.093           | **0.510**    | 0.030             | 0.132         |
| MUC F1        | 0.134           | **0.620**    | 0.032             | 0.151         |
| CEAF F1       | 0.137           | **0.510**    | 0.058             | 0.210         |
| Mention F1    | 0.215           | **0.725**    | 0.104             | 0.306         |
