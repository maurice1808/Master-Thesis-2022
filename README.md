# Master-Thesis-2022

Code files explanation

---python files---  
`Clip evaluation (MS COCO 2014 karpathy) all epochs.py`: evaluates trained models for every epoch on the karpathy MS COCO test set for text&image recall@1,5,10  
`Clip evaluation (flickr30k 2014 karpathy) all epochs.py`: same but for flickr30k  
`fine-tuning CLIP.py`: loads in the flickr30k training set and fine-tunes CLIP on it using the CLIP loss  
`image caption generator.py`: generates captions for images in the flickr30k dataset and saves the generated captions as json files  
`anserini.py`: loads in an inverted index and uses it for search on either COCO or flickr30k test set and evaluates it  
  
---notebooks---  
`Clip evaluation (flickr30k 2014 karpathy) - baseline models.ipynb`: evaluates CLIP for all the different available vision models  
`Clip evaluation (flickr30k 2014 karpathy).ipynb`: evaluates trained models for every epoch on the karpathy flickr test set for text&image recall@1,5,10  
`fine-tuning flickr30k.ipynb`: fine-tunes CLIP on flickr30k  
`MLP+MLM Model (flickr30k).ipynb`: trains both the MLP and MLM overheads on flickr30k, also includes evaluation and experiments for the image heatmap&sparsity  
`MLP+MLM+CLIP Model (flickr30k).ipynb`: trains the MLP and MLM overheads on flickr30k, this one also optimizes CLIP in the training loop  
`sparse CLIP.ipynb`: original implementation that only has the MLM overhead  

---instructions---

model type 2, MLP + MLM with clip frozen:<br />
MLM+MLP.py ‘clip model’ ‘batch_size’ ‘dataset’ ‘smalltest’<br />
MLM+MLP.py = file name<br />
‘clip model’ = 32base for the small clip model, 14large for large model, 14large336 for largest model
‘batch_size’ = batch size used in training loop
‘dataset’ = dataset used to train and test on, flickr30k for flickr dataset, mscoco for MS COCO 2014 dataset
‘smalltest’ = False by default, set to True to cut down the training and test sets to very small number of samples to do a test run

Example command line: 
python MLP+MLM.py 32base 64 flickr30k True

model type 3, MLP + MLM with clip unfrozen:
MLP+MLM+CLIP.py is the exact same, example command line:
python MLP+MLM+CLIP.py 32base 64 flickr30k True

model type 4, optimize a text encoder given an already trained image encoder:
textencoder.py has 1 small difference, at the end of the line you have to type the filename of the image encoder that you want to use. For example:
python textencoder.py 32base 64 flickr30k True image_encoder.pt

if you don’t want to use a small test, put:
python textencoder.py 32base 64 flickr30k False image_encoder.pt
