# Master-Thesis-2022

Code files explanation

---python files---  
`Clip evaluation (MS COCO 2014 karpathy) all epochs.py`: evaluates trained models for every epoch on the karpathy MS COCO test set for text&image recall@1,5,10  
Clip evaluation (flickr30k 2014 karpathy) all epochs.py: same but for flickr30k  
fine-tuning CLIP.py: loads in the flickr30k training set and fine-tunes CLIP on it using the CLIP loss  
image caption generator.py: generates captions for images in the flickr30k dataset and saves the generated captions as json files  
anserini.py: loads in an inverted index and uses it for search on either COCO or flickr30k test set and evaluates it  
  
---notebooks---  
Clip evaluation (flickr30k 2014 karpathy) - baseline models.ipynb: evaluates CLIP for all the different available vision models  
Clip evaluation (flickr30k 2014 karpathy).ipynb: evaluates trained models for every epoch on the karpathy flickr test set for text&image recall@1,5,10  
fine-tuning flickr30k.ipynb: fine-tunes CLIP on flickr30k  
MLP+MLM Model (flickr30k).ipynb: trains both the MLP and MLM overheads on flickr30k, also includes evaluation and experiments for the image heatmap&sparsity  
MLP+MLM+CLIP Model (flickr30k).ipynb: trains the MLP and MLM overheads on flickr30k, this one also optimizes CLIP in the training loop  
sparse CLIP.ipynb: original implementation that only has the MLM overhead  
