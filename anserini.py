from pyserini.search.lucene import LuceneSearcher
import json
import torch

searcher = LuceneSearcher('Generatedcaptions/newindex/flickr30k1d20b10g')
# flickr30k or coco
captions = 'flickr30k'

if captions == 'coco':
    #open the json file for mscoco annotations
    f = open('datasets/MSCOCO/annotations/captions_val2017.json')
    data = json.load(f)

    captions = [x.get('caption') for x in data['annotations']]
    caption_ids = [str(x.get('image_id')) for x in data['annotations']]

    ids = []
    for caption_id in caption_ids:
        padding = 12 - len(caption_id)
        new_id = padding * '0' + caption_id
        ids.append(new_id)
        
    captions = list(zip(ids, captions))
elif captions == 'flickr30k':
    #open the json file for flickr30k annotations
    f = open('datasets/flickr30k/dataset_flickr30k.json')
    data = json.load(f)

    caps = [x['sentences'] for x in data['images'] if x.get('split') == 'test']

    image_ids = []
    for x in data['images']:
        if x.get('split') == 'test':
            for i in range(5):
                image_ids.append(x['filename'][:-4])
    captions = []
    for x in caps:
        for y in x:
            captions.append(y)
        
    captions = [x.get('raw') for x in captions]
    captions = list(zip(image_ids, captions))

recall_1 = []
recall_5 = []
recall_10 = []
for caption_id, caption in captions:
    hits = searcher.search(caption)
    docs = []
    for i in range(len(hits)):
        docs.append(hits[i].docid)
    
    if docs:
        recall_1.append(caption_id in docs[0])
        recall_5.append(caption_id in docs[0:5])
        recall_10.append(caption_id in docs[0:10])

recall_1 = torch.Tensor(recall_1)
recall_5 = torch.Tensor(recall_5)
recall_10 = torch.Tensor(recall_10)

print(recall_1.mean())
print(recall_5.mean())
print(recall_10.mean())