#imports
import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from PIL import Image
import sys

#set command line arguments
clip_model = sys.argv[1]
batch_s = int(sys.argv[2])
data_set = sys.argv[3]
small_test = False
if len(sys.argv) > 4:
    small_test = bool(sys.argv[4])

#load the right CLIP model
if clip_model == "32base":
    clipname = "openai/clip-vit-base-patch32"
elif clip_model == "14large":
    clipname = "openai/clip-vit-large-patch14"
elif clip_model == "14large336":
    clipname = "openai/clip-vit-large-patch14-336"

from transformers import CLIPProcessor, CLIPModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(clipname).to(device)
processor = CLIPProcessor.from_pretrained(clipname)

preprocess = processor.feature_extractor
tokenizer = processor.tokenizer

# classes and functions
class MLP(nn.Module):
    def __init__(self, dense_size, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(dense_size, 1)
        self.linear.apply(self._init_weights)
        
    def _init_weights(self, module):
        torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity="relu")
    
    def forward(self, last_hidden_states, input_ids, attention_mask):
        batch_size = last_hidden_states.shape[0]
        tok_weights = torch.log1p(torch.relu(self.linear(last_hidden_states))).squeeze(-1)*attention_mask
        lex_weights = torch.zeros(batch_size, self.vocab_size).to(device)
        columns = torch.arange(batch_size).repeat((77,1)).T
        lex_weights[columns, input_ids.type(torch.int64)] = tok_weights
        return lex_weights

class MLM(nn.Module):
    def __init__(self, dense_size, vocab_size):
        super().__init__()
        self.linear = nn.Linear(dense_size, vocab_size)
        
    def forward(self, dense_vec):
        term_importances = torch.log1p(torch.relu(self.linear(dense_vec)))
        return term_importances

def test_sparse_performance(MLP, MLM, CLIP, test_loader):
    with torch.no_grad():
        encoded_images = torch.Tensor()
        encoded_captions = torch.Tensor()
        for batch in test_loader:
            ims, tokenized_caps = batch
            # encode images
            ims = model.vision_model(ims.to(device)).pooler_output
            ims = model.visual_projection(ims)
            encoded_ims = image_encoder(ims)
            # encode captions
            batch_length = tokenized_caps['input_ids'].shape[0]
            tokenized_caps['input_ids'] = tokenized_caps['input_ids'].reshape(batch_length*5, 77)
            tokenized_caps['attention_mask'] = tokenized_caps['attention_mask'].reshape(batch_length*5, 77)
            last_hidden_states = model.text_model(**tokenized_caps).last_hidden_state
            last_hidden_states = model.text_projection(last_hidden_states)
            input_ids = tokenized_caps['input_ids']
            attention_masks = tokenized_caps['attention_mask']
            encoded_caps = text_encoder(last_hidden_states, input_ids, attention_masks)
            # normalize
            encoded_ims = encoded_ims / encoded_ims.norm(dim=-1, keepdim=True)
            encoded_caps = encoded_caps / (encoded_caps + 1e-20).norm(dim=-1, keepdim=True)
            
            encoded_images = torch.cat((encoded_images, encoded_ims.to("cpu")), 0)
            encoded_captions = torch.cat((encoded_captions, encoded_caps.to("cpu")), 0)
        
        recall_1 = []
        recall_5 = []
        recall_10 = []
        encoded_images = encoded_images.to(device)
        i = 0
        image_id = 0
        for text_feature in encoded_captions:
            similarity = (100.0 * text_feature.to(device) @ encoded_images.T).softmax(dim=-1)

            values, indices = similarity.topk(1)
            recall_1.append(image_id in indices)

            values, indices = similarity.topk(5)
            recall_5.append(image_id in indices)

            values, indices = similarity.topk(10)
            recall_10.append(image_id in indices)

            i += 1
            if i == 5:
                i = 0
                image_id += 1

        recall_1 = torch.Tensor(recall_1)
        recall_5 = torch.Tensor(recall_5)
        recall_10 = torch.Tensor(recall_10)
        return recall_1.mean(), recall_5.mean(), recall_10.mean()

class TrainBatches():
    def __init__(self, image_vectors, captions):

        self.images = image_vectors
        self.captions  = captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = self.images[idx]
        caption = self.captions[idx]
        return image,caption

#open the json file for annotations
if data_set == "flickr30k":
    f = open('datasets/flickr30k/dataset_flickr30k.json')
    data = json.load(f)
    train_folder = 'datasets/flickr30k/train/'
    test_folder = 'datasets/flickr30k/test/'

elif data_set == "mscoco":
    f = open('datasets/MSCOCO2014/dataset_coco.json')
    data = json.load(f)
    train_folder = 'datasets/MSCOCO2014/train/'
    test_folder = 'datasets/MSCOCO2014/test/'

# load all captions
caps = [x['sentences'] for x in data['images'] if x.get('split') == 'train' and len(x.get('sentids')) == 5]
test_caps = [x['sentences'] for x in data['images'] if x.get('split') == 'test' and len(x.get('sentids')) == 5]

# load all image filenames
files = []
test_files = []
for x in data['images']:
    if x.get('split') == 'train':
        files.append(train_folder + x['filename'])
    
    elif x.get('split') == 'test':
        test_files.append(test_folder + x['filename'])

# every image has 5-6 captions
captions = []
for x in caps:
    for y in x:
        captions.append(y)
        
test_captions = []
for x in test_caps:
    for y in x:
        test_captions.append(y)
    
captions = [x.get('raw') for x in captions]
test_captions = [x.get('raw') for x in test_captions]

# these captions are too long for the model and need adjusting
if data_set == "flickr30k":
    captions[13035] = 'Four young adults sit outside on a wooden deck near a building around a small round table, while another person stands on the edge of the deck, leaning on the wooden railing, with the sun shining on one of them, one holding a cellphone out in front of himself and another holding a green and red soda can.'
    captions[14580] = 'A man wearing a helmet, red pants with white and a white and red shirt is on a small bicycle using only his hands, while another man wearing a light blue shirt with dark blue trim and black pants with red stripes is standing nearby, gesturing toward the first man and holding a small figurine.'
    captions[120165] = 'In this photo there is a man in a dirty white shirt and a black hat with yellow stained teeth, he looks happy and it appears that he is also repairing something.'
    test_captions[3905] = 'Two boys are looking upwards with their arms streteched to the sky, the boy on the left is wearing a blue vest jacket with a gray shirt, black jogging pants and a hat, and the boy on the right is wearing a silver vest jacket, with blue long-sleeved undershirt, gray pants, black tennis shoes and has black short hair and glasses.'

if small_test:
    files = files[0:100]
    captions = captions[0:500]

    test_files = test_files[0:10]
    test_captions = test_captions[0:50]

# load and preprocess the images
images = []
i = 0
with torch.no_grad():
    for filename in files:
        im = preprocess(Image.open(filename), return_tensors='pt')['pixel_values'].squeeze()
        for _ in range(5):
            images.append(im)
        i+=1
        if (i%100) == 0:
            print(i,"/",len(files), end='\r')
        
print("")
print("Loading images done")

# load and tokenize the captions
L = len(captions)
with torch.no_grad():
    tokenized_captions = []
    for i in range(0, L):
        if (i%1000) == 0:
            print(i,"/",L, end='\r')
        text = tokenizer(captions[i], padding='max_length', max_length=77, return_tensors='pt').to(device)
        tokenized_captions.append(text)
        
print("")
print("Loading captions done")

# load and preprocess all test images
test_images = []
i = 0
with torch.no_grad():
    for filename in test_files:
        im = preprocess(Image.open(filename), return_tensors='pt')['pixel_values'].squeeze()
        test_images.append(im)
        i+=1
        if (i%100) == 0:
            print(i,"/",len(test_files), end='\r')
        
print("")
print("loading test images done")

# load and tokenize the tests captions
L = len(test_captions)
with torch.no_grad():
    test_features = []
    for i in range(0, L, 5):
        if (i%1000) == 0:
            print(i,"/",L, end='\r')
        text = tokenizer(test_captions[i:i+5], padding='max_length', max_length=77, return_tensors='pt').to(device)
        test_features.append(text)
        
print("")
print("loading test captions done")

#test dataloader
test_batchsize = 64
test_dataset = TrainBatches(test_images, test_features)
test_dataloader = DataLoader(test_dataset, batch_size = test_batchsize, shuffle=True)

#train dataloader
batchsize = batch_s
dataset = TrainBatches(images, tokenized_captions)
train_dataloader = DataLoader(dataset, batch_size = batchsize, shuffle=True)

#load the sparse encoders
dense_text_size = model.text_projection.weight.shape[0]
dense_image_size = model.visual_projection.weight.shape[0]
vocab_size = model.text_model.config.vocab_size

text_encoder = MLP(dense_text_size, vocab_size).to(device)
image_encoder = MLM(dense_image_size, vocab_size).to(device)

#test performance before training
rec1,rec5,rec10 = test_sparse_performance(text_encoder, image_encoder, model, test_dataloader)

#training loop
loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()

params = list(text_encoder.parameters()) + list(image_encoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,8], gamma=0.1)

optimizer_clip = torch.optim.Adam(model.parameters(), lr=1e-7,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)

vocab_size = model.text_model.config.vocab_size
epochs = 12
totalbatches = int(len(dataset) / batchsize)
logit_scale = model.logit_scale.exp().item()
losses=[]
test_loss=[[rec1],[rec5],[rec10]]
for epoch in range(0, epochs):
    i = 0
    batch_loss = 0
    for batch in train_dataloader:
        if (i%10) == 0:
            print("epoch:", epoch, "batch:", i, "/", totalbatches, end='\r')
       
        optimizer.zero_grad()
        optimizer_clip.zero_grad()
        ims, tokenized_caps = batch
        # encode images
        ims = model.vision_model(ims.to(device)).pooler_output
        ims = model.visual_projection(ims)
        sparse_ims = image_encoder(ims)

        # encode captions
        tokenized_caps['input_ids'] = tokenized_caps['input_ids'].squeeze()
        tokenized_caps['attention_mask'] = tokenized_caps['attention_mask'].squeeze()

        last_hidden_states = model.text_model(**tokenized_caps).last_hidden_state
        last_hidden_states = model.text_projection(last_hidden_states)
        input_ids = tokenized_caps['input_ids']
        attention_masks = tokenized_caps['attention_mask']

        sparse_caps = text_encoder(last_hidden_states, input_ids, attention_masks)
        
        # determine logits
        sparse_ims = sparse_ims / sparse_ims.norm(dim=-1, keepdim=True)
        sparse_caps = sparse_caps / (sparse_caps + 1e-20).norm(dim=-1, keepdim=True)
        logits_per_image = logit_scale * sparse_ims @ sparse_caps.t()
        logits_per_text = logits_per_image.t()
        
        # compute losses
        ground_truth = torch.arange(len(ims),dtype=torch.long, device=device)
        loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        
        batch_loss += loss.item()
        loss.backward()
        
        optimizer.step()
        optimizer_clip.step()
        i+=1
        
    scheduler.step()
    losses.append(batch_loss)
    
    recall1,recall5,recall10 = test_sparse_performance(text_encoder, image_encoder, model, test_dataloader)
    test_loss[0].append(recall1)
    test_loss[1].append(recall5)
    test_loss[2].append(recall10)

print("")
print("Done training")
#save final model after training
torch.save({
    'epoch':epoch,
    'model_state_dict': text_encoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, "text_encoder" + ".pt")

torch.save({
    'epoch':epoch,
    'model_state_dict': image_encoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, "image_encoder" + ".pt")

torch.save({
    'epoch':epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer_clip.state_dict(),
    'loss': loss,
    }, "model" + ".pt")

print(losses)
print(test_loss)