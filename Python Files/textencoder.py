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

image_encoder_file = sys.argv[5]

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
class MLM(nn.Module):
    def __init__(self, dense_size, vocab_size):
        super().__init__()
        self.linear = nn.Linear(dense_size, vocab_size)
        
    def forward(self, dense_vec):
        term_importances = torch.log1p(torch.relu(self.linear(dense_vec)))
        return term_importances
    
def test_sparse_performance(text_model, image_model, test_ims, test_cap):
    with torch.no_grad():
        x, _ = test_ims.shape
        encoded_images = torch.empty(int(x/5), 49408)
        for i in range(0, len(test_ims), 5):
            encoded_images[int(i/5)] = image_model(test_ims[i])


        encoded_captions = torch.empty(int(x), 49408)
        for i in range(len(encoded_captions)):
            encoded_captions[i] = text_model(test_cap[i])   

        encoded_images = (encoded_images / encoded_images.norm(dim=-1, keepdim=True)).to(device)
        encoded_captions = encoded_captions / encoded_captions.norm(dim=-1, keepdim=True)

        recall_1 = []
        recall_5 = []
        recall_10 = []
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

# these captions are too long and need adjusting
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

# encode images
L = len(files)

if clip_model == "32base":
    batch_size = 128
else:
    batch_size = 32

with torch.no_grad():
    encoded_ims = torch.Tensor()
    for i in range(0, L, batch_size):
        print(i,"/",L, end='\r')
        images = torch.Tensor().to(device)
        for x in range(batch_size):
            if (i + x) < L:
                image = preprocess(Image.open(files[i+x]), return_tensors='pt')['pixel_values'].to(device)
                images = torch.cat((images, image), 0)
                
        ims = model.vision_model(images).pooler_output
        ims = model.visual_projection(ims)
        encoded_ims = torch.cat((encoded_ims, ims.to("cpu")), 0)

p = 0
encoded_images = torch.Tensor().to(device)
for image in encoded_ims:
    if (p%100) == 0:
        print(p,"/",L, end='\r')
        
    encoded_images = torch.cat((encoded_images, image.to(device).repeat(5,1)), 0)
    p += 1
    
print("")
print("Loading images done")

# encode captions
L = len(captions)

if clip_model == "32base":
    batch_size = 256
else:
    batch_size = 32

with torch.no_grad():
    text_features = torch.Tensor().to(device)
    for i in range(0, L, batch_size):
        print(i,"/",L, end='\r')
        text = tokenizer(captions[i:i+batch_size], padding=True, return_tensors='pt').to(device)
        
        text_output = model.text_model(**text).pooler_output
        text_output = model.text_projection(text_output)      
        text_features = torch.cat((text_features, text_output), 0)
        
print("")
print("Loading captions done")

# encode test images
L = len(test_files)

if clip_model == "32base":
    batch_size = 128
else:
    batch_size = 32

with torch.no_grad():
    test_ims = torch.Tensor().to(device)
    for i in range(0, L, batch_size):
        print(i,"/",L, end='\r')
        images = torch.Tensor().to(device)
        for x in range(batch_size):
            if (i + x) < L:
                image = preprocess(Image.open(test_files[i+x]), return_tensors='pt')['pixel_values'].to(device)
                images = torch.cat((images, image), 0)
                
        ims = model.vision_model(images).pooler_output
        ims = model.visual_projection(ims)
        test_ims = torch.cat((test_ims, ims), 0)

test_images = torch.Tensor().to(device)
for image in test_ims:
    test_images = images = torch.cat((test_images, image.repeat(5,1)), 0)
    
print("")
print("Encoding test images done")

# encode test captions
L = len(test_captions)

if clip_model == "32base":
    batch_size = 256
else:
    batch_size = 32

with torch.no_grad():
    test_features = torch.Tensor().to(device)
    for i in range(0, L, batch_size):
        print(i,"/",L, end='\r')
        text = tokenizer(test_captions[i:i+batch_size], padding=True, return_tensors='pt').to(device)
        
        text_output = model.text_model(**text).pooler_output
        text_output = model.text_projection(text_output)      
        test_features = torch.cat((test_features, text_output), 0)
        
print("")
print("Encoding test captions done")

batchsize = batch_s

dataset = TrainBatches(encoded_images, text_features)
train_dataloader = DataLoader(dataset, batch_size = batchsize, shuffle=True)

#load the sparse encoders
dense_text_size = model.text_projection.weight.shape[0]
dense_image_size = model.visual_projection.weight.shape[0]
vocab_size = model.text_model.config.vocab_size

text_encoder = MLM(dense_text_size, vocab_size).to(device)
image_encoder = MLM(dense_image_size, vocab_size).to(device)

#load image encoder checkpoint
im_checkpoint = torch.load(image_encoder_file, map_location='cpu')
image_encoder.load_state_dict(im_checkpoint['model_state_dict'])

#test performance before training
rec1,rec5,rec10 = test_sparse_performance(text_encoder, image_encoder, test_images, test_features)

#training loop
loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(text_encoder.parameters(), lr=1e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10], gamma=0.1)

vocab_size = model.text_model.config.vocab_size
epochs = 15
totalbatches = int(len(dataset) / batchsize)
logit_scale = model.logit_scale.exp().item()
losses=[]
test_loss=[[rec1],[rec5],[rec10]]
for epoch in range(0, epochs):
    i = 0
    batch_loss = 0
    for batch in train_dataloader:
        if i % 25 == 0:
            print("epoch:", epoch, "batch:", i, "/", totalbatches, end='\r')
       
        optimizer.zero_grad()
        dense_images, dense_caps = batch
        
        # sparse encoding
        with torch.no_grad():
            sparse_ims = image_encoder(dense_images)
            
        sparse_caps = text_encoder(dense_caps)

        # determine logits
        sparse_ims = sparse_ims / sparse_ims.norm(dim=-1, keepdim=True)
        sparse_caps = sparse_caps / sparse_caps.norm(dim=-1, keepdim=True)
        logits_per_image = logit_scale * sparse_ims @ sparse_caps.t()
        logits_per_text = logits_per_image.t()
        
        # compute losses
        ground_truth = torch.arange(len(dense_images),dtype=torch.long,device=device)
        
        loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            
        batch_loss += loss.item()
        loss.backward()
        
        optimizer.step()
        i+=1
        
    scheduler.step()
    losses.append(batch_loss)
    print("loss:", batch_loss)
    
    recall1,recall5,recall10 = test_sparse_performance(text_encoder, image_encoder, test_images, test_features)
    test_loss[0].append(recall1)
    test_loss[1].append(recall5)
    test_loss[2].append(recall10)
    print("R@1:", recall1)
    print("")
  
print("")
print("done")
torch.save({
    'epoch':epoch,
    'model_state_dict': text_encoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, "text_encoder" + ".pt")

plt.plot(losses)
plt.title('CLIP loss')
plt.show()