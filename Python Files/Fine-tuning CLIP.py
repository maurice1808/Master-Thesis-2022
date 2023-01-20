#imports
import torch
from torch.utils.data import DataLoader
import clip
from PIL import Image
import json
import glob

#open the json file for annotations
f = open('datasets/flickr30k/dataset_flickr30k.json')
data = json.load(f)


#loading all the captions
caps = [x['sentences'] for x in data['images'] if x.get('split') == 'train']
files = []
for x in data['images']:
    if x.get('split') == 'train':
        files.append('datasets/flickr30k/train/' + x['filename'])
            
captions = []
for x in caps:
    for y in x:
        captions.append(y)
    
captions = [x.get('raw') for x in captions]
# these captions are too long and need adjusting
captions[13035] = 'Four young adults sit outside on a wooden deck near a building around a small round table, while another person stands on the edge of the deck, leaning on the wooden railing, with the sun shining on one of them, one holding a cellphone out in front of himself and another holding a green and red soda can.'
captions[14580] = 'A man wearing a helmet, red pants with white and a white and red shirt is on a small bicycle using only his hands, while another man wearing a light blue shirt with dark blue trim and black pants with red stripes is standing nearby, gesturing toward the first man and holding a small figurine.'
captions[120165] = 'In this photo there is a man in a dirty white shirt and a black hat with yellow stained teeth, he looks happy and it appears that he is also repairing something.'

#loading model and images
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32",device=device,jit=False)

images = []
i = 0
for filename in files:
    im=preprocess(Image.open(filename))
    for j in range(5):
        images.append(im)
    i+=1
    if (i%1000) == 0:
        print(i,"/",len(files), end='\r')
        
print("")

#class for data loader
class image_caption_dataset():
    def __init__(self, image_list, caption_list):

        self.images = image_list
        self.captions  = clip.tokenize(caption_list)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = self.images[idx]
        caption = self.captions[idx]
        return image,caption

batchsize = 64
dataset = image_caption_dataset(images, captions)
train_dataloader = DataLoader(dataset, batch_size = batchsize, shuffle=True)

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
epochs = 30
totalbatches = int(len(dataset) / batchsize)

#training loop
for epoch in range(epochs):
    i = 0
    for batch in train_dataloader:
        print("epoch:", epoch, "batch:", i, "/", totalbatches, end='\r')
       
        optimizer.zero_grad()

        ims, caps = batch 

        ims= ims.to(device)
        caps = caps.to(device)

        logits_per_image, logits_per_text = model(ims, caps)

        ground_truth = torch.arange(len(ims),dtype=torch.long,device=device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
            i+=1
        else : 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
            i+=1
    
    torch.save({
        'epoch':epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, f"Models/" + str(epoch) + ".pt")
    
print("")
print("done")
