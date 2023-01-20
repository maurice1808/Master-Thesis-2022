#imports
import torch
from PIL import Image
import json
import glob
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#caption generator
max_length = 20
num_beams = 20
gen_kwargs = {"max_length": max_length, "num_beams": num_beams,
              "num_beam_groups": 10, "num_return_sequences": 10, "diversity_penalty":1.0}
def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
        
    return preds

# generate captions
i = 0
for filename in glob.glob('datasets/flickr30k/test/*.jpg'):
    captions = predict_step([filename])
        
    document = {}
    document['id'] = filename[24:-4]
    doc = ""
    for caption in captions:
        doc = doc+caption+'. '
    document['contents'] = doc
    json_string = json.dumps(document)

    with open(r'DocJSONs/flickr30k_b20_g10_d1/doc' + str(i) + '.json', 'w') as outfile:
        outfile.write(json_string)   
    
    i+=1
    if (i%50) == 0:
        print(i)
        print("captions generated:", i, "/1000" , end='\r')
