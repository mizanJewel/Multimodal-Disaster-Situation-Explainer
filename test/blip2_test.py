# # pip install accelerate bitsandbytes
# import torch
# import requests
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="cuda")

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# # question = "how many dogs are in the picture?"
# # inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

# # out = model.generate(**inputs)
# # print(processor.decode(out[0], skip_special_tokens=True).strip())

# # question = "how many animals in the image?"
# # inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

# # out = model.generate(**inputs)
# # print(processor.decode(out[0], skip_special_tokens=True).strip())

# inputs = processor(images=raw_image, return_tensors="pt").to("cuda", torch.float16)

# generated_ids = model.generate(**inputs)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
# print(generated_text)

import os
import csv
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load BLIP-2 Processor and Model (8-bit for lower memory usage)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    load_in_8bit=True, 
    device_map="cuda"
)

def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()
        return caption
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def caption_images_in_folder(input_folder, output_csv):
    # Open CSV for writing
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Name', 'Caption'])

        # Iterate through all files in the folder
        for filename in os.listdir(input_folder):
            image_path = os.path.join(input_folder, filename)
            if os.path.isfile(image_path):
                caption = generate_caption(image_path)
                if caption:
                    writer.writerow([filename, caption])
                    print(f"Processed {filename}: {caption}")
                else:
                    print(f"Skipped {filename} (could not generate caption)")

if __name__ == "__main__":
    input_folder = '/home/exouser/Downloads/demo_images_serial'
    output_csv = '/home/exouser/Downloads/blip2_captions.csv'
    caption_images_in_folder(input_folder, output_csv)
