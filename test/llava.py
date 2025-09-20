# import requests
# from PIL import Image

# import torch
# from transformers import AutoProcessor, LlavaForConditionalGeneration

# model_id = "llava-hf/llava-1.5-7b-hf"
# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     low_cpu_mem_usage=True, 
# ).to(0)

# processor = AutoProcessor.from_pretrained(model_id)

# # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# # Each value in "content" has to be a list of dicts with types ("text", "image") 
# conversation = [
#     {

#       "role": "user",
#       "content": [
#           {"type": "text", "text": "Create a caption for this image."},
#           {"type": "image"},
#         ],
#     },
# ]
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
# raw_image = Image.open(requests.get(image_file, stream=True).raw)
# inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

# output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
# print(processor.decode(output[0][2:], skip_special_tokens=True))

import os
import csv
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Initialize model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

def clean_caption(decoded_text):
    # Remove everything before "ASSISTANT:" if present
    if "ASSISTANT:" in decoded_text:
        return decoded_text.split("ASSISTANT:")[-1].strip()
    # Otherwise, remove the first role colon if any (fallback)
    if ':' in decoded_text:
        return decoded_text.split(':', 1)[-1].strip()
    return decoded_text.strip()


def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Create a caption for this image."},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        decoded = processor.decode(output[0][2:], skip_special_tokens=True)
        caption = clean_caption(decoded)

        # caption = processor.decode(output[0][2:], skip_special_tokens=True).strip()
        return caption
    except Exception as e:
        print(f"Error with {image_path}: {e}")
        return None

def caption_images_in_folder(input_folder, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Name', 'Caption'])

        for filename in os.listdir(input_folder):
            image_path = os.path.join(input_folder, filename)
            if os.path.isfile(image_path):
                caption = generate_caption(image_path)
                if caption:
                    writer.writerow([filename, caption])
                    print(f"{filename}: {caption}")
                else:
                    print(f"Skipped {filename}")

if __name__ == "__main__":
    input_folder = '/home/exouser/Downloads/demo_images_serial'
    output_csv = '/home/exouser/Downloads/llava_captions.csv'
    caption_images_in_folder(input_folder, output_csv)
