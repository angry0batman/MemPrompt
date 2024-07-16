import os
import random
from flask import Flask, render_template, request
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

app = Flask(__name__)

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2")

# Load the images from the dataset
image_folder = 'images'

def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="tf")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def create_meme(image_path, text):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("impact.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text size
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    width, height = img.size
    x = (width - text_width) / 2
    y = height - text_height - 10

    # Draw main text
    draw.text((x, y), text, font=font, fill="white")

    # Ensure the directory exists
    meme_dir = os.path.join('static', 'images')
    if not os.path.exists(meme_dir):
        os.makedirs(meme_dir)

    meme_path = os.path.join(meme_dir, 'meme.png')
    img.save(meme_path)
    return meme_path

@app.route('/', methods=['GET', 'POST'])
def index():
    meme_url = None
    if request.method == 'POST':
        prompt = request.form['prompt']
        image_file = request.files.get('image_file')
        if image_file:
            image_path = os.path.join('static/images', image_file.filename)
            image_file.save(image_path)
        else:
            image_path = os.path.join(image_folder, random.choice(os.listdir(image_folder)))
        
        meme_text = generate_text(prompt)
        meme_url = create_meme(image_path, meme_text)
    
    return render_template('index.html', meme_url=meme_url)

if __name__ == '__main__':
    app.run(debug=True)
