from flask import Flask, render_template, request, jsonify, send_file
import requests
import os
from io import BytesIO
import base64
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Hugging Face API - using free tier (no key needed for basic use)
HF_API_URL = "https://api-inference.huggingface.co/models/ImagenHub/kohyaa2b-inpainting"
HF_API_KEY = os.getenv('HF_API_KEY', '')  # Optional: set for higher rate limits

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inpaint', methods=['POST'])
def inpaint():
    try:
        data = request.get_json()
        
        # Decode base64 images
        image_data = data['image'].split(',')[1]
        mask_data = data['mask'].split(',')[1]
        
        # Convert base64 to bytes
        image_bytes = base64.b64decode(image_data)
        mask_bytes = base64.b64decode(mask_data)
        
        # Open images
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        mask = Image.open(BytesIO(mask_bytes)).convert('L')
        
        # Resize if too large (to avoid timeouts)
        max_size = 768
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        
        # Prepare for API
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        mask_byte_arr = BytesIO()
        mask.save(mask_byte_arr, format='PNG')
        mask_byte_arr.seek(0)
        
        # Call Hugging Face API
        headers = {}
        if HF_API_KEY:
            headers['Authorization'] = f'Bearer {HF_API_KEY}'
        
        files = {
            'image': ('image.png', img_byte_arr, 'image/png'),
            'mask': ('mask.png', mask_byte_arr, 'image/png')
        }
        
        response = requests.post(HF_API_URL, files=files, headers=headers, timeout=60)
        
        if response.status_code == 200:
            result_image = Image.open(BytesIO(response.content))
            
            # Convert to base64 for return
            buffered = BytesIO()
            result_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'result': f'data:image/png;base64,{img_str}'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'API Error: {response.status_code} - {response.text}'
            }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
