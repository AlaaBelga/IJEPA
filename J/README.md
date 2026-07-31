# Image Inpainting Tool

A simple web application for removing unwanted objects from images using AI-powered inpainting.

## Features

- 🎨 Draw masks directly on images to mark areas for removal
- 🤖 Automatic inpainting using LaMa (Local-model Autofill for MAsk areas)
- 🚀 Fast inference - runs locally on your machine
- 📱 Responsive design - works on desktop and mobile
- 🎯 Real-time brush size control

## Requirements

- Python 3.8+
- pip

## Installation

1. **Clone or download this project**

2. **Create a virtual environment** (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Note: The first run will download the LaMa model (~1GB). This is a one-time download.

## Usage

1. **Start the Flask server**:
```bash
python app.py
```

2. **Open your browser** and go to:
```
http://localhost:5000
```

3. **Upload an image**:
   - Click "Choose File" and select an image
   - Click "Load Image"

4. **Draw the mask**:
   - Use your mouse to draw white areas on the mask canvas
   - These areas will be inpainted (removed and filled)
   - Adjust brush size with the slider

5. **Inpaint**:
   - Click "Inpaint" button
   - Wait for processing (typically 5-30 seconds depending on image size and quality)
   - View the result in the right panel

## How It Works

The application uses **LaMa (Large Mask Inpainting)**, a state-of-the-art inpainting model that:
- Understands image context to fill masked areas naturally
- Works well with large masked regions
- Produces photorealistic results
- Runs entirely on your local machine

## Keyboard Shortcuts

- Hold shift + scroll to change brush size (coming soon)
- Right-click to undo (coming soon)

## Troubleshooting

**Issue: "No module named 'simple_lama_inpainting'"**
- Make sure you've installed all requirements: `pip install -r requirements.txt`
- If still not working, try: `pip install simple-lama-inpainting`

**Issue: Out of memory**
- The model requires ~2GB of RAM
- Close other applications
- If still issues, try reducing image size before uploading

**Issue: Very slow processing**
- This is normal for the first run (model is downloading)
- Subsequent runs are much faster
- GPU acceleration is not yet enabled in this version

## Future Enhancements

- [ ] GPU acceleration support
- [ ] Batch processing
- [ ] Undo/redo functionality
- [ ] Different model options (Stable Diffusion, etc.)
- [ ] Before/after comparison slider
- [ ] Image download option

## License

MIT License - Feel free to use and modify!

## References

- [LaMa Paper](https://github.com/advimman/lama)
- [Flask Documentation](https://flask.palletsprojects.com/)
