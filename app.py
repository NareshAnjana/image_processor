import os
import io
import cv2
from flask import Flask, request, render_template_string, send_from_directory
from google.cloud import vision

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Serve files from the uploads directory
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Function to analyze the image and extract text using Google Cloud Vision API
def analyze_image(image_path):
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    return response

# Function to extract text from the Vision API response
def extract_text(response):
    texts = []
    for text in response.text_annotations:
        texts.append(text.description)
    return texts

# Function to segment visual elements using OpenCV
def segment_visual_elements(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        segment = image[y:y+h, x:x+w]
        segments.append(segment)
        print(f"Segment {len(segments)}: x={x}, y={y}, w={w}, h={h}")
    return segments

# Function to generate HTML content with extracted texts and visual elements
def generate_html(texts, visual_elements):
    html_content = '''
    <html>
    <head>
        <title>Extracted Content</title>
        <style>
            .container {
                display: flex;
                flex-direction: row;
                justify-content: space-around;
            }
            .section {
                margin: 20px;
                padding: 20px;
                border: 1px solid #ddd;
                width: 45%;
            }
            .section h2 {
                margin-top: 0;
            }
            .image {
                margin: 10px 0;
                max-width: 100%;
                height: auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="section">
                <h2>Extracted Text</h2>
    '''

    for text in texts:
        # Format each text segment as a paragraph
        html_content += f'<p>{text}</p>'
    
    html_content += '''
            </div>
            <div class="section">
                <h2>Visual Elements</h2>
    '''

    if not visual_elements:
        html_content += '<p>No visual elements detected.</p>'
    else:
        for idx, element in enumerate(visual_elements):
            img_filename = f'segment_{idx}.png'
            img_path = os.path.join(UPLOAD_FOLDER, img_filename)
            cv2.imwrite(img_path, element)
            # Embed each visual element as an image
            html_content += f'<img class="image" src="/uploads/{img_filename}" alt="Visual Element {idx}">'

    html_content += '''
            </div>
        </div>
    </body>
    </html>
    '''
    return html_content

# Flask route for uploading and processing images
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['file']
        if image.filename == '':
            return 'No selected file'
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)
        
        if not os.path.exists(image_path):
            return f"File {image_path} was not saved properly"

        try:
            response = analyze_image(image_path)
            texts = extract_text(response)
            visual_elements = segment_visual_elements(image_path)
            if not visual_elements:
                print("No visual elements detected")
            else:
                print(f"{len(visual_elements)} visual elements detected")
            html_content = generate_html(texts, visual_elements)
            return render_template_string(html_content)
        except Exception as e:
            return f"An error occurred: {e}"
    return '''
    <!doctype html>
    <title>Upload an Image</title>
    <h1>Upload an Image</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
