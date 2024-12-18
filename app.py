from flask import Flask, request, jsonify
from lime import lime_image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# Load your trained model
model = load_model('my_densenet_model.h5')

# Define a prediction function for LIME
def predict_fn(images):
    preds = model.predict(images)
    return preds

# Initialize LIME
explainer = lime_image.LimeImageExplainer()

# Load and preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict if the image is cancerous
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    preds = model.predict(img_array)
    is_cancerous = preds[0][1] > 0.5
    return is_cancerous, preds

# Function to generate textual reasons based on LIME explanation
def get_textual_reasons(explanation, label):
    temp, mask = explanation.get_image_and_mask(
        label=label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    
    feature_importances = explanation.local_exp[label]
    feature_descriptions = []
    
    for i, (feature, weight) in enumerate(feature_importances):
        if weight > 0:
            feature_descriptions.append(f"Feature {i+1}: Contributes positively with a weight of {weight:.4f}. This segment may show characteristics indicative of melanoma.")
        else:
            feature_descriptions.append(f"Feature {i+1}: Contributes negatively with a weight of {weight:.4f}. This segment may not show typical features of melanoma.")
    
    medical_explanation = "Based on the features highlighted by LIME:\n"
    if np.any(mask):
        medical_explanation += "- Melanomas often present as asymmetrical lesions with irregular borders.\n"
        medical_explanation += "- Color variation within the lesion is a common sign.\n"
        medical_explanation += "- The highlighted areas correspond to these typical characteristics.\n"
    else:
        medical_explanation += "No significant melanoma features were highlighted by the model.\n"
        medical_explanation += "The model did not detect clear signs of melanoma based on the image. Further clinical evaluation is recommended.\n"

    return "\n".join(feature_descriptions) + "\n\n" + medical_explanation

# Preprocess image for segmentation
def preprocess_for_segmentation(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return img_array

# Segment the image using SLIC (Simple Linear Iterative Clustering)
def segment_image(image):
    segments = slic(image, n_segments=100, compactness=10, sigma=1)
    return segments

# Overlay boundaries on the original image
def overlay_boundaries(image, segments):
    boundaries = mark_boundaries(image, segments, color=(1, 0, 0), mode='thick')
    return boundaries

# Function to make Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    last_conv_layer_output = model.get_layer(last_conv_layer_name).output
    model_output = model.output
    
    if isinstance(model_output, list):
        model_output = model_output[0]

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer_output, model_output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    class_name = 'Malignant' if tf.argmax(predictions[0]) == 1 else 'Benign'
    confidence_score = np.max(predictions[0]) * 100
    
    return heatmap.numpy(), class_name, confidence_score


# Function to generate and save Grad-CAM image
def save_gradcam(image_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.uint8(superimposed_img)
    cv2.imwrite(cam_path, superimposed_img)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'})

    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)

    is_cancerous, preds = predict_image(image_path)

    if is_cancerous:
        img_array = preprocess_image(image_path)
        explanation = explainer.explain_instance(
            img_array[0],
            predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=1000
        )
        label = 1
        textual_reasons = get_textual_reasons(explanation, label)
        
        # Generate and save LIME explanation image
        temp, mask = explanation.get_image_and_mask(
            label=1,  # Label for cancerous class
            positive_only=False,
            num_features=10,
            hide_rest=False
        )
        
        if not os.path.exists('static/explanations'):
            os.makedirs('static/explanations')
        
        lime_explanation_path = os.path.join('static/explanations', f"{file.filename.split('.')[0]}_lime_explanation.png")
        plt.imshow(mark_boundaries(temp, mask))
        # plt.title("LIME Explanation")
        plt.axis('off')
        plt.savefig(lime_explanation_path)
        plt.close()
    else:
        textual_reasons = "The model did not detect significant signs of cancerous features based on the image."
        lime_explanation_path = None

    image_for_segmentation = preprocess_for_segmentation(image_path)
    segments = segment_image(image_for_segmentation)
    boundaries = overlay_boundaries(image_for_segmentation, segments)
    
    if not os.path.exists('static/plots'):
        os.makedirs('static/plots')
    
    plot_path = os.path.join('static/plots', f"{file.filename.split('.')[0]}_boundaries.png")
    
    plt.figure(figsize=(12, 6))
    plt.imshow(image_for_segmentation)
    plt.axis('off')
    # plt.title('Original Image')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Generate Grad-CAM heatmap and save it
    heatmap, class_name, confidence_score = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block32_concat")
    if not os.path.exists('static/gradcam'):
        os.makedirs('static/gradcam')
    gradcam_path = os.path.join('static/gradcam', f"{file.filename.split('.')[0]}_gradcam.png")
    save_gradcam(image_path, heatmap, cam_path=gradcam_path)

    gradcam_explanation = (f"The model predicts this lesion as {class_name} with a confidence of {confidence_score:.2f}%. ")
    if class_name == 'Malignant':
        gradcam_explanation += ("The red regions represent areas that the model believes to be most indicative of malignancy. "
                                "These areas may correspond to the irregular texture, shape, or color associated with cancerous tissue.")
    else:
        gradcam_explanation += ("The red regions are less pronounced, suggesting that the model does not detect features associated "
                                "with malignancy in this lesion. Smooth texture, uniform color, and regular shape are common signs "
                                "of a benign lesion, but medical examination is recommended.")

    return jsonify({
        'result': 'cancerous' if is_cancerous else 'not cancerous',
        'textual_reasons': textual_reasons,
        'plot_url': plot_path,
        'lime_explanation_url': lime_explanation_path,
        'gradcam_url': gradcam_path,
        'gradcam_explanation': gradcam_explanation
    })


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)