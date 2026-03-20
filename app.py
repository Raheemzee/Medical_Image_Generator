from flask import Flask, render_template, request, send_file
from medigan import Generators
import os
import cv2

app = Flask(__name__)

# Initialize Medigan
generators = Generators()

# Map human-readable names to Medigan model IDs
MODEL_MAP = {
    "Mammogram Calcifications": "00001_DCGAN_MMG_CALC_ROI",
    "Mammogram Masses": "00002_DCGAN_MMG_MASS_ROI"
}

# Folder for generated images
GENERATED_DIR = "generated"
os.makedirs(GENERATED_DIR, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    image_paths = []
    selected_model_name = None

    if request.method == "POST":
        # Get user input
        selected_model_name = request.form.get("model_name")
        num_samples = request.form.get("num_samples", "1")
        try:
            num_samples = int(num_samples)
        except ValueError:
            num_samples = 1

        # Clamp to safe range
        num_samples = max(1, min(50, num_samples))

        # Get model ID from map
        model_id = MODEL_MAP.get(selected_model_name)
        if model_id:
            # Directly generate images with model_id
            images = generators.generate(model_id=model_id, num_samples=num_samples, install_dependencies=False)

            # Save images
            for i, img in enumerate(images):
                # Sanitize model name for filenames
                safe_name = selected_model_name.replace(" ", "_").lower()
                filename = f"{safe_name}_{i}.png"
                filepath = os.path.join(GENERATED_DIR, filename)
                cv2.imwrite(filepath, img)
                image_paths.append(filepath)

    return render_template(
        "index.html",
        image_paths=image_paths,
        models=list(MODEL_MAP.keys()),
        selected_model=selected_model_name
    )

@app.route("/view/<filename>")
def view_image(filename):
    filepath = os.path.join(GENERATED_DIR, filename)
    return send_file(filepath, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
