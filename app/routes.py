from flask import current_app as app
from flask import request, jsonify
import os
import base64
from PIL import Image
from io import BytesIO
import shutil

from app.cosin import get_images_from_directory, generate_outfits
from app.helper import load_vgg19_feature_extractor, load_category_features, load_colors, get_images_in_categories, \
    process_images_for_categories_with_dominant_color, extract_features_and_save, save_dominant_colors

cropped_images_folder_path = r"D:\ggp\croppedImages"
cropped_images_formal_folder_path = r"D:\ggp\croppedImagesFormal"
features_folder_path = r"D:\ggp\saved features"
colors_folder_path = r"D:\ggp\savedColors"
saved_folder = "D:\\ggp\\user"


def save_images_from_base64(images_dict, parent_folder):
    if os.path.exists(parent_folder):
        shutil.rmtree(parent_folder)

        # Create parent folder if it doesn't exist
    os.makedirs(parent_folder, exist_ok=True)

    # Create subfolders for each class
    class_folders = {}
    for class_name in images_dict.keys():
        class_folder = os.path.join(parent_folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        class_folders[class_name] = class_folder

    # Decode and save each image
    for class_name, images in images_dict.items():
        class_folder = class_folders[class_name]
        for i, base64_image in enumerate(images):
            try:
                # Decode base64 to image bytes
                image_data = base64.b64decode(base64_image)
                image = Image.open(BytesIO(image_data))
                # Check and correct orientation if EXIF data is available
                exif = image._getexif()
                if exif:
                    orientation = exif.get(0x0112)
                    if orientation in (2, '2'):
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation in (3, '3'):
                        image = image.rotate(180)
                    elif orientation in (4, '4'):
                        image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation in (5, '5'):
                        image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation in (6, '6'):
                        image = image.rotate(-90, expand=True)
                    elif orientation in (7, '7'):
                        image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation in (8, '8'):
                        image = image.rotate(90, expand=True)

                image = image.resize((224, 224))

                # Save image to folder
                image_path = os.path.join(class_folder, f"{class_name}_{i + 1}.png")
                image.save(image_path)
            except Exception as e:
                print(f"Error saving {class_name} image {i + 1}: {e}")

    return parent_folder


def convert_images_to_base64(image_paths):
    base64_images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            base64_images.append(base64_image)
    return base64_images


@app.route('/api/data', methods=['GET'])
def get_data():
    formal = request.args.get('formal')
    vgg19_feature_extractor = load_vgg19_feature_extractor()

    if features_folder_path and colors_folder_path and saved_folder:
        category_features = load_category_features(features_folder_path)
        color_features = load_colors(colors_folder_path)
        if formal:
            images_list = get_images_from_directory(cropped_images_formal_folder_path)
        else:
            images_list = get_images_from_directory(cropped_images_folder_path)
        user_files = get_images_in_categories(saved_folder)
        outfits = generate_outfits(images_list, category_features, color_features, user_files, vgg19_feature_extractor)
        outfit_base64 = [convert_images_to_base64(outfits46) for outfits46 in outfits]
        return jsonify(outfit_base64)

    return jsonify({'error': 'Paths are not properly set'}), 500


@app.route('/api/pre/data', methods=['POST'])
def pre_data():
    data = request.get_json()
    pants_base64 = data.get('pants')
    shoe_base64 = data.get('shoe')
    shirt_base64 = data.get('shirt')
    jacket_base64 = data.get('jacket')
    shorts_base64 = data.get('shorts')

    images_dict = {
        "pants": pants_base64,
        "shoe": shoe_base64,
        "shirt": shirt_base64,
        "jacket": jacket_base64,
        "shorts": shorts_base64
    }

    parent_folder = "D:\\ggp\\user"
    saved_folder = save_images_from_base64(images_dict, parent_folder)
    if saved_folder and features_folder_path and colors_folder_path:

        result = get_images_in_categories(saved_folder)

        files_dict = {
            'dress': result['dress']['files'],
            'jacket': result['jacket']['files'],
            'pants': result['pants']['files'],
            'shirt': result['shirt']['files'],
            'shoe': result['shoe']['files'],
            'shorts': result['shorts']['files'],
            'skirts': result['skirt']['files']
        }
        vgg19_feature_extractor = load_vgg19_feature_extractor()
        processed_images_result, dominant_color_result = process_images_for_categories_with_dominant_color(
            files_dict)

        extract_features_and_save(processed_images_result, vgg19_feature_extractor, features_folder_path)
        save_dominant_colors(dominant_color_result, colors_folder_path)
        return jsonify({'massage': 'Done'}), 200
