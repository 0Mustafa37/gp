import os
import pickle
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from keras import Model
import tensorflow as tf
from PIL import Image


def load_vgg19_feature_extractor():
    vgg_model = tf.keras.applications.vgg19.VGG19(weights='imagenet')
    feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
    feat_extractor.summary()

    return feat_extractor


def get_images_in_categories(base_folder):
    categories = ['dress', 'jacket', 'pants', 'shirt', 'shoe', 'shorts', 'skirt']

    def get_files_and_count(directory):
        files = [os.path.join(directory, x).replace('\\', '/') for x in os.listdir(directory) if
                 os.path.isfile(os.path.join(directory, x))]
        count = len(files)
        return files, count

    images_data = {}

    for category in categories:
        category_path = os.path.join(base_folder, category)
        if os.path.exists(category_path):
            files, count = get_files_and_count(category_path)
        else:
            files, count = [], 0
        images_data[category] = {'count': count, 'files': files}

    return images_data


def remove_background_AND_resize(image_path):
    # Load the image
    imgo = cv2.imread(image_path)
    if imgo is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    # Convert to RGB (since OpenCV loads images in BGR format)
    imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2RGB)

    # Create a mask holder
    mask = np.zeros(imgo.shape[:2], np.uint8)

    # Define the rectangle (ROI) within the image boundaries
    rect = (10, 10, imgo.shape[1] - 20, imgo.shape[0] - 20)

    # Ensure the ROI rectangle is within image boundaries
    x, y, w, h = rect
    x = max(0, x)
    y = max(0, y)
    w = min(imgo.shape[1] - x, w)
    h = min(imgo.shape[0] - y, h)
    rect = (x, y, w, h)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(imgo, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Modify mask to binary mask
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply mask to image
    img1 = imgo * mask[:, :, np.newaxis]

    # Get the background
    background = imgo - img1

    # Change all pixels in the background that are not black to white
    background[np.where((background > [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    # Add the background and the image
    final = background + img1

    # Convert the NumPy array (OpenCV format) to a PIL Image
    img = Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))

    # Resize image to 224x224
    img_resized = img.resize((224, 224))

    img_bgr = np.array(img_resized)

    # Convert RGB to BGR (OpenCV format)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # plt.figure(figsize=(4, 4))
    # plt.imshow(np.uint8(img_rgb))
    # plt.title('Original Product')
    # plt.axis('off')
    # plt.show()
    return img_rgb


def get_dominant_color(image, k=1):
    """
    Get the dominant color of an image using k-means clustering.

    Parameters:
    - image: The input image in RGB format.
    - k: The number of clusters (colors) to form. Default is 1 for the dominant color.

    Returns:
    - dominant_color: The dominant color in RGB format.
    """
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)

    # Use k-means clustering to find the dominant color
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get the cluster centers (the dominant colors)
    dominant_color = kmeans.cluster_centers_.astype(int)[0]

    return dominant_color


def process_images_with_dominant_color(files):
    imported_images = []
    dominant_colors = []

    for f in files:
        filename = f
        # Load image and perform necessary preprocessing
        image = remove_background_AND_resize(filename)
        numpy_image = tf.keras.preprocessing.image.img_to_array(image)

        # Get the dominant color
        dominant_color = get_dominant_color(numpy_image)
        dominant_colors.append(dominant_color)

        # Prepare image batch
        image_batch = np.expand_dims(numpy_image, axis=0)
        preprocessed_image = tf.keras.applications.imagenet_utils.preprocess_input(image_batch.copy())
        imported_images.append(preprocessed_image)

    if imported_images:
        processed_images = np.vstack(imported_images)
        return processed_images, dominant_colors
    else:
        return None, None


def process_images_for_categories_with_dominant_color(files_dict):
    processed_images_dict = {}
    dominant_colors_dict = {}

    for category, files in files_dict.items():
        processed_images, dominant_colors = process_images_with_dominant_color(files)
        if processed_images is not None:
            processed_images_dict[category] = processed_images
            dominant_colors_dict[category] = dominant_colors
            print(f'Processed images for {category}')
        else:
            print(f'There are no images for {category}')
    return processed_images_dict, dominant_colors_dict


def extract_features_and_save(processed_images_dict, feat_extractor, output_dir):
    features_dict = {}
    scaler = StandardScaler()
    for category, processed_images in processed_images_dict.items():
        # Extract features for each category
        features = feat_extractor.predict(processed_images)
        # features_normalized = scaler.fit_transform(features)
        features_dict[category] = features

        print(features_dict[category].shape)

        # Save features to pickle file
        filename = f'{output_dir}/{category}_features.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features for category '{category}' saved to {filename}")


def save_dominant_colors(dominant_colors_dict, output_dir):
    for category, colors in dominant_colors_dict.items():
        filename = f'{output_dir}/{category}_dominant_colors.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(colors, f)
        print(f"Dominant colors for category '{category}' saved to {filename}")


def load_category_features(pkl_dir):
    # Function to load a .pkl file
    def load_pkl_file(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    # Dictionary to store features from each category
    category_features = {}

    # Iterate through the directory and load each .pkl file
    for filename in os.listdir(pkl_dir):
        if filename.endswith('.pkl'):
            category = os.path.splitext(filename)[0]  # Get category name from filename
            file_path = os.path.join(pkl_dir, filename)
            category_features[category] = load_pkl_file(file_path)

    return category_features


def load_colors(pkl_dir):
    # Function to load a .pkl file
    def load_pkl_file(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    # Dictionary to store colors from each category
    color_features = {}

    # Iterate through the directory and load each .pkl file
    for filename in os.listdir(pkl_dir):
        if filename.endswith('.pkl'):
            category = os.path.splitext(filename)[0]  # Get category name from filename
            file_path = os.path.join(pkl_dir, filename)
            color_features[category] = load_pkl_file(file_path)

    return color_features
