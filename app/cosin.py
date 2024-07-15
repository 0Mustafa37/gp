import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import shutil
from PIL import Image

from app.helper import get_dominant_color


def retrieve_most_similar_products(given_img, imgs_features, imgs_colors, files, feat_extractor):

    # Load and preprocess the given image
    image = Image.open(given_img)
    resized_image = image.resize((224, 224))
    img_bgr = np.array(resized_image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    numpy_image = tf.keras.preprocessing.image.img_to_array(img_rgb)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_img = tf.keras.applications.imagenet_utils.preprocess_input(image_batch.copy())
    img_features = feat_extractor.predict(processed_img)

    # Assuming dominant color extraction for given_img is done elsewhere and is available as a variable
    # given_img_color = get_dominant_color(numpy_image)

    # Stack features and colors for similarity comparison
    test_features = np.vstack([imgs_features, img_features])
    # test_colors = np.vstack([imgs_colors, given_img_color])

    # Make a copy of files list and append given_img
    files_copy = files.copy()
    files_copy.append(given_img)

    # Calculate cosine similarities for both features and colors
    cos_sim_features = cosine_similarity(test_features)
    # cos_sim_colors = cosine_similarity(test_colors)

    # Combine similarities in a meaningful way
    # Example: Weighted combination
    alpha = 0.7
    beta = 0.3
    # combined_similarity = alpha * cos_sim_features + beta * cos_sim_colors

    # Create DataFrame with similarities
    cos_similarities_df = pd.DataFrame(cos_sim_features, columns=files_copy, index=files_copy)

    # Retrieve closest images excluding the given_img itself
    closest_imgs = cos_similarities_df[given_img]
    closest_imgs = closest_imgs[closest_imgs.index != given_img]
    closest_imgs = closest_imgs.to_frame()
    closest_imgs.index = closest_imgs.index.str.split('\\').str[-1]

    # Sort by similarity and get top five
    sorted_imgs = closest_imgs.sort_values(by=given_img, ascending=False)
    top_five = sorted_imgs.head(5)

    # Retrieve the top match and its similarity score
    top_match = top_five.index[0]
    highest_similarity_score = top_five.iloc[0][0]

    return top_match, highest_similarity_score


def get_images_from_directory(directory_path):
    all_images = []
    subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(directory_path, subdirectory)
        images = [os.path.join(subdirectory_path, f) for f in os.listdir(subdirectory_path) if
                  os.path.isfile(os.path.join(subdirectory_path, f))]
        all_images.append(images)
    return all_images


def generate_outfits(images_list, category_features, color_features, user_files, feat_extractor,
                     target_directory='generatedOutfits/'):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    best_three_outfits_dir = os.path.join(target_directory, 'bestThreeOutfits')
    if not os.path.exists(best_three_outfits_dir):
        os.makedirs(best_three_outfits_dir)

    top_outfits = []
    outfit_details = []

    i = 1
    for outfits in images_list:
        score_piece = 0
        outfit_images = []
        similar_images = []
        counter = 0
        for piece in outfits:
            input_path = ''
            path = os.path.join(target_directory, str(i))
            os.makedirs(path, exist_ok=True)
            image_name_with_extension = piece.split('\\')[-1]
            image_name = image_name_with_extension.split('.')[0]

            if image_name == 'dress':
                input_path = 'userDataset/dress/'
                if user_files['dress']['count'] == 0:
                    continue
                similar_image_path, score = retrieve_most_similar_products(piece, category_features['dress_features'],
                                                                           color_features['dress_dominant_colors'],
                                                                           user_files['dress']['files'], feat_extractor)
            elif image_name == 'jacket':
                input_path = 'userDataset/jacket/'
                if user_files['jacket']['count'] == 0:
                    continue
                similar_image_path, score = retrieve_most_similar_products(piece, category_features['jacket_features'],
                                                                           color_features['jacket_dominant_colors'],
                                                                           user_files['jacket']['files'],
                                                                           feat_extractor)
            elif image_name == 'pants':
                input_path = 'userDataset/pants/'
                if user_files['pants']['count'] == 0:
                    continue
                similar_image_path, score = retrieve_most_similar_products(piece, category_features['pants_features'],
                                                                           color_features['pants_dominant_colors'],
                                                                           user_files['pants']['files'], feat_extractor)
            elif image_name == 'shirt':
                input_path = 'userDataset/shirt/'
                if user_files['shirt']['count'] == 0:
                    continue
                similar_image_path, score = retrieve_most_similar_products(piece, category_features['shirt_features'],
                                                                           color_features['shirt_dominant_colors'],
                                                                           user_files['shirt']['files'], feat_extractor)
            elif image_name == 'shoe':
                input_path = 'userDataset/shoe/'
                if user_files['shoe']['count'] == 0:
                    continue
                similar_image_path, score = retrieve_most_similar_products(piece, category_features['shoe_features'],
                                                                           color_features['shoe_dominant_colors'],
                                                                           user_files['shoe']['files'], feat_extractor)
            elif image_name == 'shorts':
                input_path = 'userDataset/shorts/'
                if user_files['shorts']['count'] == 0:
                    continue
                similar_image_path, score = retrieve_most_similar_products(piece, category_features['shorts_features'],
                                                                           color_features['shorts_dominant_colors'],
                                                                           user_files['shorts']['files'],
                                                                           feat_extractor)
            elif image_name == 'skirts':
                input_path = 'userDataset/skirts/'
                if user_files['skirts']['count'] == 0:
                    continue
                similar_image_path, score = retrieve_most_similar_products(piece, category_features['skirts_features'],
                                                                           color_features['skirts_dominant_colors'],
                                                                           user_files['skirts']['files'],
                                                                           feat_extractor)
            else:
                continue

            score_piece += score
            counter += 1
            similar_image = cv2.imread(similar_image_path)
            output_path = os.path.join(path, f'{image_name}.jpg')
            cv2.imwrite(output_path, similar_image)
            outfit_images.append(piece)
            similar_images.append(similar_image_path)
        if counter == 3:
            top_outfits.append((score_piece, i))
            image_paths = [os.path.join(target_directory, str(i), f'{piece.split("\\")[-1].split(".")[0]}.jpg') for
                           piece in
                           outfits]
            outfit_details.append((score_piece, i, image_paths))

        if len(top_outfits) > 3:
            top_outfits.sort(key=lambda x: x[0], reverse=True)
            top_outfits.pop()
        i += 1

    outfit_details.sort(key=lambda x: x[0], reverse=True)

    outfit_details_df = pd.DataFrame(outfit_details, columns=['Score', 'Outfit Index', 'Image Paths'])
    csv_file_path = os.path.join(target_directory, 'outfit_details.csv')
    outfit_details_df.to_csv(csv_file_path, index=False)
    outfit_details = [item[2] for item in outfit_details]

    return outfit_details
