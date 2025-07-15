import os
import yaml

# Step 1: Define paths to your dataset
train_images_dir = './aqua dataset/train/images'
train_annotations_dir = './aqua dataset/train/labels'
val_images_dir = './aqua dataset/valid/images'
val_annotations_dir = './aqua dataset/valid/labels'
output_yaml = './aquadataset.yml'

# Step 2: Define class names
class_names = ['class1', 'class2', 'class3']  # List of class names present in your dataset

# Step 3: Get list of image filenames
train_image_files = os.listdir(train_images_dir)
val_image_files = os.listdir(val_images_dir)

# Step 4: Create annotations dictionary
train_annotations = []
for image_file in train_image_files:
    # Assuming annotation filenames correspond to image filenames but with different extension
    annotation_file = image_file.split('.')[0] + '.txt'
    train_annotations.append(os.path.join(train_annotations_dir, annotation_file))

val_annotations = []
for image_file in val_image_files:
    # Assuming annotation filenames correspond to image filenames but with different extension
    annotation_file = image_file.split('.')[0] + '.txt'
    val_annotations.append(os.path.join(val_annotations_dir, annotation_file))

# Step 5: Write dataset YAML file
dataset = {
    'train': [{'img_path': os.path.join(train_images_dir, img), 'ann_path': ann} 
              for img, ann in zip(train_image_files, train_annotations)],
    'val': [{'img_path': os.path.join(val_images_dir, img), 'ann_path': ann} 
            for img, ann in zip(val_image_files, val_annotations)],
    'nc': len(class_names),
    'names': class_names
}

with open(output_yaml, 'w') as f:
    yaml.dump(dataset, f)

print("Dataset YAML file created successfully!")
