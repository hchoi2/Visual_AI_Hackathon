
import fiftyone as fo
dataset = fo.Dataset("fruit_dataset")

# Assuming you have a list of image paths and corresponding labels
# For example:
# image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg", ...]
# labels = ["cat", "dog", ...]

# Add samples with labels
# Load images and labels from ./docs/fruit_data.csv
import pandas as pd
import os

data = pd.read_csv('./docs/fruit_data.csv')
image_paths = data['image'].tolist()
labels = data['label'].tolist()
for image_path, label in zip(image_paths, labels):
    sample = fo.Sample(filepath=image_path)
    # Add classification label
    sample["classification"] = fo.Classification(label=label)
    dataset.add_sample(sample)
# Launch the visualization app
session = fo.launch_app(dataset)

# Keep the session alive (if running in a script)
session.wait()