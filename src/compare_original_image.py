import shutil
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Define the root directory containing the image directories
root_dir = 'data'
output_dir = "output"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.mkdir(output_dir)

# List all subdirectories in the root directory
subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# Number of iterations
num_iterations = 10

for i in range(num_iterations):
    # Initialize a list to store the images and their corresponding labels for this iteration
    images = []
    labels = []

    # Loop through each subdirectory
    for subdir in subdirs:
        # Extract the label from the directory name
        label = os.path.basename(subdir)

        # List all files in the subdirectory
        files = [f for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))]

        # Randomly select one image file
        selected_file = random.choice(files)

        # Open and resize the selected image
        img_path = os.path.join(subdir, selected_file)
        img = Image.open(img_path)
        img_resized = img.resize((100, 100))  # Resize to 100x100 pixels
        images.append(img_resized)
        labels.append(label)

    # Plot all resized images with their corresponding labels
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')

    # Save the plot as a PNG file with a unique name for each iteration
    plt.savefig(f"{output_dir}/output_compare_original_image_{i + 1}.png", format="png")
    plt.close(fig)  # Close the figure to free memory
