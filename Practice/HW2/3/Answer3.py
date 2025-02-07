from PIL import Image, ImageDraw
import random
import numpy as np
import matplotlib.pyplot as plt

class CreateData:
    def __init__(self, paths: list):
        self.path = paths

    def convertImageToBinary(self, path):
        """
        Convert an image to a binary representation based on pixel intensity.

        Args:
            path (str): The file path to the input image.

        Returns:
            list: A binary representation of the image where white is represented by -1 and black is represented by 1.
        """
        # Open the image file.
        image = Image.open(path)

        # Create a drawing tool for manipulating the image.
        draw = ImageDraw.Draw(image)

        # Determine the image's width and height in pixels.
        width = image.size[0]
        height = image.size[1]

        # Load pixel values for the image.
        pix = image.load()

        # Define a factor for intensity thresholding.
        factor = 100

        # Initialize an empty list to store the binary representation.
        binary_representation = []

        # Loop through all pixels in the image.
        for i in range(width):
            for j in range(height):
                # Extract the Red, Green, and Blue (RGB) values of the pixel.
                red = pix[i, j][0]
                green = pix[i, j][1]
                blue = pix[i, j][2]

                # Calculate the total intensity of the pixel.
                total_intensity = red + green + blue

                # Determine whether the pixel should be white or black based on the intensity.
                if total_intensity > (((255 + factor) // 2) * 3):
                    red, green, blue = 255, 255, 255  # White pixel
                    binary_representation.append(-1)
                else:
                    red, green, blue = 0, 0, 0  # Black pixel
                    binary_representation.append(1)

                # Set the pixel color accordingly.
                draw.point((i, j), (red, green, blue))

        del draw

        return binary_representation

    def __getNoisyBinaryImage(self, input_path, output_path, missing_point):
        """
        Add noise to an image and save it as a new file.

        Args:
            input_path (str): The file path to the input image.
            output_path (str): The file path to save the noisy image.
        """
        # Open the input image.
        image = Image.open(input_path)

        # Create a drawing tool for manipulating the image.
        draw = ImageDraw.Draw(image)

        # Determine the image's width and height in pixels.
        width = image.size[0]
        height = image.size[1]

        # Load pixel values for the image.
        pix = image.load()

        # Define a factor for introducing noise.
        
        if not missing_point:
            noise_factor = 2000 
        else:
           noise_factor = 500

        for i in range(width):
            for j in range(height):
                # Generate a random noise value within the specified factor.
                if not missing_point:
                    rand = random.randint(-noise_factor, noise_factor)
                else:
                    rand = random.randint(0, noise_factor)
                
                # Add the noise to the Red, Green, and Blue (RGB) values of the pixel.
                red = pix[i, j][0] + rand
                green = pix[i, j][1] + rand
                blue = pix[i, j][2] + rand

                # Ensure that RGB values stay within the valid range (0-255).
                if red < 0:
                    red = 0
                if green < 0:
                    green = 0
                if blue < 0:
                    blue = 0
                if red > 255:
                    red = 255
                if green > 255:
                    green = 255
                if blue > 255:
                    blue = 255

                # Set the pixel color accordingly.
                draw.point((i, j), (red, green, blue))

        image.save(output_path, "JPEG")

        del draw
    
    def generate_noisy_images(self):
        for i, image_path in enumerate(self.path, start=1):
            noisy_image_path = f"./noisy{i}.jpg"
            self.__getNoisyBinaryImage(image_path, noisy_image_path, False)
            print(f"Noisy image for {image_path} generated and saved as {noisy_image_path}")

    def generate_missing_point_images(self):
        for i, image_path in enumerate(self.path, start=1):
            missing_point_path = f"./MissingPoint{i}.jpg"
            self.__getNoisyBinaryImage(image_path, missing_point_path, True)
            print(f"Noisy image for {image_path} generated and saved as {missing_point_path}")



class HammingNetwork:
    def __init__(self, clean_image_paths):
        """
        Initialize the Hamming Network with clean binary images.

        Parameters:
            clean_images: list of np.array, the set of normal binary images.
        """
        clean_images = [self.load_and_preprocess_image(path) for path in clean_image_paths]
        self.clean_images = [image.flatten() for image in clean_images]  # Flatten for vector processing

    def match(self, noisy_image):
        """
        Matches the noisy binary image to the closest clean image.

        Parameters:
            noisy_image: np.array, the noisy binary image to be matched.

        Returns:
            index: int, the index of the matched clean image.
            matched_image: PIL.Image.Image, the matched clean image.
        """
        noisy_image = noisy_image.flatten()  # Flatten the noisy image
        # Compute Hamming distances to each clean image
        distances = [np.sum(noisy_image != clean_image) for clean_image in self.clean_images]
        index = np.argmin(distances)  # Find the index of the smallest distance

        # Convert the matched image back to 2D array
        matched_array = self.clean_images[index].reshape(96, 96)

        # Convert the binary array (0s and 1s) back to a grayscale image
        matched_image = Image.fromarray((matched_array * 255).astype(np.uint8))  # Scale 0/1 to 0/255
        return index, matched_image

    def load_and_preprocess_image(self, filepath, size=(96, 96)):
        """
        Loads and preprocesses an image for the Hamming Network.

        Parameters:
            filepath: str, path to the image file.
            size: tuple, dimensions to resize the image to.

        Returns:
            np.array: Binary representation of the image (0s and 1s).
        """
        # Load the image
        img = Image.open(filepath).convert("L")  
        img = img.resize(size) 
        binary_img = np.array(img) > 75  
        return binary_img.astype(np.int8)

if __name__ == "__main__":
    clean_image_paths = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]
    gen = CreateData(clean_image_paths)

    gen.generate_noisy_images()
    gen.generate_missing_point_images()

    noisy_image_path = "noisy1.jpg"

    hamming_network = HammingNetwork(clean_image_paths)
    noisy_image = hamming_network.load_and_preprocess_image(noisy_image_path)

    index, matched_image = hamming_network.match(noisy_image)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))  # 1 row, 2 columns

    axes[0].imshow(matched_image, cmap='gray')  # Display in grayscale
    axes[0].set_title(f"Matched Clean Image {index + 1}")
    axes[0].axis("off")  # Hide axes

    axes[1].imshow(noisy_image, cmap='gray')  # Display in grayscale
    axes[1].set_title(f"Noisy Image 1")
    axes[1].axis("off")  # Hide axes
    plt.tight_layout()
    plt.show()

    missing_point_path = "MissingPoint5.jpg"

    hamming_network = HammingNetwork(clean_image_paths)
    missing_point_image = hamming_network.load_and_preprocess_image(missing_point_path)

    index, matched_image = hamming_network.match(missing_point_image)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))  # 1 row, 2 columns

    axes[0].imshow(matched_image, cmap='gray')  # Display in grayscale
    axes[0].set_title(f"Matched Clean Image {index + 1}")
    axes[0].axis("off")  # Hide axes

    axes[1].imshow(missing_point_image, cmap='gray')  # Display in grayscale
    axes[1].set_title(f"Missing Point 5")
    axes[1].axis("off")  # Hide axes
    plt.tight_layout()
    plt.show()