import cv2
import numpy as np

def preprocess_image(image_path: str, target_size=(640, 640), alpha=1.5, beta=30) -> np.ndarray:
    """
    Loads an image, applies a simple contrast/brightness adjustment to simulate 
    low-light enhancement, resizes it, and normalizes pixel values to [0.0, 1.0].
    
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Target size to resize the image to (width, height).
        alpha (float): Contrast control (1.0-3.0).
        beta (int): Brightness control (0-100).
        
    Returns:
        np.ndarray: Preprocessed image as a numpy array, normalized to [0.0, 1.0].
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image from path: {image_path}")
    
    # Apply simple contrast and brightness adjustment
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Resize the image to standard YOLO dimensions (640x640)
    resized_image = cv2.resize(adjusted_image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize pixel values to [0.0, 1.0]
    normalized_image = resized_image.astype(np.float32) / 255.0
    
    return normalized_image

if __name__ == "__main__":
    # Example usage:
    # try:
    #     preprocessed_img = preprocess_image("path/to/sample.jpg")
    #     print("Image preprocessed successfully. Shape:", preprocessed_img.shape)
    # except Exception as e:
    #     print(f"Error: {e}")
    pass
