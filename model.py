import torch
import torch.nn.functional as F
import numpy as np
import os
import hashlib
from collections import OrderedDict
from PIL import Image
import cv2

class CTScanModelWrapper:
    """Wrapper class for loading and processing CT scans using a pre-trained PyTorch model."""

    def __init__(self, model_path="ct_scan_model.pt", device=None, cache_size=10):
        """
        Initialize the model wrapper.
        
        :param model_path: Path to the pre-trained PyTorch model (.pt).
        :param device: Use 'cuda' if available, otherwise fallback to 'cpu'.
        :param cache_size: Maximum number of cached results.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.cache = OrderedDict()  # Cache to store previous results
        self.cache_size = cache_size

    def _load_model(self, model_path):
        """
        Load the pre-trained PyTorch model.
        
        :param model_path: Path to the model file.
        :return: Loaded model.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = torch.load(model_path, map_location=self.device)
        model.eval()  # Set model to evaluation mode
        return model

    def _preprocess_image(self, image_path):
        """
        Preprocess input CT scan for model inference.
        
        :param image_path: Path to the image file.
        :return: Preprocessed image tensor.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        image = cv2.resize(image, (512, 512))  # Resize to model input size
        image = image.astype(np.float32) / 255.0  # Normalize pixel values (0-1)
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Add batch & channel dim
        return image.to(self.device)

    def _generate_cache_key(self, image_path):
        """
        Generate a hash key for caching based on image contents.
        
        :param image_path: Path to the image file.
        :return: Unique cache key.
        """
        with open(image_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash

    def _update_cache(self, key, result):
        """
        Update the cache with new results.
        
        :param key: Cache key.
        :param result: Model prediction result.
        """
        if key not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)  # Remove oldest entry
            self.cache[key] = result

    def predict(self, image_paths):
        """
        Run inference on a batch of CT scan images.
        
        :param image_paths: List of image file paths.
        :return: List of predictions.
        """
        results = []

        # Process each image
        for image_path in image_paths:
            cache_key = self._generate_cache_key(image_path)

            # Check cache first
            if cache_key in self.cache:
                results.append(self.cache[cache_key])
                continue

            # Preprocess image
            image_tensor = self._preprocess_image(image_path)

            # Run inference
            with torch.no_grad():
                prediction = self.model(image_tensor)
                confidence = torch.sigmoid(prediction).item() * 100  # Convert to percentage
                label = "Tumor Detected" if confidence > 50 else "No Tumor"

            result = {"prediction": label, "confidence": confidence, "image_path": image_path}

            # Update cache
            self._update_cache(cache_key, result)

            results.append(result)

        return results
