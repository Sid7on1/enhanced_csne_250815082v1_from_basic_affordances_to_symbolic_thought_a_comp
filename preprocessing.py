import logging
import os
import tempfile
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG = {
    "image_dir": "images",
    "annotations_file": "annotations.csv",
    "image_width": 640,
    "image_height": 480,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "data_split": 0.8,
    "batch_size": 32,
    "num_workers": 4,
    "pin_memory": True,
}

# Exception classes
class PreprocessingError(Exception):
    pass


class ImageNotFoundError(PreprocessingError):
    pass


class InvalidAnnotationError(PreprocessingError):
    pass


# Main class with methods
class ImagePreprocessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, CONFIG["image_dir"])
        self.annotations_file = os.path.join(data_dir, CONFIG["annotations_file"])
        self.images = []
        self.annotations = pd.DataFrame()
        self.classes = []
        self.class_dict = {}
        self.data = []
        self.transform = None

    def load_images(self) -> List[str]:
        images = [
            f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))
        ]
        if not images:
            raise ImageNotFoundError("No images found in the directory.")
        self.images = images
        return images

    def load_annotations(self) -> pd.DataFrame:
        try:
            self.annotations = pd.read_csv(self.annotations_file)
            if "image_id" not in self.annotations.columns or "class_id" not in self.annotations.columns:
                raise InvalidAnnotationError(
                    "Annotations file is missing required columns."
                )
            self.annotations["image_id"] = self.annotations["image_id"].apply(
                lambda x: os.path.splitext(x)[0]
            )
        except FileNotFoundError:
            raise InvalidAnnotationError("Annotations file not found.")
        return self.annotations

    def extract_classes(self) -> List[str]:
        self.classes = sorted(self.annotations["class_id"].unique().tolist())
        return self.classes

    def create_class_dict(self) -> Dict[int, str]:
        self.class_dict = {idx: cls for idx, cls in enumerate(self.classes)}
        return self.class_dict

    def preprocess_images(self) -> None:
        for image in self.images:
            image_path = os.path.join(self.image_dir, image)
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                continue

            # Apply image preprocessing steps here
            # ...

            # Add preprocessed image and its annotations to the data list
            self.data.append((image, self.annotations[self.annotations["image_id"] == image]))

        logger.info("Finished preprocessing images.")

    def create_dataset(self) -> "ImageDataset":
        dataset = ImageDataset(self.data, self.class_dict, transform=self.transform)
        return dataset


class ImageDataset(Dataset):
    def __init__(self, data: List[Tuple[np.ndarray, pd.DataFrame]], class_dict: Dict[int, str], transform=None):
        self.data = data
        self.class_dict = class_dict
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image, annotations = self.data[idx]

        # Convert image to RGB if necessary
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert annotations to labels
        labels = annotations["class_id"].map(self.class_dict).tolist()

        sample = {"image": image, "labels": labels}
        return sample


# Helper classes and utilities
class Transform:
    def __init__(self, mean: List[float], std: List[float], size: Optional[Tuple[int, int]] = None):
        self.mean = mean
        self.std = std
        self.size = size

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        image = image - np.array(self.mean)
        image = image / np.array(self.std)
        if self.size:
            image = cv2.resize(image, self.size)
        return torch.from_numpy(image.transpose((2, 0, 1)))


# Configuration management
def update_config(config_dict: Dict) -> None:
    for key, value in config_dict.items():
        if key in CONFIG:
            CONFIG[key] = value
        else:
            logger.warning(f"Invalid config key: {key}. Ignoring update.")

# Performance monitoring
def monitor_performance(dataset: ImageDataset, model: torch.nn.Module) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"]
    )
    total_time = 0.0
    for images, _ in data_loader:
        images = images.to(device)
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        _ = model(images)
        end_time.record()
        torch.cuda.synchronize()
        iteration_time = start_time.elapsed_time(end_time) * 1000
        total_time += iteration_time
    avg_time = total_time / len(data_loader)
    logger.info(f"Average inference time: {avg_time:.2f} ms")

# Resource cleanup
def cleanup_resources() -> None:
    logger.info("Cleaning up temporary files...")
    temp_dir = tempfile.gettempdir()
    for root, dirs, files in os.walk(temp_dir):
        for filename in files:
            if filename.startswith("enhanced_cs.NE_2508.15082v1_From_Basic_Affordances_to_Symbolic_Thought_A_Comp"):
                os.remove(os.path.join(root, filename))
    logger.info("Temporary files cleaned up.")

# Entry point
def main() -> None:
    data_dir = "/path/to/data"  # Replace with your data directory
    preprocessor = ImagePreprocessor(data_dir)
    images = preprocessor.load_images()
    annotations = preprocessor.load_annotations()
    classes = preprocessor.extract_classes()
    preprocessor.create_class_dict()
    preprocessor.preprocess_images()
    dataset = preprocessor.create_dataset()

    # Example usage of the dataset
    mean = CONFIG["mean"]
    std = CONFIG["std"]
    transform = Transform(mean, std, size=(CONFIG["image_width"], CONFIG["image_height"]))
    dataset.transform = transform

    batch_size = CONFIG["batch_size"]
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"]
    )

    # Example model definition
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 6, 5)
            self.conv2 = torch.nn.Conv2d(6, 16, 5)
            self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
            self.fc2 = torch.nn.Linear(120, 84)
            self.fc3 = torch.nn.Linear(84, len(classes))

        def forward(self, x):
            x = torch.nn.functional.relu(self.conv1(x))
            x = torch.nn.functional.max_pool2d(x, 2)
            x = torch.nn.functional.relu(self.conv2(x))
            x = torch.nn.functional.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = SimpleModel()

    # Example training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(10):  # Replace with your desired number of epochs
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch [{epoch+1}/{10}] Loss: {loss.item():.4f}")

    # Example performance monitoring
    monitor_performance(dataset, model)

    # Resource cleanup
    cleanup_resources()

if __name__ == "__main__":
    main()