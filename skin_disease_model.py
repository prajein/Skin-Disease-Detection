import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import cv2
from PIL import Image
import glob

# Constants
IMG_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 7

def create_model():
    """Create a simple MobileNetV2-based model for skin disease classification"""
    # Use a pre-trained MobileNetV2 model as the base
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(img_path):
    """Load and preprocess an image for prediction"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        return img
    except Exception as e:
        print(f"Error preprocessing image {img_path}: {e}")
        return None

def find_image_files(image_id):
    """Attempt to find image files in various possible locations"""
    # List of potential locations to search for the image
    potential_paths = [
        f"Dataset/images/{image_id}.jpg",
        f"Dataset/{image_id}.jpg",
        f"uploadimage/{image_id}.jpg",
        f"Dataset/HAM10000_images_part_1/{image_id}.jpg",
        f"Dataset/HAM10000_images_part_2/{image_id}.jpg"
    ]
    
    # Add any files in Dataset directory with the image_id in the name
    dataset_files = glob.glob(f"Dataset/**/{image_id}.*", recursive=True)
    potential_paths.extend(dataset_files)
    
    # Check if any of the potential paths exist
    for path in potential_paths:
        if os.path.exists(path):
            return path
    
    # If no match is found, do a more comprehensive search
    for root, dirs, files in os.walk("Dataset"):
        for file in files:
            if image_id in file:
                return os.path.join(root, file)
    
    # Check if the image is in uploadimage directory with any extension
    upload_files = glob.glob(f"uploadimage/{image_id}.*")
    if upload_files:
        return upload_files[0]
        
    return None

def analyze_metadata_file():
    """Analyze the metadata file to get dataset statistics and class distribution"""
    try:
        metadata_path = 'Dataset/HAM10000_metadata.csv'
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found at {metadata_path}")
            return None
            
        df = pd.read_csv(metadata_path)
        print(f"Dataset contains {len(df)} entries")
        
        # Show class distribution
        print("\nClass distribution:")
        class_counts = df['dx'].value_counts()
        for cls, count in class_counts.items():
            print(f"{cls}: {count} ({count/len(df)*100:.2f}%)")
            
        # Show age and sex distribution
        print(f"\nAge range: {df['age'].min()} to {df['age'].max()}, Mean: {df['age'].mean():.1f}")
        print("\nGender distribution:")
        gender_counts = df['sex'].value_counts()
        for gender, count in gender_counts.items():
            print(f"{gender}: {count} ({count/len(df)*100:.2f}%)")
            
        # Show localization distribution
        print("\nTop 5 lesion locations:")
        loc_counts = df['localization'].value_counts().head(5)
        for loc, count in loc_counts.items():
            print(f"{loc}: {count} ({count/len(df)*100:.2f}%)")
            
        return df
    except Exception as e:
        print(f"Error analyzing metadata: {e}")
        return None

def load_data_from_directory():
    """Load data from Dataset/HAM10000_metadata.csv and associated images"""
    try:
        print("Analyzing metadata file...")
        df = analyze_metadata_file()
        
        if df is None:
            print("Metadata file could not be loaded. Using dummy data.")
            return create_dummy_data()
        
        print("\nLooking for images based on metadata...")
        
        # Encode disease labels
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['dx'])
        class_names = le.classes_
        print(f"Class names (encoded): {list(zip(class_names, range(len(class_names))))}")
        
        # Find image paths matching metadata
        image_paths = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                print(f"Processing metadata row {idx}/{len(df)}")
            
            image_path = find_image_files(row['image_id'])
            
            if image_path:
                image_paths.append(image_path)
                valid_indices.append(idx)
            else:
                if idx < 10:  # Only print the first few to avoid flooding the console
                    print(f"Image not found for ID: {row['image_id']}")
        
        # Filter metadata to only include entries with found images
        df = df.iloc[valid_indices]
        
        # Check if we have enough data
        if len(image_paths) < 10:
            print(f"Not enough images found. Only {len(image_paths)} images located. Using dummy data.")
            return create_dummy_data()
        
        print(f"Found {len(image_paths)} valid images out of {len(df)} metadata entries")
        
        # Sample a subset of the data to speed up training for demonstration
        if len(image_paths) > 1000:
            sample_size = 1000
            sample_indices = np.random.choice(len(image_paths), sample_size, replace=False)
            image_paths = [image_paths[i] for i in sample_indices]
            df = df.iloc[sample_indices]
            print(f"Sampled {sample_size} images for faster training")
        
        # Load and preprocess images
        images = []
        labels = []
        
        for i, path in enumerate(image_paths):
            if i % 100 == 0:
                print(f"Processing image {i}/{len(image_paths)}")
            
            img = preprocess_image(path)
            if img is not None:
                images.append(img)
                labels.append(df.iloc[i]['label'])
        
        # Convert to numpy arrays
        X = np.array(images)
        y = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test, class_names
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return create_dummy_data()

def create_dummy_data():
    """Create dummy data for testing when real data is not available"""
    print("Creating dummy data for testing...")
    
    # Create random images and labels
    X_train = np.random.random((100, IMG_SIZE, IMG_SIZE, 3))
    X_test = np.random.random((20, IMG_SIZE, IMG_SIZE, 3))
    
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, size=100), num_classes=NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, size=20), num_classes=NUM_CLASSES)
    
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    return X_train, X_test, y_train, y_test, class_names

def train_and_save_model():
    """Train the model and save it to disk"""
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test, class_names = load_data_from_directory()
    
    # Create and compile model
    print("Creating model...")
    model = create_model()
    
    # Create data generators with augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )
    
    validation_datagen = ImageDataGenerator()
    
    # Train the model
    print(f"Training model for {EPOCHS} epochs...")
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE),
        validation_steps=len(X_test) // BATCH_SIZE
    )
    
    # Evaluate the model
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save the model
    print("Saving model...")
    model.save("sd_model.keras")
    print("Model saved as sd_model.keras")
    
    # Save class names to a file for reference
    with open("class_names.txt", "w") as f:
        f.write(",".join(class_names))
    print("Class names saved to class_names.txt")
    
    return model, class_names

if __name__ == "__main__":
    train_and_save_model() 