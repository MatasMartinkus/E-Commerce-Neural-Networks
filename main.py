import pandas as pd
import numpy as np
import tensorflow as tf
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from keras import callbacks


def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        # Convert to RGBA first to handle transparency
        img = img.convert('RGBA')
        img = img.resize((224, 224))
        # Then convert to RGB
        img = img.convert('RGB')
        return np.array(img)
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except UnidentifiedImageError as e:
        print(f"PIL error: {e}")
        return None
    except Exception as e:
        print(f"General error: {e}")
        return None

def preprocess_dataset(data_path, min_subcategory_size=30):
    df = pd.read_csv(data_path)

    if 'subcategory' not in df.columns:
        df['subcategory'] = np.nan 
        
    df['subcategory'] = df['subcategory'].fillna(df['category'])
    df = df.drop(columns=['category'])
    
    df = df.dropna(subset=['title', 'division', 'subcategory', 'description', 'price', 'seller',"image"])
    
    subcategory_counts = df['subcategory'].value_counts()
    valid_subcategories = subcategory_counts[subcategory_counts >= min_subcategory_size].index
    df = df[df['subcategory'].isin(valid_subcategories)]
    
    return df

def build_resnet50_model(num_classes, fine_tune=True):
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = False
    
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.5)(x) 
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    
    if fine_tune:
        for layer in base_model.layers[-30:]:
            layer.trainable = True
    
    return model

def build_cnn(num_classes):

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9)
    

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5), 
        tf.keras.layers.Dense(num_classes, activation='softmax')  
    ])

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )

    return model


def get_images_as_arrays(df):
    image_arrays = []
    successful_indices = []  
    
    for i, url in enumerate(df["image"]):
        try:
            img_array = load_image_from_url(url)
            if img_array is not None and img_array.shape == (224, 224, 3):
                image_arrays.append(img_array)
                successful_indices.append(i) 
            else:
                print(f"Skipping image at index {i} due to loading errors or shape issues.")

            if i % 1000 == 0 and i > 0:
                np.save('image_arrays_partial.npy', np.array(image_arrays))
                np.save('successful_indices_partial.npy', np.array(successful_indices))
                print(f"Currently converting row no. {i}")
                time.sleep(1)
        except Exception as e:
            print(f"Unexpected error at row {i}: {e}")
            np.save('image_arrays_error.npy', np.array(image_arrays))
            np.save('successful_indices_error.npy', np.array(successful_indices))
            raise
    
    np.save('image_arrays_final.npy', np.array(image_arrays))
    np.save('successful_indices_final.npy', np.array(successful_indices))
    
    return np.array(image_arrays), np.array(successful_indices)

def build_lstm(input_shape, num_classes):
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),

        # LSTM layers
        tf.keras.layers.LSTM(64, return_sequences=True),  # First LSTM layer
        tf.keras.layers.LSTM(64),  # Second LSTM layer

        # Fully connected layers
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Regularization
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )

    return model

def build_gru(num_classes):
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(1, 150528)),

        # GRU layers
        tf.keras.layers.GRU(128, return_sequences=True),  
        tf.keras.layers.GRU(64),  

        # Fully connected layers
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Regularization
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )

    return model

def visualise(model_name, history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{model_name}/loss.png")

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{model_name}/accuracy.png")

    if 'top3_accuracy' in history.history:
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['top3_accuracy'], label='Training Top-3 Accuracy')
        plt.plot(history.history['val_top3_accuracy'], label='Validation Top-3 Accuracy')
        plt.title('Model Top-3 Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Top-3 Accuracy')
        plt.legend()
        plt.savefig(f"{model_name}/TOP3_accuracy.png")

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))
    loss, accuracy, top3_accuracy = model.evaluate(X_test, y_test)
    print(f"{model_name} - Test loss: {loss}, Test accuracy: {accuracy}, Top-3 accuracy: {top3_accuracy}")
    visualise(model_name, history)
    return history

def plot_confusion_matrix(model, X_test, y_test, encoder, model_name, selected_classes=None):
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # If selected_classes is provided, filter the confusion matrix
    if selected_classes is not None:
        cm = cm[selected_classes][:, selected_classes]
        labels = [encoder.categories_[0][i] for i in selected_classes]
    else:
        labels = encoder.categories_[0]

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot normalized confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f"Normalized Confusion Matrix for {model_name}")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.savefig(f"{model_name}/Confusion_matrix_normalized.png")
    plt.show()


def split_dataset_by_subcategory(df, test_size=0.2, random_state=42):
    # Group the dataset by subcategory
    grouped = df.groupby("subcategory")

    # Create empty lists to store train and test data
    train_indices = []
    test_indices = []

    # Iterate through each subcategory group
    for subcategory, group in grouped:
        # Shuffle the group indices
        group_indices = group.index.to_list()
        np.random.seed(random_state)
        np.random.shuffle(group_indices)

        # Split the indices into train and test sets
        split_point = int(len(group_indices) * (1 - test_size))
        train_indices.extend(group_indices[:split_point])
        test_indices.extend(group_indices[split_point:])

    # Create train and test datasets
    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]

    return train_df, test_df

df = preprocess_dataset("combined_results.csv")
df = df.reset_index(drop=True)

image_arrays = np.load('image_arrays_final.npy')  
successful_indices = np.load('successful_indices_final.npy')  

df_filtered = df.iloc[successful_indices]
train_df, test_df = split_dataset_by_subcategory(df_filtered, test_size=0.2, random_state=42)

pd.DataFrame.to_csv(df_filtered,"results.csv")

train_indices = train_df.index.intersection(successful_indices)
test_indices = test_df.index.intersection(successful_indices)

train_positions = [np.where(successful_indices == idx)[0][0] for idx in train_indices if idx in successful_indices]
test_positions = [np.where(successful_indices == idx)[0][0] for idx in test_indices if idx in successful_indices]

X_train = image_arrays[train_positions]
X_test = image_arrays[test_positions]

encoder = OneHotEncoder()
y_train = encoder.fit_transform(train_df["subcategory"].to_numpy().reshape(-1, 1)).toarray()
y_test = encoder.transform(test_df["subcategory"].to_numpy().reshape(-1, 1)).toarray()

top_classes = np.argsort(np.bincount(np.argmax(y_test, axis=1)))[-20:]  # Top 5 classes

num_classes = y_train.shape[1] 

if __name__ == "__main__":


    model = build_resnet50_model(num_classes)

    history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

    loss, accuracy, top3_accuracy = model.evaluate(X_test, y_test)

    print(f"Test loss: {loss}")
    print(f"Test accuracy: {accuracy}")
    print(f"Top-3 accuracy: {top3_accuracy}")

    plot_confusion_matrix(model, X_test, y_test, encoder, "resnet",top_classes)
    visualise("resnet", history)

    model.save("resnet_model.h5")

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',  
        patience=10,
        restore_best_weights=True 
    )

    model = build_cnn(num_classes)
    history = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test),callbacks=[early_stopping])
    plot_confusion_matrix(model, X_test, y_test, encoder, "cnn", top_classes)

    loss, accuracy, top3_accuracy = model.evaluate(X_test, y_test)

    print(f"Test loss: {loss}")
    print(f"Test accuracy: {accuracy}")
    print(f"Top-3 accuracy: {top3_accuracy}")

    visualise("cnn", history)
    model.save("cnn_model.h5")

    X_train_flat = X_train.reshape((X_train.shape[0], 1, -1))
    X_test_flat = X_test.reshape((X_test.shape[0], 1, -1)) 

    # model = build_gru(num_classes)
    # history = model.fit(X_train_flat, y_train, epochs=200, batch_size=64, validation_data=(X_test_flat, y_test),callbacks=[early_stopping])
    # plot_confusion_matrix(model, X_test, y_test, encoder, "gru", top_classes)

    # loss, accuracy, top3_accuracy = model.evaluate(X_test, y_test)

    # print(f"Test loss: {loss}")
    # print(f"Test accuracy: {accuracy}")
    # print(f"Top-3 accuracy: {top3_accuracy}")

    # visualise("gru", history)
    # model.save("gru_model.h5")



