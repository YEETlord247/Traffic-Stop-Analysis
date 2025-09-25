import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ✅ Check GPU Availability
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices('GPU')) > 1 else tf.distribute.OneDeviceStrategy(device="/GPU:0")

# ✅ Configuration
IMG_SIZE = 224
PATCH_SIZE = 16
BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 10
PROJECTION_DIM = 128
NUM_HEADS = 8
MLP_UNITS = [256, 128]
DROPOUT_RATE = 0.1
L2_REGULARIZATION = 1e-5
LR = 3e-4
DATASET_PATH = "dataset"
PLOT_SAVE_PATH = "plots"
AUTOTUNE = tf.data.experimental.AUTOTUNE

# ✅ Load the datasets
df1 = pd.read_csv('path_to_speed_vs_time.csv')  # Speed vs. time dataset
df2 = pd.read_csv('path_to_event_metadata.csv')  # Metadata (RecordingId, categories)

# ✅ Convert time columns to datetime
df1['_time'] = pd.to_datetime(df1['_time'])
df2['StartTime'] = pd.to_datetime(df2['StartTime'])
df2['EndTime'] = pd.to_datetime(df2['EndTime'])

# ✅ Data Augmentation Pipeline
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
    layers.Rescaling(1.0 / 255)
])


# ✅ Function to generate speed vs time plot
def generate_speed_plot(recording_id, save_dir=PLOT_SAVE_PATH, dpi=64):
    try:
        event_data = df2[df2['RecordingId'] == recording_id]
        if event_data.empty:
            raise ValueError(f"Recording ID {recording_id} not found in metadata.")

        event_data = event_data.iloc[0]
        serial = event_data['Serial']
        start_time = event_data['StartTime']
        end_time = event_data['EndTime']

        speed_data = df1[(df1['host'] == serial) & (df1['_time'] >= start_time) & (df1['_time'] <= end_time)]
        if speed_data.empty:
            raise ValueError(f"No speed data available for Recording ID {recording_id}.")

        plt.figure(figsize=(2, 2))
        plt.plot(speed_data['_time'], speed_data['Speed'], color='blue', linewidth=1)
        plt.title('Speed vs. Time')
        plt.xlabel('Time')
        plt.ylabel('Speed')
        plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image_path = os.path.join(save_dir, f"{recording_id}.png")
        plt.savefig(image_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"Saved plot for {recording_id} as {image_path}")

    except Exception as e:
        print(f"Error generating plot for {recording_id}: {e}")

# ✅ Generate plots for multiple recordings
def generate_multiple_plots(recording_ids, save_dir=PLOT_SAVE_PATH, dpi=64):
    for recording_id in recording_ids:
        generate_speed_plot(recording_id, save_dir, dpi)

# ✅ Prepare Image Dataset for Training
def preprocess_dataset():
    dataset = keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        label_mode='int',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    ).map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)

    return dataset.prefetch(buffer_size=AUTOTUNE)

# ✅ Prepare Time-Series Data
def preprocess_time_series(df):
    timestamps = df['_time'].astype(np.int64).values.reshape(-1, 1)
    speeds = df['Speed'].values.reshape(-1, 1)

    return np.hstack([timestamps, speeds])

# ✅ Convert Time-Series Data into TensorFlow Dataset
def create_time_series_dataset(df, batch_size=BATCH_SIZE):
    tensor_data = preprocess_time_series(df)
    dataset = tf.data.Dataset.from_tensor_slices(tensor_data).batch(batch_size).prefetch(AUTOTUNE)

    return dataset

# ✅ Vision Transformer (ViT) Implementation

# ✅ Multi-Head Self-Attention Layer
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=NUM_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self._split_heads(query, batch_size)
        key = self._split_heads(key, batch_size)
        value = self._split_heads(value, batch_size)

        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        output = tf.matmul(attention_scores, value)
        output = self._combine_heads(output, batch_size)
        return self.combine_heads(output)

    def _split_heads(self, x, batch_size):
        shape = (batch_size, -1, self.num_heads, self.projection_dim)
        x = tf.reshape(x, shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _combine_heads(self, x, batch_size):
        shape = (batch_size, -1, self.embed_dim)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, shape)

for r in range(100):
    

# ✅ Transformer Encoder Block
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs):
        attention_output = self.attention(inputs)
        x = self.norm1(inputs + attention_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return self.dropout(x)

# ✅ Positional Embeddings
class PositionEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        return inputs + self.position_embedding(positions)

# ✅ Build Vision Transformer Model
def build_vit_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Convert images into patches
    patches = layers.Conv2D(PROJECTION_DIM, kernel_size=PATCH_SIZE, strides=PATCH_SIZE)(inputs)
    patches = layers.Reshape((-1, PROJECTION_DIM))(patches)

    # Position embeddings
    encoded_patches = PositionEmbedding((IMG_SIZE // PATCH_SIZE) ** 2, PROJECTION_DIM)(patches)

    # Transformer Blocks
    for _ in range(4):
        encoded_patches = TransformerEncoder(PROJECTION_DIM, NUM_HEADS, 256)(encoded_patches)

    # MLP Head
    representation = layers.LayerNormalization()(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dense(128, activation="relu")(representation)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(representation)

    return keras.Model(inputs=inputs, outputs=outputs, name="VisionTransformer")

# ✅ Compile the ViT Model
vit_model = build_vit_model()
vit_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ Summary of the ViT Model
vit_model.summary()

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
import tensorflow as tf

# ✅ Load the datasets
df1 = pd.read_csv('path_to_speed_vs_time.csv')  # Speed vs. Time dataset
df2 = pd.read_csv('path_to_event_metadata.csv')  # Metadata (RecordingId, categories)

# ✅ Convert time columns to datetime format
df1['_time'] = pd.to_datetime(df1['_time'])
df2['StartTime'] = pd.to_datetime(df2['StartTime'])
df2['EndTime'] = pd.to_datetime(df2['EndTime'])

# ✅ Function to generate speed vs time plot
def generate_speed_plot(recording_id, save_dir='plots', dpi=32):
    """
    Generates and saves a speed vs time plot for a given recording ID as a 64x64 image.
    """
    try:
        event_data = df2[df2['RecordingId'] == recording_id]
        if event_data.empty:
            raise ValueError(f"Recording ID {recording_id} not found in metadata.")

        # Extract relevant information
        event_data = event_data.iloc[0]
        serial = event_data['Serial']
        start_time = event_data['StartTime']
        end_time = event_data['EndTime']
        is_traffic_stop = event_data['TrafficStop']  # 1 = Traffic Stop, 0 = Non-Traffic Stop

        # Filter the speed data
        speed_data = df1[(df1['host'] == serial) & 
                         (df1['_time'] >= start_time) & 
                         (df1['_time'] <= end_time)]
        if speed_data.empty:
            raise ValueError(f"No speed data available for Recording ID {recording_id}.")

        # ✅ Create 64x64 pixel plot
        plt.figure(figsize=(2, 2))
        plt.plot(speed_data['_time'], speed_data['Speed'], color='blue', linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('Speed')
        plt.title('Speed vs. Time')

        # ✅ Format x-axis
        plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))

        # ✅ Ensure save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # ✅ Assign label & filename
        label = 'Traffic_Stop' if is_traffic_stop == 1 else 'Non_Traffic_Stop'
        filename = f'{recording_id}_{label}.png'
        image_path = os.path.join(save_dir, filename)

        # ✅ Save the image
        plt.savefig(image_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved plot for {recording_id} as {image_path}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# ✅ Generate multiple plots for training dataset
recording_ids = df2['RecordingId'].unique()
for recording_id in recording_ids[:500]:  # Limit to first 500 for efficiency
    generate_speed_plot(recording_id)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory

# ✅ Define Image Processing Parameters
IMG_SIZE = 64
BATCH_SIZE = 32

# ✅ Load Training Data
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    "plots",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    "plots",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation'
)

# ✅ Train the Vision Transformer Model
history = vit_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20
)

# ✅ Evaluate Performance
train_loss, train_accuracy = vit_model.evaluate(train_generator)
val_loss, val_accuracy = vit_model.evaluate(validation_generator)

print(f"🔥 Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"🔥 Validation Accuracy: {val_accuracy * 100:.2f}%")

# ✅ Plot Training Performance
plt.figure(figsize=(12, 5))

# ✅ Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')

# ✅ Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')

plt.show()