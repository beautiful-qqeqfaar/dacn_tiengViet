from cnn import build_cnn
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics

IMG_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

# Load MNIST
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

# Chuẩn hóa
train_x = train_x.astype("float32") / 255.0
test_x  = test_x.astype("float32") / 255.0

# Thêm channel dimension
train_x = tf.expand_dims(train_x, -1)  # (N,28,28,1)
test_x  = tf.expand_dims(test_x, -1)

# Resize về 32x32
train_x = tf.image.resize(train_x, (IMG_SIZE, IMG_SIZE))
test_x  = tf.image.resize(test_x, (IMG_SIZE, IMG_SIZE))

# Build model
model = build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=10)

model.compile(
    optimizer=optimizers.Adam(learning_rate=LR),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=[metrics.SparseCategoricalAccuracy(name="acc")]
)

# Train
model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1
)

# Evaluate
test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Save model
model.save("modells/cnn_baseline.keras")
print("Model saved to modells/cnn_baseline.keras")

