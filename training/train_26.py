import tensorflow as tf
import tensorflow_datasets as tfds
import os, math

BATCH = 256
EPOCHS = 25
IMG = 28

# EMNIST/byclass labels:
# 0..9   -> digits
# 10..35 -> A..Z
# 36..61 -> a..z
LETTERS_START = 10
LETTERS_END_EXCL = 62  # 10..61 are letters

def is_letter(image, label):
    return tf.logical_and(label >= LETTERS_START, label < LETTERS_END_EXCL)

def remap_letter_26(image, label):
    # normalize image
    image = tf.cast(image, tf.float32) / 255.0

    # ensure channel dim
    if tf.rank(image) == 2:
        image = tf.expand_dims(image, -1)

    # static shape
    image = tf.ensure_shape(image, (28, 28, 1))

    # map 10..61 -> 0..51
    k = label - LETTERS_START

    # collapse case:
    # 0..25 (A..Z) stays same
    # 26..51 (a..z) becomes 0..25
    label_26 = tf.where(k < 26, k, k - 26)

    return image, label_26

# Augmentation (same as before)
augment = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomTranslation(0.12, 0.12),
    tf.keras.layers.RandomZoom(0.10),
], name="augment")

def with_aug_batched(images, labels):
    images = augment(images, training=True)
    images = tf.ensure_shape(images, (None, 28, 28, 1))
    return images, labels

# Load dataset
(raw_train, raw_test), info = tfds.load(
    "emnist/byclass",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

# Base datasets (filtered + remapped to 26 classes)
ds_train_base = raw_train.filter(is_letter).map(remap_letter_26, num_parallel_calls=tf.data.AUTOTUNE)
ds_test_base  = raw_test.filter(is_letter).map(remap_letter_26, num_parallel_calls=tf.data.AUTOTUNE)

# Count samples for steps (robust)
train_count = ds_train_base.reduce(tf.constant(0, dtype=tf.int64), lambda c, _: c + 1).numpy()
test_count  = ds_test_base.reduce(tf.constant(0, dtype=tf.int64), lambda c, _: c + 1).numpy()

steps_per_epoch = math.ceil(train_count / BATCH)
val_steps = math.ceil(test_count / BATCH)

print("train_count:", train_count, "steps_per_epoch:", steps_per_epoch)
print("test_count:", test_count, "val_steps:", val_steps)

# Final pipelines
ds_train = (
    ds_train_base
    .shuffle(20000)
    .batch(BATCH)
    .map(with_aug_batched, num_parallel_calls=tf.data.AUTOTUNE)
    .repeat()
    .prefetch(tf.data.AUTOTUNE)
)

ds_test = (
    ds_test_base
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

# Sanity check
for xb, yb in ds_test.take(1):
    print("xb:", xb.shape, "yb:", yb.shape)
    print("labels range:", int(tf.reduce_min(yb)), int(tf.reduce_max(yb)))

# Model (same CNN, output is 26 now)
inputs = tf.keras.Input(shape=(IMG, IMG, 1))
x = tf.keras.layers.Conv2D(32, 3, padding="same")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.35)(x)
outputs = tf.keras.layers.Dense(26, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

os.makedirs("checkpoints26", exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-5),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints26/best_letters26.keras",
        monitor="val_accuracy",
        save_best_only=True
    ),
]

history = model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=EPOCHS,
    callbacks=callbacks,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps
)

# Load best model
best_model = tf.keras.models.load_model("checkpoints26/best_letters26.keras")

# Evaluate
loss, acc = best_model.evaluate(ds_test)
print("TEST accuracy (26 classes):", acc)

# Export for TFJS conversion
best_model.export("saved_model_letters26")
print("Exported SavedModel to saved_model_letters26")