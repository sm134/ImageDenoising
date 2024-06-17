import os
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import requests
import zipfile
import io

import keras
from keras import layers

import tensorflow as tf

# Download and unzip dataset
def download_and_extract(url, extract_to='.'):
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_to)

download_and_extract('https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip')

IMAGE_SIZE = 256
BATCH_SIZE = 16
MAX_TRAIN_IMAGES = 400

def load_data_pair(low_image_path, high_image_path):
    low_image = tf.io.read_file(low_image_path)
    low_image = tf.image.decode_png(low_image, channels=3)
    low_image = tf.image.resize(low_image, [IMAGE_SIZE, IMAGE_SIZE])
    low_image = low_image / 255.0

    high_image = tf.io.read_file(high_image_path)
    high_image = tf.image.decode_png(high_image, channels=3)
    high_image = tf.image.resize(high_image, [IMAGE_SIZE, IMAGE_SIZE])
    high_image = high_image / 255.0

    return low_image, high_image

def data_generator_pair(low_light_images, high_light_images):
    def load_data_pair_wrapper(low_image_path, high_image_path):
        low_image, high_image = load_data_pair(low_image_path, high_image_path)
        return low_image, high_image

    dataset = tf.data.Dataset.from_tensor_slices((low_light_images, high_light_images))
    dataset = dataset.map(load_data_pair_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


train_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))[:MAX_TRAIN_IMAGES]
val_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))[MAX_TRAIN_IMAGES:]
test_low_light_images = sorted(glob("./lol_dataset/eval15/low/*"))
train_high_light_images = sorted(glob("./lol_dataset/our485/high/*"))[:MAX_TRAIN_IMAGES]
val_high_light_images = sorted(glob("./lol_dataset/our485/high/*"))[MAX_TRAIN_IMAGES:]


train_dataset = data_generator_pair(train_low_light_images, train_high_light_images)
val_dataset = data_generator_pair(val_low_light_images, val_high_light_images)

print("Train Dataset:", train_dataset)
print("Validation Dataset:", val_dataset)

def build_dce_net(input_shape=(None, None, 3)):
    input_img = keras.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(input_img)
    conv2 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv1)
    conv3 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv2)
    conv4 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv3)
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(int_con1)
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(int_con2)
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same")(int_con3)
    return keras.Model(inputs=input_img, outputs=x_r)

def color_constancy_loss(x):
    mean_rgb = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
    mean_R = tf.reduce_mean(mean_rgb[:, :, :, 0])
    mean_G = tf.reduce_mean(mean_rgb[:, :, :, 1])
    mean_B = tf.reduce_mean(mean_rgb[:, :, :, 2])

    loss_RG = tf.square(mean_R - mean_G)
    loss_RB = tf.square(mean_R - mean_B)
    loss_GB = tf.square(mean_G - mean_B)

    total_loss = loss_RG + loss_RB + loss_GB
    return total_loss

def exposure_loss(x, mean_val=0.6):
    x = tf.reduce_mean(x, axis=3, keepdims=True)
    mean = tf.nn.avg_pool2d(x, ksize=16, strides=16, padding="VALID")
    return tf.reduce_mean(tf.square(mean - mean_val))

def illumination_smoothness_loss(x):
    sobel_edges = tf.image.sobel_edges(x)
    gradient_x = sobel_edges[:, :, :, :, 0]
    gradient_y = sobel_edges[:, :, :, :, 1]
    smoothness_loss = tf.reduce_mean(tf.square(tf.abs(gradient_x) + tf.abs(gradient_y)))
    return smoothness_loss

def spatial_constancy_loss(original_img, enhanced_img, patch_size):
    def gradient(img):
        img = tf.ensure_shape(img, [None, None, None, 3])
        img = tf.image.resize(img, [patch_size, patch_size])
        gradient_y, gradient_x = tf.image.image_gradients(img)
        return gradient_y, gradient_x

    original_gradient_y, original_gradient_x = gradient(original_img)
    enhanced_gradient_y, enhanced_gradient_x = gradient(enhanced_img)

    loss = tf.reduce_mean(tf.square(original_gradient_y - enhanced_gradient_y)) + \
           tf.reduce_mean(tf.square(original_gradient_x - enhanced_gradient_x))
    return loss
  
class ZeroDCE(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dce_model = build_dce_net()

    def compile(self, learning_rate, **kwargs):
        super().compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.illumination_smoothness_loss_tracker = keras.metrics.Mean(
            name="illumination_smoothness_loss"
        )
        self.spatial_constancy_loss_tracker = keras.metrics.Mean(
            name="spatial_constancy_loss"
        )
        self.color_constancy_loss_tracker = keras.metrics.Mean(
            name="color_constancy_loss"
        )
        self.exposure_loss_tracker = keras.metrics.Mean(name="exposure_loss")
        self.psnr_tracker = keras.metrics.Mean(name="psnr")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.illumination_smoothness_loss_tracker,
            self.spatial_constancy_loss_tracker,
            self.color_constancy_loss_tracker,
            self.exposure_loss_tracker,
            self.psnr_tracker,
        ]

    def get_enhanced_image(self, data, output):
        r1 = output[:, :, :, :3]
        r2 = output[:, :, :, 3:6]
        r3 = output[:, :, :, 6:9]
        r4 = output[:, :, :, 9:12]
        r5 = output[:, :, :, 12:15]
        r6 = output[:, :, :, 15:18]
        r7 = output[:, :, :, 18:21]
        r8 = output[:, :, :, 21:24]
        x = data + r1 * (tf.square(data) - data)
        x = x + r2 * (tf.square(x) - x)
        x = x + r3 * (tf.square(x) - x)
        enhanced_image = x + r4 * (tf.square(x) - x)
        x = enhanced_image + r5 * (tf.square(enhanced_image) - enhanced_image)
        x = x + r6 * (tf.square(x) - x)
        x = x + r7 * (tf.square(x) - x)
        enhanced_image = x + r8 * (tf.square(x) - x)
        return enhanced_image

    def call(self, data):
        dce_net_output = self.dce_model(data)
        return self.get_enhanced_image(data, dce_net_output)

    def compute_losses(self, data, output):
        enhanced_image = self.get_enhanced_image(data, output)
        loss_illumination = 200 * illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(spatial_constancy_loss(data, enhanced_image, patch_size=32))
        loss_color_constancy = 5 * tf.reduce_mean(color_constancy_loss(enhanced_image))
        loss_exposure = 10 * tf.reduce_mean(exposure_loss(enhanced_image))
        total_loss = (
            loss_illumination
            + loss_spatial_constancy
            + loss_color_constancy
            + loss_exposure
        )

        return {
            "total_loss": total_loss,
            "illumination_smoothness_loss": loss_illumination,
            "spatial_constancy_loss": loss_spatial_constancy,
            "color_constancy_loss": loss_color_constancy,
            "exposure_loss": loss_exposure,
        }

    def psnr(self, y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=255.0)

    def train_step(self, data):
        low_light_images, high_light_images = data
        with tf.GradientTape() as tape:
            output = self.dce_model(low_light_images)
            losses = self.compute_losses(low_light_images, output)
            enhanced_image = self.get_enhanced_image(low_light_images, output)

        gradients = tape.gradient(
            losses["total_loss"], self.dce_model.trainable_weights
        )
        self.optimizer.apply_gradients(zip(gradients, self.dce_model.trainable_weights))

        self.total_loss_tracker.update_state(losses["total_loss"])
        self.illumination_smoothness_loss_tracker.update_state(
            losses["illumination_smoothness_loss"]
        )
        self.spatial_constancy_loss_tracker.update_state(
            losses["spatial_constancy_loss"]
        )
        self.color_constancy_loss_tracker.update_state(losses["color_constancy_loss"])
        self.exposure_loss_tracker.update_state(losses["exposure_loss"])
        self.psnr_tracker.update_state(self.psnr(high_light_images, enhanced_image))

        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        low_light_images, high_light_images = data
        output = self.dce_model(low_light_images)
        losses = self.compute_losses(low_light_images, output)
        enhanced_image = self.get_enhanced_image(low_light_images, output)

        self.total_loss_tracker.update_state(losses["total_loss"])
        self.illumination_smoothness_loss_tracker.update_state(
            losses["illumination_smoothness_loss"]
        )
        self.spatial_constancy_loss_tracker.update_state(
            losses["spatial_constancy_loss"]
        )
        self.color_constancy_loss_tracker.update_state(losses["color_constancy_loss"])
        self.exposure_loss_tracker.update_state(losses["exposure_loss"])
        self.psnr_tracker.update_state(self.psnr(high_light_images, enhanced_image))

        return {metric.name: metric.result() for metric in self.metrics}

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """While saving the weights, we simply save the weights of the DCE-Net"""
        self.dce_model.save_weights(
            filepath,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        """While loading the weights, we simply load the weights of the DCE-Net"""
        self.dce_model.load_weights(
            filepath=filepath,
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )

zero_dce_model = ZeroDCE()
zero_dce_model.compile(learning_rate=1e-4)
history = zero_dce_model.fit(train_dataset, validation_data=val_dataset, epochs=100)


def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_result("total_loss")
plot_result("illumination_smoothness_loss")
plot_result("spatial_constancy_loss")
plot_result("color_constancy_loss")
plot_result("exposure_loss")

def plot_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()

def infer(original_image):
    image = keras.utils.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = zero_dce_model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    return output_image


base_directory = os.getcwd()
print("Base Directory:", base_directory)

# Define the paths relative to the base directory
test_directory = os.path.join(base_directory, "test/low")
predicted_directory = os.path.join(base_directory, "test/predicted")
original_directory = os.path.join(base_directory, "test/original")

for val_image_file in os.listdir(test_directory):
    val_image_path = os.path.join(test_directory, val_image_file)
    if os.path.isfile(val_image_path):
        original_image = Image.open(val_image_path)
        enhanced_image = infer(original_image)
        base_name = os.path.basename(val_image_file)
        save_path = os.path.join(predicted_directory, base_name)
        enhanced_image.save(save_path)
        plot_results(
            [original_image, enhanced_image],
            ["Original", "Enhanced"],
            (20, 12),
        )
