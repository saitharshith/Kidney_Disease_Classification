import os
import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from Kidney_Disease_classification.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        """Loads the pre-trained base model."""
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def _prepare_dataframes(self):
        """Creates a DataFrame of image paths and splits it into Train, Val, and Test."""
        train_dir = Path(self.config.training_data)
        classes = ['Normal', 'Cyst', 'Stone', 'Tumor']
        data = []
        # Read images and assign labels
        for class_name in classes:
            class_dir = train_dir / class_name
            for img_path in class_dir.glob('*.jpg'):
                # flow_from_dataframe requires string labels for categorical classification
                data.append((str(img_path), class_name))

        df = pd.DataFrame(data, columns=['image', 'label'])
        # Shuffle the full dataset
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
        return train_df, val_df, test_df

    def train_valid_test_generators(self):
        """Initializes the data generators with resizing, augmentation, and splitting."""
        
        train_df, val_df, test_df = self._prepare_dataframes()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        datagenerator_kwargs = dict(rescale=1./255)
        dataflow_kwargs = dict(
            x_col='image',
            y_col='label',
            target_size=self.config.params_image_size[:-1], 
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            interpolation="bilinear"
        )

        valid_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_test_datagen.flow_from_dataframe(
            dataframe=self.val_df,
            shuffle=False, 
            **dataflow_kwargs
        )
        self.test_generator = valid_test_datagen.flow_from_dataframe(
            dataframe=self.test_df,
            shuffle=False, 
            **dataflow_kwargs
        )
        if self.config.params_is_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagen = valid_test_datagen

        self.train_generator = train_datagen.flow_from_dataframe(
            dataframe=self.train_df,
            shuffle=True,
            **dataflow_kwargs
        )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        
    def train(self):
        self.steps_per_epoch = len(self.train_df) // self.config.params_batch_size
        self.validation_steps = len(self.val_df) // self.config.params_batch_size
        fresh_optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate) 
        self.model.compile(
            optimizer=fresh_optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
