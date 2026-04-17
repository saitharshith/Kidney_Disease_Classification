import tensorflow as tf
import pandas as pd
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from Kidney_Disease_classification.utils.common import save_json
from Kidney_Disease_classification.entity.config_entity import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _test_generator(self):
        """Creates a test generator using the exactly saved test split from the training phase."""
        # 1. Load the exact test set CSV saved during training
        test_csv_path = Path("artifacts/training/test_data.csv")
        test_df = pd.read_csv(test_csv_path)
        # 2. Generator settings (No augmentation for test data)
        datagenerator_kwargs = dict(rescale=1./255)
        dataflow_kwargs = dict(
            x_col='image',
            y_col='label',
            target_size=self.config.params_image_size[:-1], 
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            interpolation="bilinear"
        )

        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # 3. Create the generator directly from the dataframe
        self.test_generator = test_datagenerator.flow_from_dataframe(
            dataframe=test_df,
            shuffle=False, 
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        """Evaluates the model on the unseen test set and saves local scores."""
        self.model = self.load_model(self.config.path_of_model)
        self._test_generator()
        self.score = self.model.evaluate(self.test_generator)
        save_json(path=Path("scores.json"), data={"loss": self.score[0], "accuracy": self.score[1]})

    def log_into_mlflow(self):
        """Pushes parameters, metrics, and registers the model to the remote MLflow server."""
        
        # 1. Set BOTH Tracking and Registry URIs for DagsHub compatibility
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log all parameters from params.yaml
            mlflow.log_params(self.config.all_params)
            # Log calculated metrics (Loss & Accuracy)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # 2. Model Registry Logic
            # Model registry does not work with a local file store
            if tracking_url_type_store != "file":
                # Register the model to DagsHub Model Registry
                mlflow.keras.log_model(
                    self.model, 
                    "model", 
                    registered_model_name="VGG16Model"
                )
            else:
                mlflow.keras.log_model(self.model, "model")