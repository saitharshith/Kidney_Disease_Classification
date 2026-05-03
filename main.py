import os
from Kidney_Disease_classification import logger
from Kidney_Disease_classification.pipeline.Stage_01_data_ingestion import DataIngestionTrainingPipeline
from Kidney_Disease_classification.pipeline.Stage_02_prepare_base_model import PrepareBaseModelPipeline
from Kidney_Disease_classification.pipeline.Stage_03_model_training import ModelTrainingPipeline
from Kidney_Disease_classification.pipeline.Stage_04_model_evaluation import ModelEvaluationPipeline

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Sai_Tharshith_97/Kidney_Disease_Classification.mlflow"

if __name__ == '__main__':
    STAGE_NAME = "Data Ingestion Stage"
    try:
        logger.info(f">>> Stage {STAGE_NAME} Started <<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>> Stage {STAGE_NAME} Completed <<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e 

    STAGE_NAME = "Prepare Base Model"
    try:
        logger.info(f">>> Stage {STAGE_NAME} Started <<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>> Stage {STAGE_NAME} Completed <<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

    STAGE_NAME = "Training Pipeline Stage"
    try:
        logger.info(f">>> Stage {STAGE_NAME} Started <<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>> Stage {STAGE_NAME} Completed <<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

    STAGE_NAME = "Evaluation Stage"
    try:
        logger.info(f">>> Stage {STAGE_NAME} Started <<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>> Stage {STAGE_NAME} Completed <<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e