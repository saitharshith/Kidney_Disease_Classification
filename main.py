from Kidney_Disease_classification import logger
from Kidney_Disease_classification.pipeline.Stage_01_data_ingestion import DataIngestionTrainingPipeline
from Kidney_Disease_classification.pipeline.Stage_02_prepare_base_model import PrepareBaseModelPipeline
from Kidney_Disease_classification.pipeline.Stage_03_model_training import ModelTrainingPipeline



STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>Stage {STAGE_NAME} Started <<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>> Stage {STAGE_NAME} Completed<<<")
except Exception as e:
    logger.exception(e)
    raise e 

STAGE_NAME = "prepare base model"
try:
    logger.info(f">>> stage {STAGE_NAME} started <<<")
    obj = PrepareBaseModelPipeline()
    obj.main()
    logger.info(f">>> stage {STAGE_NAME} completed <<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training Pipeline Stage"

try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e