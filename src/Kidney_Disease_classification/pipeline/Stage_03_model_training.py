from Kidney_Disease_classification.config.configuration import CofigurationManager
from Kidney_Disease_classification.components.Model_training import Training
from Kidney_Disease_classification import logger


STAGE_NAME = "Training"
class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = CofigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_test_generators() 
        training.train()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
