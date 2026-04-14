from Kidney_Disease_classification.config.configuration import CofigurationManager
from Kidney_Disease_classification.components.Prepare_Base_Model import PrepareBaseModel
from Kidney_Disease_classification import logger

STAGE_NAME = "prepare base model"

class PrepareBaseModelPipeline:
    def __init__(self):
        pass   
    
    def main(self):
        config = CofigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config = prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
        

if __name__ == "__main__":
    try:
        logger.info(f">>> stage {STAGE_NAME} started <<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>> stage {STAGE_NAME} completed <<<")
    except Exception as e:
        logger.exception(e)
        raise e