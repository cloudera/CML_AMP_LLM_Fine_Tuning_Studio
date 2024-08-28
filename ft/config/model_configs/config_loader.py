from transformers import AutoConfig


class ModelMetadataFinder:
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path

    def fetch_model_family_from_config(self):
        config = AutoConfig.from_pretrained(self.model_name_or_path)
        print(f"The config for model {self.model_name_or_path} is \n {config} \n")
        try:
            model_family = config.architectures[0]
            return model_family
        except IndexError:
            print("Model family not found in the configuration")
            return None
    
    @staticmethod
    def fetch_bos_token_id_from_config(model_name_or_path):
        config = AutoConfig.from_pretrained(model_name_or_path)
        try:
            return config.bos_token_id
        except:
            print("Bos token Id can't be found out returning default alpaca ID")
            return 1
    


if __name__ == "__main__":
    model_family_finder = ModelMetadataFinder("nvidia/Mistral-NeMo-Minitron-8B-Base")
    model_family = model_family_finder.fetch_model_family_from_config()
    bos_token_id = model_family_finder.fetch_bos_token_id_from_config("nvidia/Mistral-NeMo-Minitron-8B-Base")
    print(f"Model family: {model_family}")
    print(f"The bos id for model is {bos_token_id}")

