from candle.models.llama import LlamaModel, Config


config = Config()
model = LlamaModel(config)
state_dict = model.state_dict()
print(state_dict.keys())
