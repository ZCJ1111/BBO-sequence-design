import importlib
import os

from .ensemble import Ensemble

model_collection = {}


def register_model(model_name):
    def register_func(model_class):
        model_collection[model_name] = model_class
        return model_class

    return register_func


def get_model(args, **kwargs):
    print(f'model_collection is {model_collection[args.net]}')
    print(f'model_collection_1 is {model_collection[args.net](args,**kwargs)}')
    
    # return Ensemble(
    #     [model_collection[args.net](args, **kwargs) for _ in range(args.ensemble_size)],
    #     args.ensemble_rule,
    # )
    return model_collection[args.net](args,**kwargs)

    # return args.net(args, **kwargs)


for file_name in os.listdir(os.path.dirname(__file__)):
    if file_name.endswith(".py") and not file_name.startswith("_"):
        importlib.import_module("model." + file_name[:-3])
