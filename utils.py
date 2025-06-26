import sys
import importlib.util
#from transformers import AutoModelForCausalLM, AutoTokenizer 
"""
def load_pretrained(model_name_or_path, **kwargs):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **kwargs
    ).eval()

    return model, tokenizer"""


def load_module_functions(module_path, module_name, functions):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return (
        getattr(module, fn, None) 
        for fn in functions
    )