
from argparse import Namespace

def default_model_type(args: Namespace) -> str:
    if args.model_type is not None:
        return args.model_type

    match args.model_id:
        case 'meta-llama/Llama-2-70b-chat-hf' | 'meta-llama/Llama-2-13b-chat-hf' | 'meta-llama/Llama-2-7b-chat-hf':
            return 'Llama2'
        case 'tiiuae/falcon-40b-instruct' | 'tiiuae/falcon-7b-instruct':
            return 'Falcon'
        case 'mistralai/Mistral-7B-Instruct-v0.2' | 'mistralai/Mistral-7B-Instruct-v0.1':
            return 'Mistral'
        case _:
            raise ValueError(f'unknown model-id {args.model_id}')

def default_model_id(args: Namespace) -> str:
    if args.model_id is not None:
        return args.model_id

    match args.model_name:
        case 'llama2-70b':
            return 'meta-llama/Llama-2-70b-chat-hf'
        case 'llama2-13b':
            return 'meta-llama/Llama-2-13b-chat-hf'
        case 'llama2-7b':
            return  'meta-llama/Llama-2-7b-chat-hf'
        case 'falcon-40b':
            return 'tiiuae/falcon-40b-instruct'
        case 'falcon-7b':
            return 'tiiuae/falcon-7b-instruct'
        case 'mistral-v2-7b':
            return 'mistralai/Mistral-7B-Instruct-v0.2'
        case 'mistral-v1-7b':
            return 'mistralai/Mistral-7B-Instruct-v0.1'
        case _:
            raise ValueError(f'unknown model-name {args.model_name}')
