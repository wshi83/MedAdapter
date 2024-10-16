import openai

def api_key_list(api_group):
    if api_group == '<API_GROUP>':
        api_key_list = [
            {
                "api_key": , # <OPENAI_API_KEY>,
                "api_version": ,# <OPENAI_API_VERSION>,
                "azure_endpoint": , # <AZURE_ENDPOINT>
                "model": # <MODEL_NAME>
            },
            {
                "api_key": , # <OPENAI_API_KEY>,
                "api_version": ,# <OPENAI_API_VERSION>,
                "azure_endpoint": , # <AZURE_ENDPOINT>
                "model": # <MODEL_NAME>
            },
        ]
    return api_key_list