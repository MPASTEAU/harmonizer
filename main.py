from config import config

from openai_model import OpenAIModel


def main():
    model = OpenAIModel(config.OPENAI_API_KEY, config.OPENAI_MODEL_NAME)
    model.add_message(role="user", content="Bonjour, comment vas-tu ?")
    response = model.chat()
    print(response)


if __name__ == "__main__":
    main()