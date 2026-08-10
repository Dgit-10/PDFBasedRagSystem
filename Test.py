from google import genai

for model in client.models.list():
    print(model.name)