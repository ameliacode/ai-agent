import os

import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

from google import genai

# Replace with a NEW key (the old one is now public/insecure!)
client = genai.Client(api_key="AIzaSyC0LFL-F4wlopzZJuB20jYAmpJLW1JFC9o")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Hi! Are you working?"
)

print(response.text)
