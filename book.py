from openai import OpenAI

# Replace 'your-openrouter-api-key' with your actual OpenRouter key
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=""
)

try:
    response = client.chat.completions.create(
        model="gpt-oss-120b",  # free model
        messages=[{"role": "user", "content": "Say the current year and say a happy message since today is new year."}]
    )
    print("✅ Success!")
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print("❌ Error:", e)