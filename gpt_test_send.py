from utils.gpt_feedback import send_report_to_chatgpt

response = send_report_to_chatgpt(
    symbol="DOGEUSDT",
    tags=["listing", "DEX inflow"],
    score=82.5,
    compressed=True,
    stage1g=True
)

print("GPT Response:\n", response)