from intent_llm import classify_intent

examples = [
    "VPN 认证失败怎么办？",
    "VPN 试了很多次都不行，请 IT 帮我处理。",
    "我想申请管理员权限，应该找谁？",
]

for message in examples:
    result = classify_intent(message)

    print("用户:", message)
    print("建议:", result.action)
    print("原因:", result.reason)
    print("---")