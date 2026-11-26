import json
from app import agent_graph

# 第一课中的本地演示用户。
# 生产系统不能由客户端自行构造身份。
demo_user = {
    "id": "alice",
    "tenant": "a",
    "groups": ["employee"],
    "role": "employee",
}

initial_state = {
    "message": "VPN 认证失败应该怎么处理？",
    "operation": "draft_ticket",
    "user": demo_user,
}

print("=== OpsPilot Workflow Debug ===")

# stream_mode="updates" 观察每个节点返回的更新
for event in agent_graph.stream(
    initial_state,
    stream_mode="updates",
):
    for node_name, updates in event.items():
        print(f"\n[Node] {node_name}")

        # 避免将原始内部知识内容写入调试日志
        safe_updates = {}

        for key, value in updates.items():
            if key == "docs":
                safe_updates["doc_ids"] = [
                    doc["id"] for doc in value
                ]
            elif key == "answer":
                safe_updates["answer_length"] = len(
                    value
                )
            else:
                safe_updates[key] = value

        print(
            json.dumps(
                safe_updates,
                ensure_ascii=False,
                indent=2,
            )
        )

print("\nWorkflow finished.")