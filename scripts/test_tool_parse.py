import re
from typing import List

def run_tool(assistant_output: str) -> List[tuple[str, str]]:
    import json

    outputs = []

    try:
        matches = assistant_output.split("TOOL_CALL:")[1:]
        for match in matches:
            print("current match: ", match, "\n\n")
            end = match.rfind('}')
            match = match[:end+1] if end != -1 else None
            # Attempt to parse the JSON payload
            spec = json.loads(match.strip())
            name = spec["name"]
            payload = spec.get("payload", {})
            print(f"Processing tool call: {name} with payload: {payload}")
    except Exception as e:
        print("Error processing tool call:", e)

    return outputs

if __name__ == "__main__":
    # Example usage
    assistant_output = '''
    Here is some response text.
    TOOL_CALL: {
      "name": "example_tool",
      "payload": {
        "key": "value"
      }
    } Some unrelated text TOOL_CALL: {"name": "another_tool", "payload": {}}
    '''

    run_tool(assistant_output)
