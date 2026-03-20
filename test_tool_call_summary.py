#!/usr/bin/env python3
"""Test script to verify tool call handling in assistant messages."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from nanobot.agent.context import ContextBuilder
from nanobot.providers.base import LLMProvider

class MockProvider(LLMProvider):
    """Mock provider for testing."""
    pass

def test_tool_call_handling():
    """Test that tool calls are properly converted to human-readable summaries."""
    workspace = Path(__file__).parent / "test_workspace"
    workspace.mkdir(exist_ok=True)
    
    context = ContextBuilder(workspace=workspace, provider=None)
    messages = []
    
    # Test 1: Simple tool call with string arguments
    tool_calls = [{
        "function": {
            "name": "search_web",
            "arguments": json.dumps({"query": "test search"})
        },
        "id": "call_123",
        "type": "function"
    }]
    
    result = context.add_assistant_message(
        messages,
        None,
        tool_calls=tool_calls,
    )
    
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "tool_calls" not in result[0]  # tool_calls should be omitted
    assert "search_web" in result[0]["content"]
    print("✓ Test 1 passed: Simple tool call converted to summary")
    print(f"  Content: {result[0]['content']}")
    
    # Test 2: Tool call with large arguments
    large_args = {"content": "x" * 10000}
    tool_calls_large = [{
        "function": {
            "name": "process_content",
            "arguments": json.dumps(large_args)
        },
        "id": "call_456",
        "type": "function"
    }]
    
    messages2 = []
    result2 = context.add_assistant_message(
        messages2,
        None,
        tool_calls=tool_calls_large,
    )
    
    assert len(result2) == 1
    assert "tool_calls" not in result2[0]
    assert "process_content" in result2[0]["content"]
    print("✓ Test 2 passed: Large tool call arguments properly summarized")
    print(f"  Content: {result2[0]['content']}")
    
    # Test 3: Multiple tool calls
    multi_tool_calls = [
        {
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": "/test/file.txt"})
            },
            "id": "call_789",
            "type": "function"
        },
        {
            "function": {
                "name": "search_web",
                "arguments": json.dumps({"query": "another search"})
            },
            "id": "call_101",
            "type": "function"
        }
    ]
    
    messages3 = []
    result3 = context.add_assistant_message(
        messages3,
        None,
        tool_calls=multi_tool_calls,
    )
    
    assert len(result3) == 1
    assert "tool_calls" not in result3[0]
    assert "read_file" in result3[0]["content"]
    assert "search_web" in result3[0]["content"]
    print("✓ Test 3 passed: Multiple tool calls converted to summary")
    print(f"  Content: {result3[0]['content']}")
    
    # Test 4: Regular message without tool calls
    messages4 = []
    result4 = context.add_assistant_message(
        messages4,
        "This is a regular response",
        tool_calls=None,
    )
    
    assert len(result4) == 1
    assert result4[0]["content"] == "This is a regular response"
    assert "tool_calls" not in result4[0]
    print("✓ Test 4 passed: Regular message preserved")
    print(f"  Content: {result4[0]['content']}")
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_tool_call_handling()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
