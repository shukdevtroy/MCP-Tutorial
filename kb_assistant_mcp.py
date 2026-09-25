"""
Personal Knowledge Base Assistant — real MCP client + Gradio UI
------------------------------------------------------------------
This version is a genuine MCP client: it launches mcp_server.py as a
subprocess, connects over stdio using the official `mcp` Python SDK,
discovers the server's tools at runtime (no hardcoded tool schema), and
lets a free OpenRouter model call those tools to answer questions grounded
in your notes.

The OpenRouter API key is entered in the UI (not an env var) and kept only
in memory for the session — it's never written to disk.

Setup
-----
pip install mcp gradio requests scikit-learn

mkdir notes   # put your .md files here (same folder as this script)
python kb_assistant_mcp.py

Then open the local URL Gradio prints, paste your OpenRouter key into the
"API Key" box, and start chatting.

Model note
----------
Not every free OpenRouter model supports tool-calling reliably. Check
https://openrouter.ai/models?supported_parameters=tools for the current
free-tier lineup and change MODEL_NAME below if needed.
"""

import os
import json
import asyncio
import threading
from contextlib import AsyncExitStack

import requests
import gradio as gr

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = os.environ.get("KB_MODEL", "stealth/union-alpha")
MAX_TOOL_ITERATIONS = 5

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "user's personal notes. Always use the search_notes tool first to find "
    "relevant notes before answering. If you need more detail from a "
    "specific note, use read_note. If the notes don't contain the answer, "
    "say so honestly instead of making things up. Cite the note filename "
    "when you use information from it."
)

# --------------------------------------------------------------------------
# Background event loop that owns the persistent MCP connection.
# Gradio's callbacks are sync, so we run one asyncio loop in a daemon
# thread for the lifetime of the app and hand it coroutines to run.
# --------------------------------------------------------------------------

_loop = asyncio.new_event_loop()


def _run_loop():
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


threading.Thread(target=_run_loop, daemon=True).start()

_session: ClientSession | None = None
_mcp_tools_schema = None
_exit_stack_holder = {}  # keeps the stdio/session context managers alive


def _mcp_tools_to_openai_schema(tools):
    """Convert MCP tool definitions (discovered from the server) into the
    OpenAI-style 'tools' schema OpenRouter expects."""
    schema = []
    for t in tools:
        schema.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        })
    return schema


async def _connect():
    global _session, _mcp_tools_schema

    # Close old connection if one exists
    try:
        old_stack = _exit_stack_holder.get("stack")
        if old_stack:
            await old_stack.aclose()
    except Exception:
        pass

    server_params = StdioServerParameters(
        command="python",
        args=[SERVER_SCRIPT]
    )

    stack = AsyncExitStack()

    read, write = await stack.enter_async_context(
        stdio_client(server_params)
    )

    session = await stack.enter_async_context(
        ClientSession(read, write)
    )

    await session.initialize()

    tools_result = await session.list_tools()

    _session = session
    _mcp_tools_schema = _mcp_tools_to_openai_schema(
        tools_result.tools
    )

    _exit_stack_holder["stack"] = stack

    print(
        f"[info] connected to MCP server, discovered tools: "
        f"{[t.name for t in tools_result.tools]}"
    )


def ensure_connected():
    global _session

    if _session is None:
        fut = asyncio.run_coroutine_threadsafe(
            _connect(),
            _loop
        )
        fut.result(timeout=30)


async def _call_tool_async(name, args):
    global _session

    if _session is None:
        raise RuntimeError("MCP session is not connected.")

    result = await _session.call_tool(name, args)

    texts = [
        c.text
        for c in result.content
        if hasattr(c, "text")
    ]

    return "\n".join(texts) if texts else str(result.content)


def call_mcp_tool(name, args):
    ensure_connected()

    try:
        fut = asyncio.run_coroutine_threadsafe(
            _call_tool_async(name, args),
            _loop
        )

        return fut.result(timeout=30)

    except Exception as e:
        print(f"[warn] MCP connection error: {e}")

        # Try reconnecting once
        try:
            fut = asyncio.run_coroutine_threadsafe(
                _connect(),
                _loop
            )
            fut.result(timeout=30)

            fut = asyncio.run_coroutine_threadsafe(
                _call_tool_async(name, args),
                _loop
            )

            return fut.result(timeout=30)

        except Exception as retry_error:
            raise RuntimeError(
                f"MCP tool failed after reconnect: {retry_error}"
            )


# --------------------------------------------------------------------------
# OpenRouter call
# --------------------------------------------------------------------------


def call_openrouter(messages, api_key):
    if not api_key:
        raise RuntimeError("Enter your OpenRouter API key in the box above first.")

    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": MODEL_NAME,
            "messages": messages,
            "tools": _mcp_tools_schema,
        }),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def run_tool_calling_loop(messages, api_key):
    ensure_connected()

    for _ in range(MAX_TOOL_ITERATIONS):
        data = call_openrouter(messages, api_key)

        if "error" in data:
            return f"OpenRouter error: {data['error']}"

        choice = data["choices"][0]
        msg = choice["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return msg.get("content", "(empty response)")

        for call in tool_calls:
            fn_name = call["function"]["name"]
            try:
                fn_args = json.loads(call["function"]["arguments"])
            except json.JSONDecodeError:
                fn_args = {}

            try:
                result_text = call_mcp_tool(fn_name, fn_args)
            except Exception as e:
                result_text = json.dumps({"error": f"MCP tool call failed: {e}"})

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": fn_name,
                "content": result_text,
            })

    return "Stopped after too many tool-calling steps — try rephrasing your question."


# --------------------------------------------------------------------------
# Gradio UI
# --------------------------------------------------------------------------


def chat_fn(message, history, api_key):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Gradio newer versions use OpenAI-style message dictionaries
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")

            # Only add valid text messages
            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({
                    "role": role,
                    "content": content
                })

        # Backward compatibility with older Gradio tuple format
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            user_msg, bot_msg = item[0], item[1]

            if user_msg:
                messages.append({
                    "role": "user",
                    "content": str(user_msg)
                })

            if bot_msg:
                messages.append({
                    "role": "assistant",
                    "content": str(bot_msg)
                })

    messages.append({
        "role": "user",
        "content": message
    })

    try:
        return run_tool_calling_loop(messages, api_key)
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    with gr.Blocks(title="Personal Knowledge Base Assistant (MCP)") as demo:
        gr.Markdown(
            "# 📚 Personal Knowledge Base Assistant\n"
            "Backed by a real MCP server (`mcp_server.py`) and a free OpenRouter model.\n\n"
            "Paste your OpenRouter API key below (kept in memory only, never saved to disk)."
        )
        api_key_box = gr.Textbox(
            label="OpenRouter API Key",
            type="password",
            placeholder="sk-or-...",
        )
        gr.ChatInterface(fn=chat_fn, additional_inputs=[api_key_box])

    demo.launch()
