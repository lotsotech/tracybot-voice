"""
ClawdbotAgent - Bridge between Vocode and Clawdbot
Connects voice conversations to your existing Clawdbot instance via the CLI
"""

import asyncio
import json
import re
import subprocess
from typing import AsyncGenerator, Optional, Tuple

from vocode.streaming.agent.base_agent import RespondAgent, GeneratedResponse
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.message import BaseMessage


class ClawdbotAgentConfig(AgentConfig, type="agent_clawdbot"):  # type: ignore
    """Configuration for ClawdbotAgent"""
    agent_id: str = "main"
    clawdbot_timeout: int = 60
    session_id: Optional[str] = None


class ClawdbotAgent(RespondAgent[ClawdbotAgentConfig]):
    """
    Voice agent that forwards conversations to Clawdbot via CLI:
      clawdbot agent --agent main --message "..." --json
    Response text is in result.payloads[*].text
    """

    async def _call_clawdbot(self, message: str) -> str:
        """Run clawdbot CLI and return the text response"""
        cmd = [
            "clawdbot", "agent",
            "--agent", self.agent_config.agent_id,
            "--message", message,
            "--json",
        ]
        if self.agent_config.session_id:
            cmd += ["--session-id", self.agent_config.session_id]

        print(f"🤖 Sending to Clawdbot: {message}")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.agent_config.clawdbot_timeout
            )
        except asyncio.TimeoutError:
            return "Sorry, I'm thinking too slowly right now. Can you try again?"

        if proc.returncode != 0:
            err = stderr.decode().strip()
            print(f"❌ Clawdbot CLI error: {err}")
            return "Sorry, I'm having trouble connecting to my brain right now."

        try:
            data = json.loads(stdout.decode())
            payloads = data.get("result", {}).get("payloads", [])
            reply = " ".join(p.get("text", "") for p in payloads if p.get("text"))
            reply = reply.strip() or "(no response)"
        except Exception as e:
            print(f"❌ JSON parse error: {e}")
            return "Sorry, I received an unexpected response."

        print(f"📝 Clawdbot: {reply}")
        return reply

    async def respond(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[Optional[str], bool]:
        """Turn-based response (non-streaming)"""
        reply = await self._call_clawdbot(human_input)
        return reply, False

    async def generate_response(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
        bot_was_in_medias_res: bool = False,
    ) -> AsyncGenerator[GeneratedResponse, None]:
        """Streaming response — split reply into sentences for natural speech pacing"""
        if is_interrupt:
            print("🛑 Human interrupted — getting fresh response")

        reply = await self._call_clawdbot(human_input)
        sentences = [s.strip() for s in re.split(r"[.!?]+", reply) if s.strip()]
        if not sentences:
            sentences = [reply]

        for i, sentence in enumerate(sentences):
            is_interruptible = i < len(sentences) - 1
            yield GeneratedResponse(
                message=BaseMessage(text=sentence),
                is_interruptible=is_interruptible,
            )
