"""
Custom ACE Generator that wraps the CIExMAS pipeline.

Instead of a single LLM call, the "generation" runs the full pipeline:
Entity Extraction → Triple Extraction → URI Retrieval → Turtle Generation.

The playbook is injected as the `instruction` field into the pipeline state,
influencing all stages of extraction.
"""

import re
import json
from typing import Tuple, List, Dict, Any, Optional

from approaches.Pipeline_ACE.pipeline import invoke_pipeline


class PipelineGenerator:
    """
    ACE-compatible Generator that runs the CIExMAS extraction pipeline.

    Implements the same interface as ace.core.Generator so it can be used
    as a drop-in replacement in the ACE training loop.
    """

    def __init__(self, model: str = "pipeline"):
        self.model = model

    def generate(
        self,
        question: str,
        playbook: str,
        context: str = "",
        reflection: str = "(empty)",
        use_json_mode: bool = False,
        call_id: str = "gen",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Run the full CIExMAS pipeline and return the Turtle output as "answer".

        Args:
            question: Task instruction (ignored, the pipeline knows its task)
            playbook: Current ACE playbook - injected as pipeline instruction
            context: Source text to extract from
            reflection: Previous reflection (prepended to instruction if non-empty)
            use_json_mode: Unused (pipeline has its own structured output)
            call_id: Identifier for logging
            log_dir: Unused (pipeline logs via Langfuse)

        Returns:
            Tuple of (turtle_output, bullet_ids_used, call_info)
        """
        instruction = self._build_instruction(playbook, reflection)

        try:
            result = invoke_pipeline(text=context, instruction=instruction, trace=True)
            turtle_output = result.get("turtle", "")
        except Exception as e:
            turtle_output = ""
            result = {"error": str(e)}

        bullet_ids = self._extract_bullet_ids(playbook)

        call_info = {
            "role": "generator",
            "call_id": call_id,
            "model": self.model,
            "prompt_length": len(context),
            "response_length": len(turtle_output),
            "entities": list(result.get("entities", set())),
            "triples": list(result.get("triples", set())),
        }

        return turtle_output, bullet_ids, call_info

    def _build_instruction(self, playbook: str, reflection: str) -> str:
        """Combine playbook and reflection into a single instruction string."""
        parts = []

        if playbook and playbook.strip():
            parts.append(playbook.strip())

        if reflection and reflection.strip() != "(empty)":
            parts.append(f"\n\n## REFLECTION FROM PREVIOUS ATTEMPT\n{reflection.strip()}")

        return "\n\n".join(parts) if parts else ""

    def _extract_bullet_ids(self, playbook: str) -> List[str]:
        """Extract bullet IDs from playbook for tracking."""
        pattern = r'\[([a-z]{3,}-\d{5})\]'
        return re.findall(pattern, playbook)
