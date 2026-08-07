from __future__ import annotations

import json
import sys

from src.agents.extraction_agent import ExtractionAgent
from src.contracts.agent_result import ExecutionStatus
from tests.v16.agent_environment import ExtractionAgentEnvironment


def main() -> int:
    environment = ExtractionAgentEnvironment()
    try:
        result = ExtractionAgent(environment.dependencies).execute(
            environment.agent_input
        )
        payload = {
            "status": "OK" if result.execution_status is ExecutionStatus.COMPLETED else "FAILED",
            "execution_status": result.execution_status.value,
            "quality_status": result.quality_status.value,
            "attempt_number": result.attempt_number,
            "requested_transition": result.requested_transition.to_dict(),
            "failure_reason_codes": list(result.failure_reason_codes),
            "services": "CONTROLLED_DOUBLES",
            "direct_module_call": True,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["status"] == "OK" else 1
    finally:
        environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
