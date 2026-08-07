from pathlib import Path
import json, tempfile
from tests.v16.test_quantitative_capability_v16 import make_input, capability

def main():
    with tempfile.TemporaryDirectory() as tmp:
        result=capability().execute(make_input(Path(tmp)))
        report={
            "status":"OK" if result.execution_status.value=="COMPLETED" else "FAILED",
            "controlled_doubles":True,
            "execution_status":result.execution_status.value,
            "quality_status":result.quality_status.value,
            "transition":result.requested_transition.to_dict(),
            "artifact_count":len(result.output_artifacts),
            "artifact_names":sorted(result.output_artifacts),
            "metrics":result.quality_metrics,
        }
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0 if report["status"]=="OK" and report["artifact_count"]==9 else 1
if __name__=="__main__": raise SystemExit(main())
