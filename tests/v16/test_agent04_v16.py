import json,tempfile,unittest,hashlib
from pathlib import Path
import pandas as pd
from src.contracts.agent_input import AgentInput,AgentContext,ArtifactReference,ExecutionMode,PreviousAttemptSummary
from src.agents.thematic_analysis_agent import ThematicAnalysisAgent
from src.adapters.thematic_analysis_runtime import ThematicRuntimeDependencies,parse_json
from src.tools.thematic_analysis.schema_validation import normalize_thematic_output
from src.tools.thematic_analysis.reference_validation import validate_references
from src.tools.thematic_analysis.coverage_validation import calculate_diagnostic_metrics
from src.tools.thematic_analysis.repair_strategy import build_repair_plan
from src.runtime.thematic_analysis_protocol import execute_thematic_transaction,build_thematic_fingerprints
from src.state.state_store import StateStore
from src.state.pipeline_state import PipelineIdentity,PipelineState

def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class T(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory(); self.r=Path(self.t.name); self.o=self.r/'out';self.o.mkdir()
  self.kb=self.r/'kb.csv';pd.DataFrame([{'source_filename':'a.pdf','title':'Alpha','include_in_state_of_art':True,'relevance_level':'alta','methods_or_models':'ANN','limitations_or_gaps':'Need more data'},{'source_filename':'b.pdf','title':'Beta','include_in_state_of_art':True,'relevance_level':'alta','methods_or_models':'SVM','limitations_or_gaps':'Limited sites'}]).to_csv(self.kb,index=False)
  self.kbj=self.r/'kb.jsonl';self.kbj.write_text('{}\n')
  self.m=self.r/'m.json';self.m.write_text(json.dumps({'experiment_id':'e','run_id':'r','stage':'03_agente_extraccion_kb','safety_policy':{'uses_ground_truth':False}}))
  self.payload={'corpus_summary':{},'themes':[{'theme_id':'T1','theme_name':'Models','description':'ANN and SVM','representative_papers':[{'source_filename':'a.pdf','title':'Alpha'},{'source_filename':'b.pdf','title':'Beta'}]}],'research_gaps':[{'gap_id':'G1','description':'More data','basis':'limitations','supporting_sources':['a.pdf']}],'suggested_state_of_art_structure':[{'section_id':'S1','section_title':'Models','recommended_sources':['a.pdf','b.pdf']}],'comparative_dimensions':[{'dimension':'Method','description':'Compare models','relevant_sources':['a.pdf','b.pdf']}]}
 def ai(self,payload=None,attempt=1,prev=None,deps=None):
  d={'scientific_knowledge_base_csv':ArtifactReference(str(self.kb),h(self.kb)),'scientific_knowledge_base_jsonl':ArtifactReference(str(self.kbj),h(self.kbj)),'scientific_extraction_manifest':ArtifactReference(str(self.m),h(self.m))}; d.update(deps or {})
  return AgentInput('e','r','04_agente_analisis_tematico',attempt,ExecutionMode.FULL_RUN,AgentContext(('llm',),str(self.o),{}),d,{'max_attempts':2,'min_sections':1,'max_sections':5,'manual_review_policy':{'allowed':True}},prev)
 def agent(self,payload=None):
  p=self.payload if payload is None else payload
  return ThematicAnalysisAgent(ThematicRuntimeDependencies(lambda prompt:json.dumps(p),parse_json))
 def test_direct_contract(self):
  x=self.agent().execute(self.ai());self.assertEqual(x.execution_status.value,'COMPLETED');self.assertEqual(x.requested_transition.target_stage,None);self.assertEqual(len(x.output_artifacts),12)
 def test_advance(self):self.assertEqual(self.agent().execute(self.ai()).requested_transition.action.value,'ADVANCE')
 def test_atomic_outputs(self):
  x=self.agent().execute(self.ai());self.assertTrue(all(Path(v.path).is_file() and h(v.path)==v.hash for v in x.output_artifacts.values()))
 def test_invalid_json(self):
  with self.assertRaises(Exception): self.agent().dependencies.parse('{')
 def test_singleton(self):self.assertEqual(len(normalize_thematic_output([self.payload])[0]['themes']),1)
 def test_invented_source(self):
  p=json.loads(json.dumps(self.payload));p['themes'][0]['representative_papers']=[{'source_filename':'x.pdf','title':'X'}];x=self.agent(p).execute(self.ai());self.assertIn('INVALID_REPRESENTATIVE_SOURCE',x.failure_reason_codes)
 def test_title_mismatch(self):
  p=json.loads(json.dumps(self.payload));p['themes'][0]['representative_papers'][0]['title']='Wrong';x=self.agent(p).execute(self.ai());self.assertNotIn('TITLE_MISMATCH',x.failure_reason_codes) # deterministic repair
 def test_theme_without_papers(self):
  p=json.loads(json.dumps(self.payload));p['themes'][0]['representative_papers']=[];x=self.agent(p).execute(self.ai());self.assertIn('EMPTY_REPRESENTATIVE_SOURCE',x.failure_reason_codes)
 def test_gap_without_sources(self):
  p=json.loads(json.dumps(self.payload));p['research_gaps'][0]['supporting_sources']=[];self.assertIn('MISSING_GAP_EVIDENCE',self.agent(p).execute(self.ai()).failure_reason_codes)
 def test_dimension_without_sources(self):
  p=json.loads(json.dumps(self.payload));p['comparative_dimensions'][0]['relevant_sources']=[];self.assertIn('MISSING_COMPARATIVE_EVIDENCE',self.agent(p).execute(self.ai()).failure_reason_codes)
 def test_gt_blocked(self):
  a=self.ai(); d=a.to_dict();d['policy']['ground_truth_path']='/x';self.assertIn('GROUND_TRUTH_POLICY_VIOLATION',self.agent().execute(AgentInput.from_dict(d)).failure_reason_codes)
 def test_kb_missing_failed(self):
  a=self.ai();d=a.to_dict();d['dependencies']['scientific_knowledge_base_csv']['path']=str(self.r/'none');x=self.agent().execute(AgentInput.from_dict(d));self.assertEqual(x.execution_status.value,'FAILED')
 def test_manifest_mismatch(self):
  self.m.write_text(json.dumps({'experiment_id':'other','stage':'03','safety_policy':{}}));a=self.ai();a.dependencies['scientific_extraction_manifest'] if False else None
  # recreate ref
  d=a.to_dict();d['dependencies']['scientific_extraction_manifest']={'path':str(self.m),'hash':h(self.m)};self.assertIn('SCIENTIFIC_EXTRACTION_MANIFEST_MISMATCH',self.agent().execute(AgentInput.from_dict(d)).failure_reason_codes)
 def test_hash_mismatch(self):
  d=self.ai().to_dict();d['dependencies']['scientific_knowledge_base_csv']['hash']='bad';self.assertIn('DEPENDENCY_HASH_MISMATCH',self.agent().execute(AgentInput.from_dict(d)).failure_reason_codes)
 def test_03b_absent_optional(self):self.assertEqual(self.agent().execute(self.ai()).execution_status.value,'COMPLETED')
 def test_03b_partial_failed(self):
  q=self.r/'q.csv';q.write_text('source_filename\na.pdf\n');d={'quantitative_comparative_table':ArtifactReference(str(q),h(q))};self.assertEqual(self.agent().execute(self.ai(deps=d)).execution_status.value,'FAILED')
 def test_fingerprint_stable(self):self.assertEqual(build_thematic_fingerprints(self.ai()).composite,build_thematic_fingerprints(self.ai()).composite)
 def test_fingerprint_policy_change(self):
  a=self.ai();d=a.to_dict();d['policy']['x']=1;self.assertNotEqual(build_thematic_fingerprints(a).composite,build_thematic_fingerprints(AgentInput.from_dict(d)).composite)
 def test_fingerprint_kb_change(self):
  a=self.ai();old=build_thematic_fingerprints(a).composite;self.kb.write_text(self.kb.read_text()+'\n');d=a.to_dict();d['dependencies']['scientific_knowledge_base_csv']['hash']=h(self.kb);self.assertNotEqual(old,build_thematic_fingerprints(AgentInput.from_dict(d)).composite)
 def test_attempt2_plan(self):self.assertEqual(build_repair_plan(['MISSING_THEME_EVIDENCE'])[0]['strategy'],'REGENERATE_THEME_ONLY')
 def test_no_attempt3(self):self.assertEqual(self.agent().execute(self.ai(attempt=3)).execution_status.value,'FAILED')
 def test_transaction(self):
  sp=self.r/'state.json';store=StateStore(sp);
  from datetime import datetime,timezone
  now=datetime.now(timezone.utc).isoformat();store.initialize(PipelineState(PipelineIdentity('e','r',now,now,'v1')));z=execute_thematic_transaction(store=store,agent=self.agent(),agent_input=self.ai());self.assertIsNone(z.committed_state.pending_execution)
 def test_metrics_keys(self):
  x=self.agent().execute(self.ai());self.assertIn('paper_coverage',x.quality_metrics['scientific'])
 def test_no_agent05_import(self):
  import inspect,src.agents.thematic_analysis_agent as m;self.assertNotIn('generador_esquema',inspect.getsource(m));self.assertNotIn('agent05',inspect.getsource(m).lower())
 def test_raw_artifact(self):self.assertTrue(Path(self.agent().execute(self.ai()).output_artifacts['raw'].path).read_text())
 def test_manifest_safety(self):
  x=self.agent().execute(self.ai());m=json.load(open(x.output_artifacts['manifest'].path));self.assertFalse(m['safety_policy']['uses_ground_truth'])
 def test_parse_json(self):self.assertEqual(parse_json('{"x":1}')['x'],1)
 def test_schema_invalid(self):
  with self.assertRaises(ValueError):normalize_thematic_output('x')
 def test_empty_output(self):
  with self.assertRaises(ValueError):normalize_thematic_output({})
 def test_structure_short(self):
  p=json.loads(json.dumps(self.payload));p['suggested_state_of_art_structure']=[];x=self.agent(p).execute(self.ai());self.assertIn('STRUCTURE_TOO_SHORT',x.failure_reason_codes)
 def test_invalid_comparative_source(self):
  p=json.loads(json.dumps(self.payload));p['comparative_dimensions'][0]['relevant_sources']=['x.pdf'];x=self.agent(p).execute(self.ai());self.assertIn('INVALID_COMPARATIVE_SOURCE',x.failure_reason_codes)
 def test_quality_approved(self):self.assertEqual(self.agent().execute(self.ai()).quality_status.value,'APPROVED')
 def test_output_names(self):self.assertEqual(len(self.agent().execute(self.ai()).output_artifacts),12)
 def test_attempt_preserved(self):self.assertEqual(self.agent().execute(self.ai(attempt=2,prev=PreviousAttemptSummary('NEEDS_REVISION',failure_reason_codes=('MISSING_THEME_EVIDENCE',)))).attempt_number,2)
 def test_attempt2_not_full_retry_marker(self):
  prev=PreviousAttemptSummary('NEEDS_REVISION',failure_reason_codes=('MISSING_THEME_EVIDENCE',));x=self.agent().execute(self.ai(attempt=2,prev=prev));m=json.load(open(x.output_artifacts['validation'].path));self.assertEqual(m['repair_plan'][0]['strategy'],'REGENERATE_THEME_ONLY')
 def test_credential_not_fingerprint(self):
  a=self.ai();d=a.to_dict();d['agent_context']['runtime_resources']['OPENAI_API_KEY']='secret';self.assertEqual(build_thematic_fingerprints(a).composite,build_thematic_fingerprints(AgentInput.from_dict(d)).composite)
 def test_comparative_table_rows(self):self.assertEqual(len(pd.read_csv(self.agent().execute(self.ai()).output_artifacts['comparative'].path)),2)
 def test_validation_artifact(self):self.assertTrue(Path(self.agent().execute(self.ai()).output_artifacts['validation'].path).is_file())
 def test_manifest_artifact(self):self.assertTrue(Path(self.agent().execute(self.ai()).output_artifacts['manifest'].path).is_file())
 def test_warning_not_approved(self):
  p=json.loads(json.dumps(self.payload));p['research_gaps'][0]['supporting_sources']=[];self.assertNotEqual(self.agent(p).execute(self.ai()).quality_status.value,'APPROVED_WITH_WARNINGS')
 def tearDown(self):self.t.cleanup()
if __name__=='__main__':unittest.main()
