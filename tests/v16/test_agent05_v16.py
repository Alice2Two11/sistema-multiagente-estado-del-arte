from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
import pandas as pd
from src.agents.outline_generation_agent import OutlineGenerationAgent
from src.adapters.outline_generation_runtime import OutlineGenerationRuntime
from src.contracts.agent_input import AgentInput,AgentContext,ArtifactReference,ExecutionMode,PreviousAttemptSummary
from src.contracts.agent_result import ExecutionStatus,QualityStatus,TransitionAction
from src.state.fingerprints import sha256_file
from src.tools.outline_generation.source_repair import repair_outline_sources
from src.tools.outline_generation.outline_validation import validate_outline

VALID_OUTLINE={'title':'Estado del arte','topic':'x','objective':'obj','narrative_strategy':'orden','sections':[{'section_id':'S1','section_title':'Introducción','section_type':'introduccion','purpose':'p','themes_used':['T1'],'key_arguments':['a'],'papers_to_use':[],'evidence_needs':['e'],'expected_output':'o','transition_to_next':'t'},{'section_id':'S2','section_title':'Métodos','section_type':'linea_tematica','purpose':'p','themes_used':['T1'],'key_arguments':['a'],'papers_to_use':[{'source_filename':'a.pdf','title':'Paper A','reason_to_use':'r'}],'evidence_needs':['e'],'expected_output':'o','transition_to_next':'t'},{'section_id':'S3','section_title':'Comparación','section_type':'comparacion','purpose':'p','themes_used':['T1'],'key_arguments':['a'],'papers_to_use':[{'source_filename':'b.pdf','title':'Paper B','reason_to_use':'r'}],'evidence_needs':['e'],'expected_output':'o','transition_to_next':'t'},{'section_id':'S4','section_title':'Gaps','section_type':'gaps','purpose':'p','themes_used':['T1'],'key_arguments':['a'],'papers_to_use':[],'evidence_needs':['e'],'expected_output':'o','transition_to_next':''}], 'paper_coverage_summary':[{'source_filename':'a.pdf','title':'Paper A','used_in_sections':['S2'],'role':'metodologico'}],'global_writing_guidelines':['g'],'risks_or_warnings':['w']}
class Env:
 def __init__(self,outline=VALID_OUTLINE,attempt=1):
  self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);self.inp=self.root/'05_outputs'/'03_thematic_analysis';self.out=self.root/'05_outputs'/'04_outline';self.inp.mkdir(parents=True);self.out.mkdir(parents=True)
  files={
   'thematic_analysis.json':{'themes':[]},'thematic_analysis_manifest.json':{'experiment_id':'exp','run_id':'run','fingerprint':'f'},'thematic_validation_report.json':{'validation_ok':True},
  }
  for n,v in files.items():(self.inp/n).write_text(json.dumps(v),encoding='utf-8')
  pd.DataFrame([{'theme_id':'T1','theme_name':'Tema','description':'d','representative_papers':'a.pdf'}]).to_csv(self.inp/'themes_summary.csv',index=False)
  pd.DataFrame([{'gap_id':'G1','description':'gap','basis':'b','supporting_sources':'a.pdf'}]).to_csv(self.inp/'research_gaps.csv',index=False)
  pd.DataFrame([{'source_filename':'a.pdf','title':'Paper A'},{'source_filename':'b.pdf','title':'Paper B'}]).to_csv(self.inp/'comparative_table_papers.csv',index=False)
  pd.DataFrame([{'source_filename':'a.pdf','title':'Paper A','research_problem':'p'},{'source_filename':'b.pdf','title':'Paper B','research_problem':'p'}]).to_csv(self.inp/'kb_final_for_thematic_analysis.csv',index=False)
  (self.inp/'suggested_state_of_art_structure.json').write_text(json.dumps([{'section_id':'S1'}]),encoding='utf-8')
  names={'thematic_analysis_json':'thematic_analysis.json','thematic_analysis_manifest':'thematic_analysis_manifest.json','thematic_validation_report':'thematic_validation_report.json','themes_summary_csv':'themes_summary.csv','research_gaps_csv':'research_gaps.csv','comparative_table_papers_csv':'comparative_table_papers.csv','kb_final_for_thematic_analysis_csv':'kb_final_for_thematic_analysis.csv','suggested_structure_json':'suggested_state_of_art_structure.json'}
  deps={k:ArtifactReference(str(self.inp/v),sha256_file(self.inp/v)) for k,v in names.items()}
  self.calls=0
  def invoke(prompt):self.calls+=1;return json.dumps(outline)
  self.agent=OutlineGenerationAgent(OutlineGenerationRuntime(invoke))
  self.ai=AgentInput(experiment_id='exp',run_id='run',stage_name='05_generador_esquema',attempt_number=attempt,mode=ExecutionMode.FULL_RUN,agent_context=AgentContext(allowed_tools=('llm',),output_directory=str(self.out)),dependencies=deps,policy={'min_sections':4,'max_sections':5,'max_field_chars':1800,'title_match_cutoff':0.55,'output_language':'español académico','current_fingerprint':'fp','validation_version':'v2','prompt_version':'v2','schema_version':'v2'},previous_attempt=PreviousAttemptSummary(quality_status='NEEDS_REVISION') if attempt==2 else None)
 def close(self):self.t.cleanup()
class TestAgent05(unittest.TestCase):
 def test_direct_contract_approved(self):
  e=Env();r=e.agent.execute(e.ai);self.assertEqual(r.execution_status,ExecutionStatus.COMPLETED);self.assertEqual(r.quality_status,QualityStatus.APPROVED);self.assertEqual(r.requested_transition.action,TransitionAction.ADVANCE);self.assertEqual(len(r.output_artifacts),7);self.assertEqual(e.calls,1);e.close()
 def test_exact_artifacts(self):
  e=Env();r=e.agent.execute(e.ai);self.assertEqual({Path(x.path).name for x in r.output_artifacts.values()},{'state_of_art_outline.json','state_of_art_outline_raw.txt','state_of_art_outline.md','outline_sections.csv','outline_paper_mapping.csv','outline_validation_report.json','outline_generation_manifest.json'});e.close()
 def test_validation_false_preserves_artifacts(self):
  bad=dict(VALID_OUTLINE);bad['sections']=[];e=Env(bad);r=e.agent.execute(e.ai);self.assertEqual(r.quality_status,QualityStatus.NEEDS_REVISION);self.assertEqual(r.requested_transition.action,TransitionAction.RETRY);self.assertEqual(len(r.output_artifacts),7);e.close()
 def test_attempt2_halts_no_attempt3(self):
  bad=dict(VALID_OUTLINE);bad['sections']=[];e=Env(bad,2);r=e.agent.execute(e.ai);self.assertEqual(r.quality_status,QualityStatus.NEEDS_REVISION);self.assertEqual(r.requested_transition.action,TransitionAction.HALT_STAGE);e.close()
 def test_reuse_zero_llm_calls(self):
  e=Env();r1=e.agent.execute(e.ai);self.assertEqual(e.calls,1);e.calls=0;r2=e.agent.execute(e.ai);self.assertEqual(e.calls,0);self.assertEqual(r2.tool_usage.llm_calls,0);e.close()
 def test_title_repair_cutoff(self):
  o={'sections':[{'section_id':'S1','papers_to_use':[{'source_filename':'x','title':'Paper A'}]}]};rep,un=repair_outline_sources(o,{'a.pdf'},{'a.pdf':'Paper A'},{'Paper A':'a.pdf'},0.55);self.assertEqual(o['sections'][0]['papers_to_use'][0]['source_filename'],'a.pdf');self.assertEqual(len(rep),1);self.assertFalse(un)
 def test_trim_max_sections(self):
  o=json.loads(json.dumps(VALID_OUTLINE));o['sections'] += [dict(o['sections'][-1]),dict(o['sections'][-1])];v=validate_outline(o,{'a.pdf','b.pdf'},4,5,[],[],[],[]);self.assertEqual(len(o['sections']),5);self.assertTrue(v['sections_trimmed_to_max'])
 def test_no_rag_import(self):
  text=(Path(__file__).parents[2]/'src'/'agents'/'outline_generation_agent.py').read_text();self.assertNotIn('chromadb',text.lower());self.assertNotIn('retriever',text.lower())
 def test_prompt_temperature_runtime_contract(self):
  text=(Path(__file__).parents[2]/'src'/'adapters'/'outline_generation_runtime.py').read_text();self.assertIn('temperature=0',text)
 def test_stage_name(self):
  e=Env();self.assertEqual(e.ai.stage_name,'05_generador_esquema');e.close()
