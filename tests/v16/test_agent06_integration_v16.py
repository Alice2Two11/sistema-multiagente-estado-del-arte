from __future__ import annotations
import json,tempfile,unittest,subprocess,sys,os
from pathlib import Path
import pandas as pd
from src.adapters.draft_writing_runtime import build_real_draft_execution,DraftWritingRuntime
from src.runtime.draft_writing_protocol import execute_draft_transaction
from src.state.state_store import StateStore
from src.state.pipeline_state import PipelineState,PipelineIdentity
class Collection:
 def query(self,**kw):
  src=kw['where']['source_filename'];return {'documents':[[f'The evaluated method achieved a reported accuracy value of 95 percent in {src}.']],'metadatas':[[{'chunk_id':'c1'}]],'distances':[[0.1]]}
class TestAgent06Integration(unittest.TestCase):
 def make_project(self):
  t=tempfile.TemporaryDirectory();root=Path(t.name);eid='exp';exp=root/eid;outputs=exp/'05_outputs';
  (outputs/'04_outline').mkdir(parents=True);(outputs/'03_thematic_analysis').mkdir();(outputs/'01_rag').mkdir();(outputs/'00_orchestrator_planner').mkdir();(exp/'03_chunks').mkdir();(exp/'04_chroma').mkdir();(exp/'04_chroma_index').mkdir();(exp/'04_chroma_index'/'chroma.sqlite3').write_bytes(b'fake')
  active={'active_experiment_id':eid,'run_id':'run','openai_model':'stub','embedding_model_name':'emb','chroma_collection_name':'col','chroma_dir':str(exp/'04_chroma'),'chunks_clean_path':str(exp/'03_chunks'/'chunks_clean_for_rag.csv'),'chroma_manifest_path':str(outputs/'01_rag'/'chroma_index_manifest.json'),'generation_profile':{'output_language':'español','writing_mode':'síntesis crítica','focus_mode':'comparativo','citation_style':'trazable','target_total_words':120,'min_total_words':50,'max_total_words':400},'draft_generation_policy':{'temperature':0.4,'max_section_revision_attempts':1,'top_k_evidence_per_section':3,'max_evidence_chars':18000,'max_quantitative_rows_per_section':5,'allow_open_search_outside_outline_sources':False,'validate_citations_against_section_evidence':True,'validate_numeric_values_against_source_chunks':True,'fail_on_invalid_draft':True}}
  (root/'active_experiment.json').write_text(json.dumps(active))
  outline={'title':'D','sections':[{'section_id':'S1','section_title':'Methods','section_type':'linea_tematica','purpose':'p','key_arguments':['accuracy'],'evidence_needs':['results'],'papers_to_use':[{'source_filename':'a.pdf','title':'A'}]}]}
  for n,v in {'state_of_art_outline.json':outline,'outline_validation_report.json':{'validation_ok':True},'outline_generation_manifest.json':{'experiment_id':eid,'safety_policy':{'uses_ground_truth':False}}}.items():(outputs/'04_outline'/n).write_text(json.dumps(v))
  pd.DataFrame([{'section_id':'S1','source_filename':'a.pdf','title':'A'}]).to_csv(outputs/'04_outline'/'outline_paper_mapping.csv',index=False)
  for n,v in {'thematic_analysis_manifest.json':{'experiment_id':eid,'safety_policy':{'uses_ground_truth':False}},'thematic_validation_report.json':{'validation_ok':True}}.items():(outputs/'03_thematic_analysis'/n).write_text(json.dumps(v))
  pd.DataFrame([{'source_filename':'a.pdf','title':'A'}]).to_csv(outputs/'03_thematic_analysis'/'kb_final_for_thematic_analysis.csv',index=False)
  pd.DataFrame([{'source_filename':'a.pdf','chunk_id':'c1','text':'The evaluated method achieved a reported accuracy value of 95 percent.'}]).to_csv(exp/'03_chunks'/'chunks_clean_for_rag.csv',index=False)
  (outputs/'01_rag'/'chroma_index_manifest.json').write_text(json.dumps({'experiment_id':eid,'collection_name':'col','embedding_model':'emb','num_chunks_indexed':1,'ground_truth_indexed':False,'review_sections_indexed':False,'bibliography_indexed':False,'excluded_chunks_indexed':False,'safety_policy':{'uses_ground_truth':False}}))
  state=PipelineState(identity=PipelineIdentity(experiment_id=eid,run_id='run',created_at='2026-01-01T00:00:00Z',updated_at='2026-01-01T00:00:00Z',schema_version='1.0'))
  store=StateStore(outputs/'00_orchestrator_planner'/'pipeline_state.json');store.initialize(state)
  return t,root,store
 def test_build_real_execution_without_llm_utils_and_transaction(self):
  t,root,store=self.make_project();calls={'n':0}
  def rf(model,temp,collection,project_dir=None):
   self.assertEqual(temp,0.4)
   def invoke(prompt):calls['n']+=1;return json.dumps({'section_id':'S1','section_title':'Methods','draft_text':'The evaluated method achieved a reported accuracy value of 95 percent while documenting experimental conditions, dataset characteristics, evaluation scope, methodological assumptions, and reproducibility limitations within the cited source evidence while also documenting preprocessing decisions, baseline definitions, uncertainty sources, implementation constraints, comparative observations, error analysis procedures, and the practical implications reported for reproducible scientific assessment, including sensitivity analysis, comparative robustness checks, data quality considerations, computational tradeoffs, deployment assumptions, external validity limitations, and recommendations for transparent replication across independent research settings [a.pdf | c1].','claims':[{'claim':'The evaluated method achieved a reported accuracy value of 95 percent while documenting experimental conditions, dataset characteristics, evaluation scope, methodological assumptions, and reproducibility limitations within the cited source evidence while also documenting preprocessing decisions, baseline definitions, uncertainty sources, implementation constraints, comparative observations, error analysis procedures, and the practical implications reported for reproducible scientific assessment, including sensitivity analysis, comparative robustness checks, data quality considerations, computational tradeoffs, deployment assumptions, external validity limitations, and recommendations for transparent replication across independent research settings','supporting_citations':['[a.pdf | c1]']}]})
   return DraftWritingRuntime(invoke,collection)
  agent,ai,cfg=build_real_draft_execution(root,1,collection_factory=lambda cfg:Collection(),runtime_factory=rf,chroma_client_factory=lambda path=None,**kw:type('C',(),{'list_collections':lambda self:[type('N',(),{'name':'col'})()]})())
  self.assertEqual(Path(cfg['chroma_dir']).name,'04_chroma_index')
  self.assertEqual(ai.stage_name,'06_agente_redactor');self.assertFalse((root/'src'/'llm_utils.py').exists())
  tx=execute_draft_transaction(store=store,agent=agent,agent_input=ai);self.assertEqual(tx.agent_result.quality_status.value,'APPROVED');self.assertEqual(calls['n'],1);t.cleanup()
 def test_notebook_clean_kernel_precheck_structure(self):
  root=Path(__file__).parents[2];nb=json.loads((root/'06_agente_redactor_migrado_v16.ipynb').read_text());code='\n'.join(c['source'] if isinstance(c['source'],str) else ''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code');compile(code,'notebook','exec');self.assertLess(code.index('git", "clone'),code.index('from src.adapters'))
