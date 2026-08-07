from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from src.agents.draft_writing_agent import DraftWritingAgent
from src.adapters.draft_writing_runtime import DraftWritingRuntime,build_openai_draft_runtime
from src.contracts.agent_input import AgentInput,AgentContext,ArtifactReference,ExecutionMode,PreviousAttemptSummary
from src.contracts.agent_result import ExecutionStatus,QualityStatus,TransitionAction
from src.state.fingerprints import sha256_file
from src.tools.draft_writing.retrieval import retrieve_section_evidence
from src.tools.draft_writing.validation import validate_generated_section

class Collection:
 def __init__(self,empty=False):self.empty=empty;self.wheres=[]
 def query(self,**kw):
  self.wheres.append(kw.get('where'))
  if self.empty:return {'documents':[[]],'metadatas':[[]],'distances':[[]]}
  src=kw['where']['source_filename'];return {'documents':[[f'Method accuracy 95 percent from {src}.']],'metadatas':[[{'chunk_id':'c1','source_filename':src}]],'distances':[[0.1]]}
class Runtime(DraftWritingRuntime):
 pass
class Env:
 def __init__(self,attempt=1,collection=None,invalid_global=False):
  self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);self.inp=self.root/'in';self.out=self.root/'out';self.inp.mkdir();self.out.mkdir();self.collection=collection or Collection();self.calls=0
  outline={'title':'Draft','topic':'x','sections':[{'section_id':'S1','section_title':'Methods','section_type':'linea_tematica','purpose':'Compare methods','key_arguments':['accuracy'],'evidence_needs':['results'],'papers_to_use':[{'source_filename':'a.pdf','title':'A'}]},{'section_id':'S2','section_title':'Conclusión','section_type':'conclusion','purpose':'Cerrar','key_arguments':[],'evidence_needs':[],'papers_to_use':[]}]}
  files={'outline.json':outline,'outline_validation.json':{'validation_ok':True},'outline_manifest.json':{'experiment_id':'exp','safety_policy':{'uses_ground_truth':False}},'thematic_manifest.json':{'experiment_id':'exp','safety_policy':{'uses_ground_truth':False}},'thematic_validation.json':{'validation_ok':True},'chroma_manifest.json':{'experiment_id':'exp','collection_name':'col','embedding_model':'emb','num_chunks_indexed':2,'ground_truth_indexed':False,'review_sections_indexed':False,'bibliography_indexed':False,'excluded_chunks_indexed':False,'safety_policy':{'uses_ground_truth':False}}}
  for n,v in files.items():(self.inp/n).write_text(json.dumps(v),encoding='utf-8')
  pd.DataFrame([{'section_id':'S1','source_filename':'a.pdf','title':'A'}]).to_csv(self.inp/'mapping.csv',index=False)
  pd.DataFrame([{'source_filename':'a.pdf','title':'A'}]).to_csv(self.inp/'kb.csv',index=False)
  pd.DataFrame([{'source_filename':'a.pdf','chunk_id':'c1','text':'Method accuracy 95 percent from a.pdf.'},{'source_filename':'b.pdf','chunk_id':'b1','text':'Other source.'}]).to_csv(self.inp/'chunks.csv',index=False)
  names={'outline_json':'outline.json','outline_mapping':'mapping.csv','outline_validation':'outline_validation.json','outline_manifest':'outline_manifest.json','kb_final':'kb.csv','thematic_manifest':'thematic_manifest.json','thematic_validation':'thematic_validation.json','chunks_clean':'chunks.csv','chroma_manifest':'chroma_manifest.json'}
  deps={k:ArtifactReference(str(self.inp/v),sha256_file(self.inp/v)) for k,v in names.items()}
  def invoke(prompt):
   self.calls+=1
   return json.dumps({'section_id':'S1','section_title':'Methods','draft_text':'The evaluated method reports 95 percent accuracy and provides a detailed comparison of model behavior across multiple experimental conditions, datasets, evaluation settings, methodological assumptions, and reproducibility constraints documented in the selected scientific evidence while also documenting preprocessing decisions, baseline definitions, uncertainty sources, implementation constraints, comparative observations, error analysis procedures, and the practical implications reported for reproducible scientific assessment [a.pdf | c1].','claims':[{'claim':'The evaluated method reports 95 percent accuracy and provides a detailed comparison of model behavior across multiple experimental conditions, datasets, evaluation settings, methodological assumptions, and reproducibility constraints documented in the selected scientific evidence while also documenting preprocessing decisions, baseline definitions, uncertainty sources, implementation constraints, comparative observations, error analysis procedures, and the practical implications reported for reproducible scientific assessment','supporting_citations':['[a.pdf | c1]']}]})
  self.runtime=DraftWritingRuntime(invoke,self.collection);self.agent=DraftWritingAgent(self.runtime)
  self.ai=AgentInput(experiment_id='exp',run_id='run',stage_name='06_agente_redactor',attempt_number=attempt,mode=ExecutionMode.FULL_RUN,agent_context=AgentContext(allowed_tools=('llm','chroma'),output_directory=str(self.out),runtime_resources={'chroma_collection_name':'col','embedding_model_name':'emb'}),dependencies=deps,policy={'temperature':0.0,'max_section_revision_attempts':2,'top_k_evidence_per_section':3,'max_evidence_chars':18000,'max_quantitative_rows_per_section':12,'current_fingerprint':'fp','prompt_version':'p','rag_version':'r','validation_version':'v','output_language':'español','writing_mode':'síntesis crítica','focus_mode':'comparativo','citation_style':'trazable','target_total_words':120,'min_total_words':50,'max_total_words':400},previous_attempt=PreviousAttemptSummary(quality_status='NEEDS_REVISION') if attempt==2 else None)
 def close(self):self.t.cleanup()
class TestAgent06(unittest.TestCase):
 def test_approved_contract_and_exact_artifacts(self):
  e=Env();r=e.agent.execute(e.ai);self.assertEqual(r.execution_status,ExecutionStatus.COMPLETED);self.assertEqual(r.quality_status,QualityStatus.APPROVED);self.assertEqual(r.requested_transition.action,TransitionAction.ADVANCE);self.assertEqual(len([k for k in r.output_artifacts if k!='raw_section_outputs']),12);self.assertIn('raw_section_outputs',r.output_artifacts);e.close()
 def test_restricted_rag_no_outside_source(self):
  e=Env();r=e.agent.execute(e.ai);self.assertEqual(e.collection.wheres,[{'source_filename':'a.pdf'}]);self.assertNotIn('b.pdf',(e.out/'draft_rag_evidence.csv').read_text());e.close()
 def test_csv_fallback_restricted(self):
  c=Collection(empty=True);section={'section_title':'accuracy','purpose':'method','papers_to_use':[{'source_filename':'a.pdf'}]};df=pd.DataFrame([{'source_filename':'a.pdf','chunk_id':'c1','text':'accuracy method evidence'},{'source_filename':'b.pdf','chunk_id':'b1','text':'accuracy method evidence'}]);rows=retrieve_section_evidence(section,c,df,5);self.assertEqual({r['source_filename'] for r in rows},{'a.pdf'});self.assertEqual(rows[0]['retrieval_method'],'csv_lexical_restricted')
 def test_internal_revision_attempts_separate(self):
  e=Env();count={'n':0}
  def invoke(prompt):
   count['n']+=1
   if count['n']==1:return json.dumps({'section_id':'S1','section_title':'Methods','draft_text':'This unsupported scientific statement lacks any valid evidence citation.','claims':[]})
   return json.dumps({'section_id':'S1','section_title':'Methods','draft_text':'The evaluated method reports 95 percent accuracy and provides a detailed comparison of model behavior across multiple experimental conditions, datasets, evaluation settings, methodological assumptions, and reproducibility constraints documented in the selected scientific evidence while also documenting preprocessing decisions, baseline definitions, uncertainty sources, implementation constraints, comparative observations, error analysis procedures, and the practical implications reported for reproducible scientific assessment [a.pdf | c1].','claims':[{'claim':'The evaluated method reports 95 percent accuracy and provides a detailed comparison of model behavior across multiple experimental conditions, datasets, evaluation settings, methodological assumptions, and reproducibility constraints documented in the selected scientific evidence while also documenting preprocessing decisions, baseline definitions, uncertainty sources, implementation constraints, comparative observations, error analysis procedures, and the practical implications reported for reproducible scientific assessment','supporting_citations':['[a.pdf | c1]']}]})
  e.agent=DraftWritingAgent(DraftWritingRuntime(invoke,e.collection));r=e.agent.execute(e.ai);self.assertEqual(r.attempt_number,1);self.assertEqual(r.tool_usage.llm_calls,2);self.assertTrue((e.out/'raw_section_outputs'/'S1_attempt_2.txt').exists());e.close()
 def test_literal_numeric_validation(self):
  gen={'section_id':'S1','section_title':'M','draft_text':'The evaluated method achieved a reported accuracy value of 99 percent [a.pdf | c1].','claims':[{'claim':'The evaluated method achieved a reported accuracy value of 99 percent','supporting_citations':['[a.pdf | c1]']}]};sec={'section_id':'S1'};ev=[{'source_filename':'a.pdf','chunk_id':'c1','text':'The evaluated method achieved a reported accuracy value of 95 percent.'}];v=validate_generated_section(gen,sec,ev);self.assertTrue(any('99' in x for x in v['numeric_errors']))
 def test_citation_format(self):
  e=Env();r=e.agent.execute(e.ai);self.assertIn('[a.pdf | c1]',(e.out/'state_of_art_draft.md').read_text());e.close()
 def test_validation_false_attempt1_retry_partial_only(self):
  e=Env()
  with patch('src.agents.draft_writing_agent.validate_draft_global',return_value={'validation_ok':False,'invalid_sections':['S1']}):r=e.agent.execute(e.ai)
  self.assertEqual(r.quality_status,QualityStatus.NEEDS_REVISION);self.assertEqual(r.requested_transition.action,TransitionAction.RETRY);self.assertFalse((e.out/'state_of_art_draft.json').exists());self.assertTrue((e.out/'draft_validation_report.json').exists());e.close()
 def test_validation_false_attempt2_halt(self):
  e=Env(attempt=2)
  with patch('src.agents.draft_writing_agent.validate_draft_global',return_value={'validation_ok':False,'invalid_sections':['S1']}):r=e.agent.execute(e.ai)
  self.assertEqual(r.requested_transition.action,TransitionAction.HALT_STAGE);e.close()
 def test_reuse_zero_calls(self):
  e=Env();r1=e.agent.execute(e.ai);self.assertEqual(e.calls,1);e.calls=0;r2=e.agent.execute(e.ai);self.assertEqual(r2.tool_usage.llm_calls,0);self.assertEqual(e.calls,0);e.close()
 def test_runtime_without_llm_utils(self):
  class LLM:
   def __init__(self,**kw):self.kw=kw
   def invoke(self,msgs):return type('R',(),{'content':'{"ok":true}'})()
  rt=build_openai_draft_runtime('m',0.3,Collection(),project_dir=self._credential_dir(),llm_factory=LLM,human_message_factory=lambda content:type('M',(),{'content':content})())
  self.assertEqual(rt.parse(rt.invoke('x')),{'ok':True})
 def _credential_dir(self):
  t=Path(tempfile.mkdtemp());d=t/'.runtime_secrets';d.mkdir();(d/'openai_api_key.txt').write_text('test');return t
