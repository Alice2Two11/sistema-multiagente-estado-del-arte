from __future__ import annotations
import hashlib,json,tempfile,unittest
from pathlib import Path
import pandas as pd
from src.tools.thematic_analysis.schema_validation import normalize_thematic_output,inspect_thematic_payload,validate_json_to_tables
from src.tools.thematic_analysis.artifacts import thematic_table_counts
from src.tools.thematic_analysis.deterministic_repair import execute_deterministic_thematic_repair


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class TestAgent04DeterministicFlatteningRepair(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        pd.DataFrame([
            {'source_filename':'a.pdf','title':'Alpha','methods_or_models':'ANN','limitations_or_gaps':'Gap A'},
            {'source_filename':'b.pdf','title':'Beta','methods_or_models':'SVM','limitations_or_gaps':'Gap B'},
        ]).to_csv(self.root/'kb_final_for_thematic_analysis.csv',index=False)
        pd.DataFrame(columns=['source_filename','title']).to_csv(self.root/'kb_excluded_from_thematic_analysis.csv',index=False)
        self.payload={
            'themes':[{'theme':'Modelos predictivos','representative_papers':[{'source_filename':'a.pdf','title':'Alpha'}]}],
            'research_gaps':[{'gap':'Falta validación externa','sources':['a.pdf']}],
            'suggested_state_of_art_structure':[{'section':'Modelos','content':'Comparar ANN y SVM','sources':['a.pdf','b.pdf']}],
            'comparative_dimensions':[{'dimension':'Método','description':'Comparación','sources':['a.pdf','b.pdf']}],
        }
        (self.root/'thematic_analysis.json').write_text(json.dumps(self.payload,ensure_ascii=False),encoding='utf-8')
        (self.root/'thematic_analysis_raw.txt').write_text(json.dumps(self.payload,ensure_ascii=False),encoding='utf-8')
        (self.root/'thematic_analysis_manifest.json').write_text(json.dumps({'outputs':{}}),encoding='utf-8')
    def test_theme_alias_and_id(self):
        data,issues,repairs=normalize_thematic_output(self.payload,return_repairs=True)
        self.assertEqual(data['themes'][0]['theme_name'],'Modelos predictivos'); self.assertEqual(data['themes'][0]['theme_id'],'T1')
    def test_representative_papers_preserved(self):
        data,_,_=normalize_thematic_output(self.payload,return_repairs=True);self.assertEqual(data['themes'][0]['representative_papers'][0]['source_filename'],'a.pdf')
    def test_gap_aliases(self):
        data,_,_=normalize_thematic_output(self.payload,return_repairs=True);g=data['research_gaps'][0];self.assertEqual(g['gap_id'],'G1');self.assertEqual(g['description'],'Falta validación externa');self.assertEqual(g['supporting_sources'],['a.pdf'])
    def test_structure_aliases(self):
        data,_,_=normalize_thematic_output(self.payload,return_repairs=True);s=data['suggested_state_of_art_structure'][0];self.assertEqual(s['section_id'],'S1');self.assertEqual(s['section_title'],'Modelos');self.assertEqual(s['description'],'Comparar ANN y SVM')
    def test_dimension_sources_alias(self):
        data,_,_=normalize_thematic_output(self.payload,return_repairs=True);self.assertEqual(data['comparative_dimensions'][0]['relevant_sources'],['a.pdf','b.pdf'])
    def test_cross_validation_detects_failure(self):
        raw=inspect_thematic_payload(self.payload);codes,_=validate_json_to_tables(raw,{'flattened_theme_semantic_rows':0,'flattened_gap_semantic_rows':0,'flattened_structure_semantic_rows':0,'flattened_comparative_dimension_semantic_rows':0});self.assertIn('THEME_FLATTENING_FAILED',codes);self.assertIn('ALIAS_MAPPING_REQUIRED',codes)
    def test_cross_validation_passes_after_normalization(self):
        data,_,_=normalize_thematic_output(self.payload,return_repairs=True);codes,_=validate_json_to_tables(inspect_thematic_payload(self.payload),thematic_table_counts(data));self.assertEqual(codes,[])
    def test_repair_without_openai_and_preserves_json(self):
        before=sha(self.root/'thematic_analysis.json'); result=execute_deterministic_thematic_repair(output_dir=self.root);self.assertEqual(result.tool_usage.llm_calls,0);self.assertEqual(before,sha(self.root/'thematic_analysis.json'))
    def test_repair_regenerates_tables(self):
        execute_deterministic_thematic_repair(output_dir=self.root);self.assertEqual(pd.read_csv(self.root/'themes_summary.csv').loc[0,'theme_name'],'Modelos predictivos');self.assertEqual(pd.read_csv(self.root/'research_gaps.csv').loc[0,'description'],'Falta validación externa')
    def test_manifest_records_repair(self):
        execute_deterministic_thematic_repair(output_dir=self.root);m=json.loads((self.root/'thematic_analysis_manifest.json').read_text());self.assertTrue(m['deterministic_thematic_repair']);self.assertFalse(m['openai_called']);self.assertTrue(m['original_json_preserved'])
    def test_validation_records_repairs(self):
        execute_deterministic_thematic_repair(output_dir=self.root);v=json.loads((self.root/'thematic_validation_report.json').read_text());self.assertTrue(v['repairs']);self.assertTrue(v['deterministic_thematic_repair'])
    def test_quality_not_manual_review_for_flattening(self):
        from src.tools.thematic_analysis.quality import classify_quality
        q,a=classify_quality(['THEME_FLATTENING_FAILED'],2,True);self.assertEqual(q.value,'NEEDS_REVISION');self.assertEqual(a,'HALT_STAGE')
    def tearDown(self): self.tmp.cleanup()

if __name__=='__main__': unittest.main()
