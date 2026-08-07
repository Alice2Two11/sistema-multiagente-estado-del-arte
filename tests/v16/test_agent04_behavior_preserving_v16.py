import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.tools.thematic_analysis.coverage_validation import calculate_diagnostic_metrics
from src.tools.thematic_analysis.schema_validation import normalize_thematic_output
from src.tools.thematic_analysis.artifacts import ARTIFACT_FILENAMES


class BehaviorPreservingAgent04Tests(unittest.TestCase):
    def test_representative_papers_are_not_exhaustive_assignment(self):
        data={
            'themes':[{'theme_id':'T1','theme_name':'Tema','description':'Tema','representative_papers':['a.pdf']}],
            'research_gaps':[{'gap_id':'G1','description':'Gap','basis':'Gap','supporting_sources':['a.pdf']}],
            'suggested_state_of_art_structure':[{'section_id':'S1','section_title':'Sec','description':'Sec','recommended_sources':[]}],
            'comparative_dimensions':[{'dimension':'Método','description':'Comparar','relevant_sources':['a.pdf']}],
        }
        df=pd.DataFrame([{'source_filename':'a.pdf'},{'source_filename':'b.pdf'}])
        metrics=calculate_diagnostic_metrics(data,df,{'reference_total':3,'valid_references':3,'title_total':0,'title_matches':0})
        self.assertIsNone(metrics['paper_coverage'])
        self.assertIsNone(metrics['papers_assigned_to_theme_rate'])
        self.assertEqual(metrics['coverage_semantics'],'NOT_APPLICABLE_REPRESENTATIVE_PAPERS_ARE_NON_EXHAUSTIVE')

    def test_safe_aliases_preserve_llm_information(self):
        payload={
            'themes':[{'theme':'Tema A','representative_papers':['a.pdf']}],
            'research_gaps':[{'gap':'Vacío A','sources':['a.pdf']}],
            'suggested_state_of_art_structure':[{'section':'Sección A','content':'Contenido'}],
            'comparative_dimensions':[{'dimension':'Método','description':'Comparación','sources':['a.pdf']}],
        }
        data,issues,repairs=normalize_thematic_output(payload,return_repairs=True)
        self.assertFalse(issues)
        self.assertEqual(data['themes'][0]['theme_name'],'Tema A')
        self.assertEqual(data['themes'][0]['theme_id'],'T1')
        self.assertEqual(data['research_gaps'][0]['description'],'Vacío A')
        self.assertEqual(data['research_gaps'][0]['gap_id'],'G1')
        self.assertEqual(data['suggested_state_of_art_structure'][0]['section_title'],'Sección A')
        self.assertEqual(data['suggested_state_of_art_structure'][0]['description'],'Contenido')
        self.assertEqual(data['comparative_dimensions'][0]['relevant_sources'],['a.pdf'])
        self.assertTrue(repairs)

    def test_no_assignment_artifact(self):
        self.assertNotIn('assignments',ARTIFACT_FILENAMES)
        self.assertNotIn('paper_theme_assignments.csv',ARTIFACT_FILENAMES.values())

    def test_no_theme_assignment_module(self):
        root=Path(__file__).resolve().parents[2]
        self.assertFalse((root/'src/tools/thematic_analysis/theme_assignment.py').exists())
        source='\n'.join(p.read_text(encoding='utf-8') for p in (root/'src').rglob('*.py'))
        self.assertNotIn('assigned_papers',source)
        self.assertNotIn('ASSIGN_UNASSIGNED_PAPERS_ONLY',source)
