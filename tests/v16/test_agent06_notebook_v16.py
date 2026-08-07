import json,unittest
from pathlib import Path
class TestAgent06Notebook(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.root=Path(__file__).parents[2];cls.path=cls.root/'06_agente_redactor_migrado_v16.ipynb';cls.nb=json.loads(cls.path.read_text());cls.code='\n'.join(c.get('source','') if isinstance(c.get('source',''),str) else ''.join(c.get('source',[])) for c in cls.nb['cells'] if c['cell_type']=='code')
 def test_thin_shell_no_scientific_functions(self):
  for token in ('def retrieve_section_evidence','def build_section_prompt','def validate_generated_section','chromadb.PersistentClient') : self.assertNotIn(token,self.code)
 def test_bootstrap_before_src_import(self):
  self.assertLess(self.code.index('git", "clone'),self.code.index('from src.adapters'))
 def test_real_repository_url(self):self.assertIn('https://github.com/Alice2Two11/tesis-sistema-multiagente.git',self.code)
 def test_modes_and_attempt(self):self.assertIn('AGENT06_RUN_MODE',self.code);self.assertIn('AGENT06_ATTEMPT_NUMBER',self.code)
 def test_no_agent07(self):self.assertNotIn('agente_verificador',self.code.lower());self.assertNotIn('07_',self.code)
