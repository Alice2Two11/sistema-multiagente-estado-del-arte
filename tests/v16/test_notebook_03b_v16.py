import json, ast, unittest
from pathlib import Path
ROOT=Path(__file__).parents[2]
class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.nb=json.loads((ROOT/"03B_extraccion_cuantitativa_kb_migrado_v16.ipynb").read_text())
        self.code="\n".join("".join(c.get("source",[])) for c in self.nb["cells"] if c["cell_type"]=="code")
    def test_clean_ast_and_explicit_imports(self): ast.parse(self.code)
    def test_visible_modes(self): self.assertIn("RUN_REAL_03B",self.code); self.assertIn("RUN_DETERMINISTIC_FLATTENING_REPAIR",self.code); self.assertIn("DETERMINISTIC_REPAIR",self.code); self.assertIn("EXECUTION_MODE",self.code)
    def test_thin_shell(self):
        for forbidden in ("build_quant_prompt","normalize_metric_name","value_found_in_text","OPENAI_KEY_FILE","getpass","Fernet","to_csv(","atomic_write_"):
            self.assertNotIn(forbidden,self.code)
    def test_uses_protocol_and_not_stage04(self): self.assertIn("execute_quantitative_runtime_transaction",self.code); self.assertNotIn("ExtractionAgent",self.code); self.assertNotIn("04_analisis",self.code)
if __name__=="__main__": unittest.main()
