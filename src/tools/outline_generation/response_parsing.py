from __future__ import annotations
import json,re
def extract_first_valid_json(text, fallback=None):
 text=str(text).strip(); candidates=[]; fenced=re.search(r'```(?:json)?\s*(.*?)```',text,re.S|re.I)
 if fenced:candidates.append(fenced.group(1).strip())
 candidates.append(text); dec=json.JSONDecoder()
 for cand in candidates:
  for m in re.finditer(r'\{',cand):
   try:return dec.raw_decode(cand[m.start():])[0]
   except json.JSONDecodeError:pass
 if fallback:return fallback(text)
 raise ValueError('INVALID_LLM_OUTPUT')
