from __future__ import annotations
from difflib import get_close_matches
def as_list(v): return [] if v is None else (v if isinstance(v,list) else [v])
def repair_outline_sources(outline,valid_sources,source_to_title,title_to_source,cutoff=0.55):
 repairs=[]; unresolved=[]; titles=list(title_to_source)
 for sec in as_list(outline.get('sections',[])):
  if not isinstance(sec,dict): continue
  clean=[]
  for p in as_list(sec.get('papers_to_use',[])):
   if not isinstance(p,dict): continue
   source=str(p.get('source_filename','')).strip(); title=str(p.get('title','')).strip()
   if source in valid_sources:
    if not title:p['title']=source_to_title.get(source,'')
    clean.append(p); continue
   matches=get_close_matches(title,titles,n=1,cutoff=cutoff)
   if matches:
    mt=matches[0]; ns=title_to_source[mt]; repairs.append({'section_id':sec.get('section_id'),'old_source_filename':source,'generated_title':title,'matched_title':mt,'new_source_filename':ns}); p['source_filename']=ns;p['title']=mt;clean.append(p)
   else: unresolved.append({'section_id':sec.get('section_id'),'source_filename':source,'title':title})
  sec['papers_to_use']=clean
 return repairs,unresolved
def repair_coverage_summary(outline,valid_sources,source_to_title,title_to_source,cutoff=0.55):
 repairs=[];unresolved=[];clean=[];titles=list(title_to_source)
 for item in as_list(outline.get('paper_coverage_summary',[])):
  if not isinstance(item,dict):continue
  source=str(item.get('source_filename','')).strip();title=str(item.get('title','')).strip()
  if source in valid_sources:
   if not title:item['title']=source_to_title.get(source,'')
   clean.append(item);continue
  matches=get_close_matches(title,titles,n=1,cutoff=cutoff)
  if matches:
   mt=matches[0];ns=title_to_source[mt];repairs.append({'old_source_filename':source,'generated_title':title,'matched_title':mt,'new_source_filename':ns});item['source_filename']=ns;item['title']=mt;clean.append(item)
  else:unresolved.append({'source_filename':source,'title':title})
 outline['paper_coverage_summary']=clean;return repairs,unresolved
