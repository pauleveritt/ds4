# Generalized form of the committed probe.py: identical request payload,
# prompt file taken as an argument so Task A (p2) and Task B (p3) share one path.
import json,sys,urllib.request,time,re
model,tag,maxtok,think,pf = sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4],sys.argv[5]
base=pf.rsplit('/',1)[-1].rsplit('.',1)[0]
p=open(f'/tmp/mellum-eval/{pf}').read()
payload={"model":model,"messages":[{"role":"user","content":p}],
         "max_tokens":maxtok,"temperature":0.2,"top_p":0.9}
if think=="off":
    payload["reasoning_effort"]="none"
    payload["chat_template_kwargs"]={"enable_thinking":False}
req=urllib.request.Request('http://127.0.0.1:8125/v1/chat/completions',
  data=json.dumps(payload).encode(),
  headers={'Content-Type':'application/json','Authorization':'Bearer evalkey'})
t=time.time()
try:
    r=json.load(urllib.request.urlopen(req,timeout=1800))
    m=r['choices'][0]['message']
    txt=m.get('content') or ''
    reason=m.get('reasoning_content') or ''
    fin=r['choices'][0].get('finish_reason')
    u=r.get('usage',{})
    inline=re.search(r'<think>(.*?)</think>', txt, re.S)
    if inline and not reason:
        reason=inline.group(1); txt=txt[inline.end():]
    open(f'/tmp/mellum-eval/out/{tag}.{base}.txt','w').write(txt)
    print(f"{tag:<24} finish={fin:<10} out={u.get('completion_tokens','?'):>5} "
          f"think~{len(reason.split()):>5}w ans~{len(txt.split()):>4}w {time.time()-t:6.1f}s")
except Exception as e:
    print(f"{tag:<24} FAIL {type(e).__name__} {str(e)[:100]}")
