const API='';var _reportText='';var _queryText='';var _currentTaskId=null;var _aborted=false;
var _chartInstances=[];

function setQuery(q){document.getElementById('query').value=q}
function updateToggle(){document.getElementById('toggleText').textContent=document.getElementById('webSearch').checked?'联网搜索':'仅本地知识库'}
function toggleSettings(){document.getElementById('settingsPanel').classList.toggle('open')}
function clearAll(){document.getElementById('query').value='';var c=document.getElementById('resultCard');c.className='result-card';document.getElementById('reportBody').innerHTML='';document.getElementById('metrics').innerHTML='';document.getElementById('btnDownload').style.display='none';document.getElementById('btnStop').style.display='none';clearCharts()}
function clearCharts(){_chartInstances.forEach(function(c){c.destroy()});_chartInstances=[]}

var SAMPLE=String.raw`# 中国新型储能市场分析报告

## 摘要
2025年中国新型储能新增装机106.3 GW，占全球48.3%，系统成本降至0.5-0.7元/Wh。

## SWOT 分析

| 优势 | 劣势 | 机会 | 威胁 |
|------|------|------|------|
| 规模全球第一，装机106.3 GW占全球48.3% | 安全风险突出，仅83%项目达设计容量 | 市场化机制深化，24省现货试点 | 钠离子电池产业化提速，百MWh级示范 |
| 成本全球最优，系统0.5-0.7元/Wh | 标准体系滞后，效率比标称值低8-12% | 海外订单366 GWh，同比增长144% | 地缘政治风险，欧盟拟强制碳足迹披露 |
| 政策支持罕见，2800项地方政策+211标准 | 回收体系薄弱，网络布局不合理 | 十五五列为新型电力系统关键支撑 | — |

## 中国光伏LCOE趋势（元/kWh）

| 年份 | 集中式 | 分布式 | 海上光伏 |
|------|--------|--------|----------|
| 2020 | 0.38 | 0.45 | 0.52 |
| 2021 | 0.35 | 0.42 | 0.49 |
| 2022 | 0.30 | 0.38 | 0.45 |
| 2023 | 0.26 | 0.34 | 0.41 |
| 2024 | 0.22 | 0.30 | 0.37 |
| 2025 | 0.19 | 0.27 | 0.33 |

## 储能技术LCOS对比（元/kWh）

| 技术路线 | 2023 | 2024 | 2025E |
|----------|------|------|-------|
| 磷酸铁锂 | 0.45 | 0.40 | 0.36 |
| 钠离子电池 | 0.65 | 0.55 | 0.44 |
| 全钒液流 | 0.85 | 0.78 | 0.70 |
| 压缩空气 | 0.72 | 0.65 | 0.58 |

## 参考来源
[1] CNESA 2025储能白皮书
[2] Wood Mackenzie 2025
[3] CPIA 2024年度报告`;

async function stopResearch(){if(!_currentTaskId)return;_aborted=true;try{await fetch(API+'/stop/'+_currentTaskId,{method:'POST'})}catch(e){}document.getElementById('resultCard').className='result-card error';document.getElementById('spinner').style.display='none';document.getElementById('statusText').textContent='已中断';document.getElementById('reportBody').innerHTML='<div class="error-msg">研究已中断</div>';document.getElementById('btnSubmit').disabled=false;document.getElementById('btnStop').style.display='none'}

function formatCheck(){
  _queryText='输出案例';_reportText=SAMPLE;
  document.getElementById('resultCard').className='result-card done';
  document.getElementById('spinner').style.display='none';
  document.getElementById('statusText').textContent='输出案例';document.getElementById('statusText').className='status-text status-done';
  document.getElementById('reportBody').innerHTML=renderMarkdown(SAMPLE);
  renderMathInElement(document.getElementById('reportBody'));
  renderCharts();
  document.getElementById('metrics').innerHTML='<div class="metric"><div class="value">4</div><div class="label">表格图表</div></div>';
  document.getElementById('btnDownload').style.display='inline-flex';
}

function getSettings(){
  return {web_search:!!document.getElementById('webSearch').checked,enable_ner:!!document.getElementById('sNER').checked,enable_qe:!!document.getElementById('sQE').checked,enable_bm25:!!document.getElementById('sBM25').checked,enable_reranker:!!document.getElementById('sRerank').checked,enable_pypsa:!!document.getElementById('sPyPSA').checked,enable_review:!!document.getElementById('sReview').checked,fast_review:!!document.getElementById('sFastReview').checked,enable_lora:!!document.getElementById('sLoRA').checked,enable_replan:!!document.getElementById('sReplan').checked}
}

async function submitResearch(){
  var q=document.getElementById('query').value.trim();if(!q)return;
  var card=document.getElementById('resultCard'),btn=document.getElementById('btnSubmit'),sp=document.getElementById('spinner'),st=document.getElementById('statusText'),rb=document.getElementById('reportBody'),md=document.getElementById('metrics'),dl=document.getElementById('btnDownload');
  card.className='result-card loading';rb.innerHTML='';md.innerHTML='';dl.style.display='none';btn.disabled=true;document.getElementById('btnStop').style.display='inline-flex';
  _aborted=false;_queryText=q;_reportText='';clearCharts();
  st.textContent='正在准备...';
  var cfg=getSettings();
  try{
    var res=await fetch(API+'/research',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,settings:cfg})});
    if(!res.ok)throw new Error('服务不可用');
    var t=await res.json();_currentTaskId=t.task_id;
    for(var i=0;i<360&&!_aborted;i++){await sleep(3000);
      var s=await fetch(API+'/status/'+t.task_id).then(function(r){return r.json()});
      if(s.progress&&s.progress!==st.textContent)st.textContent=s.progress;
      if(s.status==='completed'){
        var rep=await fetch(API+'/report/'+t.task_id).then(function(r){return r.json()});
        _reportText=rep.report||'';
        card.className='result-card done';sp.style.display='none';st.textContent='完成';st.className='status-text status-done';
        rb.innerHTML=renderMarkdown(_reportText);renderMathInElement(rb);renderCharts();
        if(rep.citations)md.innerHTML='<div class="metric"><div class="value">'+rep.citations.length+'</div><div class="label">引用</div></div>';
        dl.style.display='inline-flex';btn.disabled=false;document.getElementById('btnStop').style.display='none';_currentTaskId=null;return
      }
      if(s.status==='failed')throw new Error(s.progress||'研究失败');
    }
    throw new Error('超时');
  }catch(e){if(_aborted)return;card.className='result-card error';sp.style.display='none';st.textContent='';rb.innerHTML='<div class="error-msg">'+e.message+'</div>';btn.disabled=false;document.getElementById('btnStop').style.display='none'}
}

function downloadReport(){if(!_reportText)return;var b=new Blob([_reportText],{type:'text/markdown;charset=utf-8'});var a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=(_queryText||'report').replace(/[^a-zA-Z0-9]/g,'_').substring(0,40)+'.md';a.click()}

function renderMarkdown(md){
  if(!md)return"<p>empty</p>";
  var blk=[];md=md.replace(/<table>[\s\S]*?<\/table>/gi,function(m){blk.push(m);return'\x00T'+(blk.length-1)+'\x00'});
  md=md.replace(/(<br\s*\/?>)/gi,function(m){blk.push(m);return'\x00T'+(blk.length-1)+'\x00'});
  var lines=md.split('\n'),o=[],r=[],hd=0;
  function ft(){if(r.length===0)return;var h='<table>';r.forEach(function(w,i){h+='<tr>';w.forEach(function(c){h+=(i===0&&hd?'<th>':'<td>')+c+(i===0&&hd?'</th>':'</td>')});h+='</tr>'});h+='</table>';o.push(h);r=[];hd=0}
  for(var i=0;i<lines.length;i++){var l=lines[i].trim();if(l.startsWith('|')&&l.endsWith('|')&&!/^\|[\s\-:|]+\|$/.test(l)){r.push(l.split('|').slice(1,-1).map(function(c){return c.trim()}))}else if(/^\|[\s\-:|]+\|$/.test(l)&&r.length===1){hd=1}else{ft();o.push(l)}}
  ft();var h=o.join('\n');
  h=h.replace(/&(?!amp;|lt;|gt;)/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  blk.forEach(function(b,j){h=h.split('\x00T'+j+'\x00').join(b)});
  h=h.replace(/&lt;table&gt;/g,'<table>').replace(/&lt;\/table&gt;/g,'</table>').replace(/&lt;tr&gt;/g,'<tr>').replace(/&lt;\/tr&gt;/g,'</tr>').replace(/&lt;th&gt;/g,'<th>').replace(/&lt;\/th&gt;/g,'</th>').replace(/&lt;td&gt;/g,'<td>').replace(/&lt;\/td&gt;/g,'</td>');
  h=h.replace(/^### (.+)$/gm,'<h3>$1</h3>').replace(/^## (.+)$/gm,'<h2>$1</h2>').replace(/^# (.+)$/gm,'<h1>$1</h1>');
  h=h.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/\*(.+?)\*/g,'<em>$1</em>').replace(/\[(\d+)\]/g,'<sup>[$1]</sup>');
  var ps=h.split(/(<table>[\s\S]*?<\/table>)/g),res='';
  ps.forEach(function(p){if(p.startsWith('<table>')){res+=p}else{var xs=p.split(/\n\n+/);xs.forEach(function(x){x=x.trim();if(x){x=x.replace(/\n/g,'<br>');res+='<p>'+x+'</p>'}})}});
  return res||'<p>empty</p>'
}

function renderCharts(){
  clearCharts();var el=document.getElementById('reportBody');if(!el)return;
  setTimeout(function(){
    var colors=['#38bdf8','#4ade80','#fbbf24','#f87171','#a78bfa','#fb923c','#2dd4bf','#f472b6'];
    var ts=el.querySelectorAll('table');
    ts.forEach(function(t,ti){
      var hd=[],rw=[];
      t.querySelectorAll('tr').forEach(function(tr,ri){var cs=[];tr.querySelectorAll('th,td').forEach(function(td){cs.push(td.textContent.trim())});if(ri===0)hd=cs;else if(cs.length===hd.length)rw.push(cs)});
      if(rw.length<2||hd.length<2||rw.length>12)return;
      var nm=[];for(var c=1;c<hd.length;c++){var vs=rw.map(function(r){return parseFloat(r[c])});if(vs.filter(function(v){return!isNaN(v)}).length>=rw.length*0.6)nm.push(c)}
      if(nm.length<2)return;
      var th='';var p=t.previousElementSibling;while(p&&!th){if(/^H[1-3]$/i.test(p.tagName))th=p.textContent.trim();p=p.previousElementSibling}
      var ds=nm.slice(0,4).map(function(ci,i){return{label:hd[ci],data:rw.map(function(r){return parseFloat(r[ci])||0}),backgroundColor:colors[i%colors.length]+'99',borderColor:colors[i%colors.length],borderWidth:1}});
      var id='c'+ti+'_'+Math.random().toString(36).slice(2);
      var w=document.createElement('div');w.style.cssText='margin:1.5rem 0;padding:1rem;background:var(--bg);border:1px solid var(--border);border-radius:12px';
      w.innerHTML='<div style="font-size:.85rem;color:var(--muted);margin-bottom:.5rem">'+(th||'数据图表')+'</div><canvas id="'+id+'" style="max-height:320px"></canvas><div style="display:flex;justify-content:space-between;font-size:.7rem;color:var(--muted);margin-top:.3rem"><span>X: '+hd[0]+'</span><span>Y: '+nm.map(function(c){return hd[c]}).join(' / ')+'</span></div>';
      t.parentNode.insertBefore(w,t.nextSibling);
      _chartInstances.push(new Chart(document.getElementById(id).getContext('2d'),{type:'bar',data:{labels:rw.map(function(r){return r[0]}),datasets:ds},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#94a3b8',font:{size:11}}}},scales:{x:{title:{display:true,text:hd[0],color:'#94a3b8'},ticks:{color:'#94a3b8',maxRotation:45,font:{size:10}}},y:{title:{display:true,text:nm.map(function(c){return hd[c]}).join(' / '),color:'#94a3b8'},ticks:{color:'#94a3b8',font:{size:10}}}}}}))
    })
  },200)
}

function sleep(ms){return new Promise(function(r){setTimeout(r,ms)})}
document.getElementById('query').addEventListener('keydown',function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submitResearch()}});