from flask import Flask, render_template_string, jsonify, request
from pathlib import Path
import pandas as pd, ast

app = Flask(__name__)

EXPERIMENTS_DIR = Path("experiments")
DEFAULT_EXP = "exp_001"

# ───────────────────────── HTML minimal (sans légende, sans gestion de statut « mort »)
HTML_TEMPLATE = r"""
<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8">
<title>🌱 Viz GA – {{ exp }}</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
 :root{
   --bg:#f5f6f8;--fg:#000;--card:#fff;--border:#ccc;--mut:#ffa500;--grid:#ddd;
 }
 .dark{
   --bg:#121212;--fg:#eee;--card:#1e1e1e;--border:#444;--mut:#ffb347;--grid:#444;
 }
 html,body{height:100%;}
 body{font-family:system-ui,-apple-system,sans-serif;margin:0;background:var(--bg);color:var(--fg);transition:.3s}
 header{display:flex;align-items:center;gap:12px;padding:16px 16px 8px}
 h1{font-size:20px;font-weight:700;margin:0}
 select,button{font-size:15px;padding:4px 6px;color:var(--fg);background:var(--card);border:1px solid var(--border);border-radius:4px}
 #bestRun{margin:8px 16px;padding:10px;border:1px solid var(--border);border-radius:6px;background:var(--card);min-width:160px}
 svg{width:100%;border:1px solid var(--border);background:var(--card);border-radius:8px}
 .tooltip{position:absolute;pointer-events:none;opacity:0;font-size:13px;background:var(--card);border:1px solid var(--border);border-radius:6px;padding:8px;box-shadow:0 2px 6px rgba(0,0,0,.12)}
 button#themeBtn{margin-left:auto;cursor:pointer}
</style></head><body>

<header>
  <h1>🌱 Évolution des hyperparamètres</h1>
  <label>Exp:</label>
  <select id="expSel"></select>
  <label>Métrique:</label>
  <select id="metricSel">
    <option value="fitness_score">Fitness</option>
    <option value="precision">Précision</option>
    <option value="recall">Rappel</option>
    <option value="mAP" selected>mAP50-95</option>
  </select>
  <button id="themeBtn">🌙 Thème</button>
</header>

<div id="bestRun"></div>
<svg id="viz"></svg>
<div class="tooltip" id="tt"></div>

<script>
const exps={{ exps|tojson }}, currentExp="{{ exp }}";
const sel=document.getElementById("expSel");
exps.forEach(e=>{const o=document.createElement("option");o.value=o.textContent=e;if(e===currentExp)o.selected=true;sel.appendChild(o);});
sel.addEventListener("change",()=>location.search="?exp="+sel.value);

document.getElementById("themeBtn").addEventListener("click",()=>document.body.classList.toggle("dark"));

const colorScale=d3.scaleSequential(d3.interpolateBlues);
let rawData=null,selectedMetric="mAP";

document.getElementById("metricSel").addEventListener("change",e=>{selectedMetric=e.target.value;draw(rawData);});
fetch("/data?exp="+currentExp).then(r=>r.json()).then(d=>{rawData=d;draw(d);});

function draw(raw){if(!raw)return;
  const svg=d3.select("#viz"),tt=d3.select("#tt"),bp=d3.select("#bestRun");
  const nodes=raw.nodes,links=raw.links;
  const vals=nodes.map(d=>d[selectedMetric]||0);
  if(selectedMetric==="fitness_score"){const ext=d3.extent(vals);colorScale.domain(ext[0]===ext[1]?[0,1]:ext);}else colorScale.domain([0,1]);

  /* Best run */
  const best=nodes.sort((a,b)=>b[selectedMetric]-a[selectedMetric])[0];
  bp.html(best?`<strong>🏆 Meilleur</strong><br>${best.label}<br>${selectedMetric}: ${best[selectedMetric].toFixed(3)}`:"&nbsp;");

  /* Layout grid */
  const byGen=d3.group(nodes,d=>d.gen),gens=Array.from(byGen.keys()).map(Number).sort((a,b)=>a-b),maxRows=d3.max(Array.from(byGen.values(),a=>a.length));
  const mT=40,mL=110,sX=260,sY=Math.max(40,Math.min(120,800/maxRows));
  const W=mL*2+sX*(gens.length-1),H=mT*2+sY*(maxRows-1);
  svg.attr("width",W).attr("height",H).selectAll("*").remove();
  gens.forEach(g=>byGen.get(g).forEach((d,i)=>{d.x=mL+g*sX;d.y=mT+i*sY;}));
  gens.forEach(g=>{const x=mL+g*sX;svg.append("line").attr("x1",x).attr("y1",10).attr("x2",x).attr("y2",H-10).attr("stroke",getComputedStyle(document.body).getPropertyValue("--grid"));svg.append("text").attr("x",x).attr("y",24).attr("text-anchor","middle").attr("font-size","13px").text("Gen "+g);});

  /* Parent links */
  const byId=new Map(nodes.map(d=>[d.id,d]));
  svg.append("g").selectAll("line").data(links).join("line")
      .attr("x1",d=>byId.get(d.source)?.x).attr("y1",d=>byId.get(d.source)?.y)
      .attr("x2",d=>byId.get(d.target)?.x).attr("y2",d=>byId.get(d.target)?.y)
      .attr("stroke",getComputedStyle(document.body).getPropertyValue("--fg"))
      .attr("stroke-width",2).attr("stroke-dasharray","10,4");

  /* Nodes */
  const gNode=svg.append("g").selectAll("g").data(nodes).join("g")
      .attr("transform",d=>`translate(${d.x},${d.y})`)
      .on("mouseover",(e,d)=>showTip(e,d)).on("mouseout",hideTip);
  gNode.append("circle").attr("r",10)
      .attr("fill",d=>colorScale(d[selectedMetric]||0))
      .attr("stroke",d=>d.mutated?getComputedStyle(document.body).getPropertyValue("--mut"):getComputedStyle(document.body).getPropertyValue("--fg"))
      .attr("stroke-width",d=>d.mutated?3:2);
  gNode.append("text").attr("x",14).attr("y",4).style("font-size","11px").text(d=>d.label);

  /* Tooltip */
  function showTip(evt,d){
    const prm=Object.entries(d.params||{}).map(([k,v])=>`${k}: ${v}`).join("<br>");
    const mut=d.mutated?"<br><u>Mutations</u><br>"+Object.entries(d.mutation_details||{}).map(([k,v])=>`${k}: ${v.after} (av: ${v.before})`).join("<br>"):"";
    const label=document.querySelector("#metricSel option:checked").textContent;
    tt.html(`<strong>${d.label}</strong><br>Génération : ${d.gen}<br>${label} : ${(d[selectedMetric]||0).toFixed(3)}<br><br><u>Paramètres</u><br>${prm}${mut}`)
      .style("left",(evt.pageX+15)+"px").style("top",(evt.pageY-40)+"px").transition().duration(150).style("opacity",.97);
  }
  function hideTip(){tt.transition().duration(250).style("opacity",0);}  
}
</script></body></html>
"""

# ───────────────────────── Routes
@app.route("/")
def index():
    exp=request.args.get("exp", DEFAULT_EXP)
    exps = sorted(p.name for p in EXPERIMENTS_DIR.iterdir() if p.is_dir())
    return render_template_string(HTML_TEMPLATE, exp=exp, exps=exps)

@app.route("/data")
def data():
    exp = request.args.get("exp", DEFAULT_EXP)
    csv_path = EXPERIMENTS_DIR / exp / "summary.csv"
    if not csv_path.exists():
        return jsonify({"nodes": [], "links": []})

    df = pd.read_csv(csv_path)
    # Dernier enregistrement par (run_id, generation)
    df = df.sort_values("timestamp").groupby(["run_id", "generation"]).tail(1)

    nodes, links = [], []
    for _, r in df.iterrows():
        rid, gen = int(r.run_id), int(r.generation)
        nid = f"{rid}_{gen}"
        mutd = r.get("mutation_details", {})
        if isinstance(mutd, str):
            try:
                mutd = ast.literal_eval(mutd)
            except Exception:
                mutd = {}
        nodes.append({
            "id": nid,
            "label": r.run_name,
            "gen": gen,
            "fitness_score": float(r.get("fitness_score", 0)) if pd.notna(r.get("fitness_score")) else 0,
            "precision": float(r.get("precision", 0)) if pd.notna(r.get("precision")) else 0,
            "recall": float(r.get("recall", 0)) if pd.notna(r.get("recall")) else 0,
            "mAP": float(r.get("mAP_50:95", 0)) if pd.notna(r.get("mAP_50:95")) else 0,
            "mutation_details": mutd,
            "mutated": bool(r.get("mutated", False)),
            "params": {k: r[k] for k in ("lr0", "batch", "positive_ratio", "dataset_size", "imgsz", "epochs", "model") if k in r and pd.notna(r[k])}
        })

    id_map = {(int(r.run_id), int(r.generation)): f"{int(r.run_id)}_{int(r.generation)}" for _, r in df.iterrows()}

    # Links = parenthood only
    for _, r in df.iterrows():
        rid, gen = int(r.run_id), int(r.generation)
        child = id_map[(rid, gen)]
        for pc in ("parent1", "parent2"):
            if pd.notna(r.get(pc)):
                pk = (int(r[pc]), gen - 1)
                if pk in id_map:
                    links.append({"source": id_map[pk], "target": child, "type": "parent"})
    return jsonify({"nodes": nodes, "links": links})

if __name__ == "__main__":
    app.run(debug=True)
