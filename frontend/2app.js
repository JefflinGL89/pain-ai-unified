const $=id=>document.getElementById(id);
Chart.defaults.color="#8290a2";Chart.defaults.borderColor="#202b36";
const charts={};

function makeChart(id,type,labels,datasets,options={}){
  const c=$(id);if(!c)return;
  charts[id]=new Chart(c,{type,data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:datasets.length>1}},scales:{x:{grid:{color:"#202b36"}},y:{grid:{color:"#202b36"},beginAtZero:true}},...options}});
}

makeChart("classChart","bar",["No Pain","Low Pain","Moderate","High Pain"],[
  {label:"True frames",data:[0,0,0,0],backgroundColor:"#65d9c8",borderRadius:5}
]);
makeChart("metricChart","bar",["Accuracy","Balanced Accuracy","Macro F1"],[
  {label:"Validation",data:[78.43,56.59,59.44],backgroundColor:"#65d9c8",borderRadius:5}
]);
makeChart("ordinalChart","bar",["Exact","1-level","2-level","3-level"],[
  {label:"Validation",data:[0,0,0,0],backgroundColor:"#65d9c8",borderRadius:5}
]);
makeChart("pspiChart","line",[],[
  {label:"Accuracy",data:[],borderColor:"#65d9c8",backgroundColor:"#65d9c8",tension:.35}
]);
makeChart("confidenceChart","bar",["Correct","Incorrect"],[
  {label:"Mean confidence",data:[0,0],backgroundColor:"#65d9c8",borderRadius:5}
],{scales:{x:{grid:{color:"#202b36"}},y:{grid:{color:"#202b36"},beginAtZero:true,max:1}}});
makeChart("subjectChart","bar",[],[
  {label:"Accuracy",data:[],backgroundColor:"#65d9c8",borderRadius:5}
]);
makeChart("historyChart","line",[],[
  {label:"Train loss",data:[],borderColor:"#65d9c8",backgroundColor:"#65d9c8",tension:.35},
  {label:"Validation loss",data:[],borderColor:"#8290a2",backgroundColor:"#8290a2",tension:.35}
]);
charts.historyChart.options.scales.y.beginAtZero=true;
makeChart("trajectoryChart","line",[],[
  {label:"Prediction confidence",data:[],borderColor:"#65d9c8",backgroundColor:"#65d9c8",tension:.35}
]);

function matrix(m){
  const root=$("matrix");root.innerHTML="";
  if(!m||!m.length){root.innerHTML="<div class='empty'>Confusion matrix data unavailable.</div>";return}
  const labels=["N","L","M","H"].slice(0,m.length),max=Math.max(...m.flat(),1);
  root.style.display="grid";root.style.gridTemplateColumns=`35px repeat(${m.length},1fr)`;root.style.gap="4px";
  ["",...labels].forEach(x=>{const d=document.createElement("div");d.textContent=x;d.style.padding="10px";d.style.color="#718094";root.appendChild(d)});
  m.forEach((row,i)=>{
    let h=document.createElement("div");h.textContent=labels[i];h.style.padding="10px";h.style.color="#718094";root.appendChild(h);
    row.forEach(v=>{let d=document.createElement("div");d.textContent=v;d.style.padding="12px";d.style.textAlign="center";d.style.background=`rgba(101,217,200,${Math.min(.7,.06+v/max)})`;root.appendChild(d)})
  });
}
matrix(null);

async function loadResults(){
  try{
    const r=await fetch("/api/results");const d=await r.json();
    if(d.metrics){
      if(d.metrics.precision!=null)$("precision").textContent=(d.metrics.precision*100).toFixed(2)+"%";
      if(d.metrics.recall!=null)$("recall").textContent=(d.metrics.recall*100).toFixed(2)+"%";
      if(d.metrics.mcc!=null)$("mcc").textContent=d.metrics.mcc.toFixed(4);
    }
    const cd=d.class_distribution;
    if(cd){
      charts.classChart.data.labels=cd.labels||charts.classChart.data.labels;
      charts.classChart.data.datasets[0].data=cd.true||[];
      charts.classChart.data.datasets[0].label="True frames";
      charts.classChart.update();
    }
    if(d.confusion_matrix){
      matrix(d.confusion_matrix.values||d.confusion_matrix);
    }
    if(d.ordinal_error){
      charts.ordinalChart.data.labels=d.ordinal_error.labels;
      charts.ordinalChart.data.datasets[0].data=(d.ordinal_error.values||[]).map(v=>v*100);
      charts.ordinalChart.update();
    }
    if(Array.isArray(d.pspi_analysis)){
      charts.pspiChart.data.labels=d.pspi_analysis.map(x=>String(x.pspi));
      charts.pspiChart.data.datasets[0].data=d.pspi_analysis.map(x=>x.accuracy*100);
      charts.pspiChart.update();
    }
    if(d.confidence_summary){
      charts.confidenceChart.data.labels=d.confidence_summary.labels;
      charts.confidenceChart.data.datasets[0].data=d.confidence_summary.mean;
      charts.confidenceChart.update();
    }
    if(Array.isArray(d.subject_analysis)){
      charts.subjectChart.data.labels=d.subject_analysis.map(x=>x.subject);
      charts.subjectChart.data.datasets[0].data=d.subject_analysis.map(x=>x.accuracy*100);
      charts.subjectChart.update();
    }
    if(Array.isArray(d.training_history)){
      charts.historyChart.data.labels=d.training_history.map(x=>String(x.epoch));
      charts.historyChart.data.datasets[0].data=d.training_history.map(x=>x.train_loss);
      charts.historyChart.data.datasets[1].data=d.training_history.map(x=>x.val_loss);
      charts.historyChart.update();
    }
  }catch(e){console.log("Analytics load failed:",e)}
}
loadResults();

$("upload").onclick=()=>$("file").click();
$("file").onchange=()=>{const f=$("file").files[0];if(!f)return;$("preview").src=URL.createObjectURL(f);predict(f)};

async function predict(file){
  $("label").textContent="Analysing…";$("conf").textContent="";
  const fd=new FormData();fd.append("file",file);
  try{
    const r=await fetch("/api/predict",{method:"POST",body:fd});const d=await r.json();
    if(!r.ok)throw Error(d.detail||"Inference failed");renderPrediction(d)
  }catch(e){$("label").textContent="Inference unavailable";$("pspi").textContent=e.message}
}
function renderPrediction(d){
  $("label").textContent=d.prediction.label;$("conf").textContent=d.prediction.confidence+"%";
  $("bar").style.width=d.prediction.confidence+"%";
  $("frames").textContent=d.frames_analyzed;$("duration").textContent=d.duration_seconds+" s";
  $("fps").textContent=d.sampled_fps;$("face").textContent=d.face_detection_rate+"%";
  $("pspi").textContent=d.pspi_message;
  const p=d.prediction.probabilities,names=["No Pain","Low Pain","Moderate","High Pain"];
  $("probs").innerHTML=names.map((n,i)=>`<div class="prob"><span>${n}</span><i><b style="width:${p[i]}%"></b></i><em>${p[i]}%</em></div>`).join("");
  const tr=d.trajectory||[];
  charts.trajectoryChart.data.labels=tr.map(x=>x.time+"s");
  charts.trajectoryChart.data.datasets[0].data=tr.map(x=>x.confidence);
  charts.trajectoryChart.update()
}
let stream=null;
$("camera").onclick=async()=>{
  try{
    if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;$("camera").textContent="Start Webcam";return}
    stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
    const v=document.createElement("video");v.autoplay=true;v.muted=true;v.srcObject=stream;
    v.style.width="100%";v.style.height="210px";v.style.objectFit="contain";
    $("webcam").innerHTML="";$("webcam").appendChild(v);$("camera").textContent="Stop Webcam";
    $("label").textContent="Webcam preview active";
  }catch(e){alert("Camera permission is unavailable: "+e.message)}
};
