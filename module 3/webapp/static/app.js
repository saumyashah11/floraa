(function(){
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var REPORT = null;

  function svg(tag, attrs, cls){
    var e = document.createElementNS(NS, tag);
    attrs = attrs || {};
    for(var k in attrs) e.setAttribute(k, attrs[k]);
    if(cls) e.setAttribute("class", cls);
    return e;
  }
  function txt(x,y,str,cls,attrs){
    var t = svg("text", Object.assign({x:x,y:y},attrs||{}), cls);
    t.textContent = str;
    return t;
  }
  function fmtPct(v){ return (Math.round(v*1000)/10) + "%"; }
  function fmtPct0(v){ return Math.round(v*100) + "%"; }
  function label(s){ return String(s).replace(/_/g,' '); }

  function addTooltip(container){
    container.style.position = "relative";
    var tip = document.createElement("div");
    tip.className = "chart-tooltip";
    container.appendChild(tip);
    return tip;
  }
  function showTip(tip, container, evt, html){
    var rect = container.getBoundingClientRect();
    tip.innerHTML = html;
    tip.style.left = (evt.clientX - rect.left) + "px";
    tip.style.top = (evt.clientY - rect.top) + "px";
    tip.classList.add("show");
  }
  function hideTip(tip){ tip.classList.remove("show"); }

  /* ================= TABS ================= */
  function initTabs(){
    var btns = document.querySelectorAll(".tab-btn");
    btns.forEach(function(btn){
      btn.addEventListener("click", function(){
        btns.forEach(function(b){b.classList.remove("active");});
        btn.classList.add("active");
        document.querySelectorAll(".panel").forEach(function(p){p.classList.remove("active");});
        document.getElementById("panel-"+btn.dataset.tab).classList.add("active");
        if(btn.dataset.tab === "followup") loadCases();
      });
    });
  }

  /* ================= ROUTE COMPARISON ================= */
  var ROUTES = ["image_only","text_only","fusion_confidence","fusion_gate","fusion_learned"];
  var ROUTE_LINES = {
    image_only:["Image","only"], text_only:["Text","only"],
    fusion_confidence:["Fusion","(conf.)"], fusion_gate:["Fusion","(proto)"],
    fusion_learned:["Fusion","(learned)"]
  };
  var ROUTE_LABEL = {image_only:"Image only", text_only:"Text only",
    fusion_confidence:"Fusion (confidence)", fusion_gate:"Fusion (prototype)",
    fusion_learned:"Fusion (learned)"};

  function renderRouteChart(){
    var el = document.getElementById("routeChart");
    el.innerHTML = "";
    var data = REPORT.route_comparison.filter(function(d){return d.protocol==="species_restricted";});
    var W=1040,H=300,padL=38,padR=16,padT=16,padB=54;
    var plotW=W-padL-padR, plotH=H-padT-padB;
    var s = svg("svg",{viewBox:"0 0 "+W+" "+H},"chart");
    var tip = addTooltip(el);
    [0,0.25,0.5,0.75,1.0].forEach(function(v){
      var y = padT+plotH*(1-v);
      s.appendChild(svg("line",{x1:padL,x2:padL+plotW,y1:y,y2:y},"grid-line"));
      s.appendChild(txt(padL-6,y+3,Math.round(v*100)+"%","axis-label",{"text-anchor":"end"}));
    });
    s.appendChild(svg("line",{x1:padL,x2:padL+plotW,y1:padT+plotH,y2:padT+plotH},"baseline"));

    var groupW = plotW/ROUTES.length, barW = groupW*0.30;
    ROUTES.forEach(function(route, ri){
      var gx = padL + ri*groupW + groupW/2;
      ["seen_test","zsl_unseen_test"].forEach(function(subset, si){
        var row = data.filter(function(r){return r.route===route && r.subset===subset;})[0];
        var v = row ? row.accuracy : 0;
        var bh = plotH*v;
        var bx = gx + (si===0 ? -barW-2 : 2);
        var by = padT+plotH-bh;
        var rect = svg("rect",{x:bx,y:by,width:barW,height:Math.max(bh,0.5),rx:3},
          (si===0?"s-seen":"s-unseen")+" hoverable");
        rect.addEventListener("mousemove", function(e){
          showTip(tip, el, e, "<b>"+ROUTE_LABEL[route]+"</b><br>"+(subset==="seen_test"?"seen test":"zsl unseen test")+": "+fmtPct(v));
        });
        rect.addEventListener("mouseleave", function(){hideTip(tip);});
        s.appendChild(rect);
        if(v>0.02) s.appendChild(txt(bx+barW/2, by-4, fmtPct0(v), "bar-label", {"text-anchor":"middle"}));
      });
      var lines = ROUTE_LINES[route];
      s.appendChild(txt(gx, H-38, lines[0], "axis-label", {"text-anchor":"middle"}));
      s.appendChild(txt(gx, H-27, lines[1], "axis-label", {"text-anchor":"middle"}));
    });
    el.appendChild(s);
    document.getElementById("routeLegend").innerHTML =
      '<span class="legend-item"><span class="legend-swatch bg-seen"></span>seen test</span>'+
      '<span class="legend-item"><span class="legend-swatch bg-unseen"></span>zsl unseen test</span>';
  }

  /* ================= ROBUSTNESS ================= */
  function renderRobustnessChart(){
    var el = document.getElementById("robustnessChart");
    el.innerHTML = "";
    var rows = REPORT.robustness_sweep;
    var W=520,H=300,padL=38,padR=14,padT=14,padB=32;
    var plotW=W-padL-padR, plotH=H-padT-padB, xMax=9;
    var series = [
      {key:"image_only_acc", cls:"s-image", label:"Image only", dash:null},
      {key:"fusion_confidence_acc", cls:"s-conf", label:"Fusion (confidence)", dash:"5,4"},
      {key:"fusion_gate_acc", cls:"s-gate", label:"Fusion (prototype)", dash:null},
      {key:"fusion_learned_acc", cls:"s-text", label:"Fusion (learned)", dash:null}
    ].filter(function(ser){return rows[0][ser.key]!==undefined;});
    var yMaxActual = Math.max.apply(null, rows.map(function(r){
      return Math.max.apply(null, series.map(function(s){return r[s.key];}));
    }));
    var yTop = Math.ceil(yMaxActual*20)/20 + 0.03;
    var s = svg("svg",{viewBox:"0 0 "+W+" "+H},"chart");
    var tip = addTooltip(el);
    function X(v){ return padL + (v/xMax)*plotW; }
    function Y(v){ return padT + plotH - (v/yTop)*plotH; }
    for(var i=0;i<=4;i++){
      var v=yTop*i/4, y=Y(v);
      s.appendChild(svg("line",{x1:padL,x2:padL+plotW,y1:y,y2:y},"grid-line"));
      s.appendChild(txt(padL-6,y+3,Math.round(v*100)+"%","axis-label",{"text-anchor":"end"}));
    }
    [0,1,2,3,4,6,9].forEach(function(v){
      s.appendChild(txt(X(v), H-8, "σ="+v, "axis-label", {"text-anchor":"middle"}));
    });
    series.forEach(function(ser){
      var d = "";
      rows.forEach(function(r,i){
        var x=X(r.noise_sigma), y=Y(r[ser.key]);
        d += (i===0?"M":"L")+x.toFixed(1)+","+y.toFixed(1)+" ";
      });
      var pathAttrs = {d:d, fill:"none","stroke-width": ser.dash?"2.2":"2.4"};
      if(ser.dash) pathAttrs["stroke-dasharray"] = ser.dash;
      s.appendChild(svg("path", pathAttrs, ser.cls));
      rows.forEach(function(r){
        var x=X(r.noise_sigma), y=Y(r[ser.key]);
        var c = svg("circle",{cx:x,cy:y,r:"3.2"}, ser.cls+" hoverable");
        c.addEventListener("mousemove", function(e){
          showTip(tip, el, e, "<b>"+ser.label+"</b><br>σ="+r.noise_sigma+": "+fmtPct(r[ser.key]));
        });
        c.addEventListener("mouseleave", function(){hideTip(tip);});
        s.appendChild(c);
      });
    });
    el.appendChild(s);
    var bgOf = {"s-image":"bg-image","s-conf":"bg-conf","s-gate":"bg-gate","s-text":"bg-text"};
    document.getElementById("robustnessLegend").innerHTML = series.map(function(ser){
      return '<span class="legend-item"><span class="legend-line '+bgOf[ser.cls]+(ser.dash?' dashed':'')+'"></span>'+ser.label+'</span>';
    }).join("");
  }

  /* ================= ATTRIBUTE ACCURACY ================= */
  function renderAttrChart(){
    var el = document.getElementById("attrChart");
    el.innerHTML = "";
    var acc = REPORT.attr_accuracy;
    var heads = Object.keys(acc).filter(function(k){return k!=="coverage_mae";});
    document.getElementById("covMae").textContent = acc.coverage_mae.toFixed(3);
    var W=520,H=290,padL=34,padR=10,padT=14,padB=50;
    var plotW=W-padL-padR, plotH=H-padT-padB;
    var s = svg("svg",{viewBox:"0 0 "+W+" "+H},"chart");
    var tip = addTooltip(el);
    [0,0.25,0.5,0.75,1].forEach(function(v){
      var y=padT+plotH*(1-v);
      s.appendChild(svg("line",{x1:padL,x2:padL+plotW,y1:y,y2:y},"grid-line"));
      s.appendChild(txt(padL-6,y+3,Math.round(v*100)+"%","axis-label",{"text-anchor":"end"}));
    });
    var bw = (plotW/heads.length)*0.5;
    heads.forEach(function(h,i){
      var cx = padL + (i+0.5)*(plotW/heads.length);
      var v = acc[h];
      var bh = plotH*v, bx=cx-bw/2, by=padT+plotH-bh;
      var rect = svg("rect",{x:bx,y:by,width:bw,height:Math.max(bh,0.5),rx:3},"s-image hoverable");
      rect.addEventListener("mousemove", function(e){ showTip(tip, el, e, "<b>"+h+"</b>: "+fmtPct(v)); });
      rect.addEventListener("mouseleave", function(){hideTip(tip);});
      s.appendChild(rect);
      s.appendChild(txt(cx,by-6,fmtPct0(v),"bar-label",{"text-anchor":"middle"}));
      var lbl = txt(cx, H-30, h, "axis-label", {"text-anchor":"end"});
      lbl.setAttribute("transform","rotate(-32 "+cx+" "+(H-30)+")");
      s.appendChild(lbl);
    });
    el.appendChild(s);
  }

  /* ================= CORPUS COMPOSITION ================= */
  function renderCorpusChart(){
    var el = document.getElementById("corpusChart");
    el.innerHTML = "";
    document.getElementById("corpusTotal").textContent = REPORT.corpus_total.toLocaleString();
    var counts = REPORT.corpus_per_class;
    var entries = Object.keys(counts).map(function(k){return [k,counts[k]];}).sort(function(a,b){return b[1]-a[1];});
    var W=520, rowH=10.5, gap=3, padL=142, padR=34, padT=6;
    var listH = entries.length*(rowH+gap);
    var H = padT + listH + 40;
    var plotW = W-padL-padR;
    var maxV = Math.max.apply(null, entries.map(function(e){return e[1];}));
    var s = svg("svg",{viewBox:"0 0 "+W+" "+H},"chart");
    var tip = addTooltip(el);
    entries.forEach(function(e,i){
      var cls=e[0], v=e[1];
      var y = padT + i*(rowH+gap);
      var bw = plotW*(v/maxV);
      s.appendChild(txt(padL-8, y+rowH-1.5, label(cls), "axis-label", {"text-anchor":"end"}));
      var rect = svg("rect",{x:padL,y:y,width:Math.max(bw,1),height:rowH,rx:2},"s-image hoverable");
      rect.addEventListener("mousemove", function(e2){ showTip(tip, el, e2, "<b>"+cls+"</b>: "+v+" queries"); });
      rect.addEventListener("mouseleave", function(){hideTip(tip);});
      s.appendChild(rect);
    });
    var my = padT + listH + 16;
    var mix = REPORT.corpus_script_mix, total = REPORT.corpus_total;
    var mixOrder = [["mixed_script","bg-mix"],["latin_hinglish","bg-latin"],["devanagari","bg-deva"]];
    s.appendChild(txt(padL, my-6, "script mix of the corpus", "axis-label", {}));
    var cx = padL;
    mixOrder.forEach(function(m){
      var v = mix[m[0]]||0, w = plotW*(v/total);
      var rect = svg("rect",{x:cx,y:my,width:Math.max(w,0.5),height:14},m[1]+" hoverable");
      rect.addEventListener("mousemove", function(e2){ showTip(tip, el, e2, "<b>"+label(m[0])+"</b>: "+Math.round(v/total*100)+"% ("+v+")"); });
      rect.addEventListener("mouseleave", function(){hideTip(tip);});
      s.appendChild(rect);
      cx += w;
    });
    el.appendChild(s);
    document.getElementById("corpusLegend").innerHTML = mixOrder.map(function(m){
      return '<span class="legend-item"><span class="legend-swatch '+m[1]+'"></span>'+label(m[0])+'</span>';
    }).join("");
  }

  /* ================= GATE WEIGHT HISTOGRAM ================= */
  function renderGateChart(){
    var el = document.getElementById("gateChart");
    el.innerHTML = "";
    var seen = REPORT.gate_weight_hist.seen_test, zsl = REPORT.gate_weight_hist.zsl_unseen_test;
    var n = seen.counts.length;
    var W=520,H=290,padL=30,padR=10,padT=14,padB=30;
    var plotW=W-padL-padR, plotH=H-padT-padB;
    var maxV = Math.max.apply(null, seen.counts.concat(zsl.counts));
    var s = svg("svg",{viewBox:"0 0 "+W+" "+H},"chart");
    var tip = addTooltip(el);
    [0,0.25,0.5,0.75,1].forEach(function(f){
      var y=padT+plotH*(1-f);
      s.appendChild(svg("line",{x1:padL,x2:padL+plotW,y1:y,y2:y},"grid-line"));
      s.appendChild(txt(padL-5,y+3,Math.round(maxV*f),"axis-label",{"text-anchor":"end"}));
    });
    var groupW = plotW/n, barW = groupW*0.36;
    for(var i=0;i<n;i++){
      var x0 = padL+i*groupW;
      var loE = seen.edges[i], hiE = seen.edges[i+1];
      [["seen","s-seen",seen.counts[i]],["zsl unseen","s-unseen",zsl.counts[i]]].forEach(function(t,j){
        var name=t[0], cls=t[1], v=t[2];
        var bh = plotH*(v/maxV);
        var bx = x0 + groupW/2 + (j===0? -barW-1:1);
        var by = padT+plotH-bh;
        var rect = svg("rect",{x:bx,y:by,width:barW,height:Math.max(bh,0),rx:2},cls+" hoverable");
        rect.addEventListener("mousemove", function(e){
          showTip(tip, el, e, "<b>"+name+"</b><br>weight "+loE.toFixed(2)+"–"+hiE.toFixed(2)+": "+v);
        });
        rect.addEventListener("mouseleave", function(){hideTip(tip);});
        s.appendChild(rect);
      });
      if(i%3===0) s.appendChild(txt(x0+groupW/2, H-10, loE.toFixed(1), "axis-label", {"text-anchor":"middle"}));
    }
    el.appendChild(s);
    document.getElementById("gateLegend").innerHTML =
      '<span class="legend-item"><span class="legend-swatch bg-seen"></span>seen test (n='+seen.n+')</span>'+
      '<span class="legend-item"><span class="legend-swatch bg-unseen"></span>zsl unseen test (n='+zsl.n+')</span>';
  }

  /* ================= CONFUSION MATRICES ================= */
  function seqClass(v){
    if(v<=0) return "seq-0"; if(v<0.15) return "seq-1"; if(v<0.35) return "seq-2";
    if(v<0.6) return "seq-3"; if(v<0.85) return "seq-4"; return "seq-5";
  }
  function renderConfusionPanel(container, matrix, classes, title){
    var n = classes.length, cell = 22;
    var padL=112, padT=24, padR=6, padB=112;
    var W = padL+n*cell+padR, H = padT+n*cell+padB;
    var s = svg("svg",{viewBox:"0 0 "+W+" "+H},"chart");
    var tip = addTooltip(container);
    s.appendChild(txt(padL, 12, title, "axis-label", {"font-weight":"700","font-size":"12px", fill:"var(--ink)"}));
    var rowSums = matrix.map(function(row){return row.reduce(function(a,b){return a+b;},0);});
    for(var r=0;r<n;r++){
      s.appendChild(txt(padL-6, padT+r*cell+cell*0.68, label(classes[r]), "axis-label", {"text-anchor":"end"}));
      for(var c=0;c<n;c++){
        var v = matrix[r][c];
        var norm = rowSums[r]>0 ? v/rowSums[r] : 0;
        var x = padL+c*cell, y = padT+r*cell;
        var rect = svg("rect",{x:x,y:y,width:cell-1.5,height:cell-1.5,rx:2}, seqClass(norm)+" hoverable");
        (function(rr,cc,vv){
          rect.addEventListener("mousemove", function(e){
            showTip(tip, container, e, "true <b>"+classes[rr]+"</b><br>pred <b>"+classes[cc]+"</b>: "+vv);
          });
        })(r,c,v);
        rect.addEventListener("mouseleave", function(){hideTip(tip);});
        s.appendChild(rect);
        if(v>0) s.appendChild(txt(x+cell/2-0.75, y+cell*0.66, v, norm>0.55?"cell-text-hi":"cell-text-lo", {"text-anchor":"middle","font-size":"8.5px"}));
      }
    }
    for(var c2=0;c2<n;c2++){
      var x2 = padL+c2*cell+cell/2, ty = padT+n*cell+10;
      var t = txt(x2, ty, label(classes[c2]), "axis-label", {"text-anchor":"end"});
      t.setAttribute("transform","rotate(-60 "+x2+" "+ty+")");
      s.appendChild(t);
    }
    container.appendChild(s);
  }
  function renderConfusion(){
    var wrap = document.getElementById("confusionWrap");
    wrap.innerHTML = "";
    var conf = REPORT.confusion;
    var panels = [["image_only","Image only"], ["fusion_learned","Fusion (learned)"], ["fusion_gate","Fusion (prototype)"]]
      .filter(function(p){return conf[p[0]];});
    wrap.style.gridTemplateColumns = "repeat("+panels.length+", minmax(260px, 1fr))";
    panels.forEach(function(p){
      var c = document.createElement("div");
      wrap.appendChild(c);
      renderConfusionPanel(c, conf[p[0]], conf.classes, p[1]);
    });
  }

  /* ================= GALLERY ================= */
  function renderGallery(){
    var tbody = document.querySelector("#galleryTable tbody");
    var rows = REPORT.qualitative_examples;
    function pill(ok, text){
      return '<span class="pill '+(ok?'good':'bad')+'"><span class="ic">'+(ok?'✓':'✕')+'</span>'+label(text)+'</span>';
    }
    tbody.innerHTML = rows.map(function(r){
      return "<tr>"+
        '<td class="qtext">'+r.text+"</td>"+
        '<td class="cls-mono">'+label(r.true_label)+"</td>"+
        "<td>"+pill(r.image_correct, r.image_pred)+"</td>"+
        '<td class="cls-mono">'+label(r.text_pred)+"</td>"+
        "<td>"+pill(r.learned_correct!==undefined?r.learned_correct:r.fusion_correct, r.fusion_learned_pred||r.fusion_pred)+"</td>"+
        "<td>"+pill(r.gate_correct!==undefined?r.gate_correct:r.fusion_correct, r.fusion_gate_pred||r.fusion_pred)+"</td>"+
        '<td class="mono">'+r.image_weight_in_fusion.toFixed(2)+"</td>"+
        "</tr>";
    }).join("");
  }

  /* ================= DECODER ================= */
  var EXAMPLES_CACHE = null;
  var SYMPTOM_SCHEMA = {
    colour: ["not_applicable","chlorotic_yellow","necrotic_brown","black","rust_orange","white_powdery","water_soaked","purple_red"],
    margin: ["not_applicable","sharply_defined","diffuse_feathered","halo_bearing","angular_vein_limited"],
    distribution: ["not_applicable","marginal","interveinal","random_scattered","concentric_ringed","uniform_general"],
    texture: ["not_applicable","powdery","downy_fuzzy","velvety_sporulating","sunken_cankerous","raised_pustular","smooth_flat"],
    organ: ["leaf","petal","bud","sepal","stem"],
    severity: ["none","trace","mild","moderate","severe"],
    coverage: ["low","medium","high"],
    is_healthy: ["healthy","unhealthy"]
  };

  var SELECTED_SYMPTOMS = [];

  function buildSymptomQueryFromSelection(){
    var text = SELECTED_SYMPTOMS.map(function(item){
      return item.value;
    }).join(", ");
    var q = document.getElementById("queryInput");
    if (q.dataset.userTyped !== "true") {
      q.value = text;
    }
    return text;
  }

  function renderSelectedSymptoms(){
    var el = document.getElementById("selectedSymptoms");
    if (!SELECTED_SYMPTOMS.length) {
      el.innerHTML = '<span class="selected-empty">No symptom selected yet. Click the tags to build the query.</span>';
      return;
    }
    el.innerHTML = '<span class="selected-label">Chosen symptoms</span>' +
      SELECTED_SYMPTOMS.map(function(item){
        return '<span class="symptom-pill active" data-key="'+item.key+'" data-value="'+item.value+'">'+label(item.value)+'</span>';
      }).join("");
    el.querySelectorAll(".symptom-pill").forEach(function(pill){
      pill.addEventListener("click", function(){
        var key = pill.dataset.key;
        var value = pill.dataset.value;
        SELECTED_SYMPTOMS = SELECTED_SYMPTOMS.filter(function(item){ return !(item.key === key && item.value === value); });
        renderSymptomLexicon();
        renderSelectedSymptoms();
        buildSymptomQueryFromSelection();
      });
    });
  }

  function renderSymptomLexicon(){
    var el = document.getElementById("symptomLexicon");
    var groups = Object.keys(SYMPTOM_SCHEMA).map(function(key){
      var values = SYMPTOM_SCHEMA[key];
      return '<div class="symptom-group"><span class="title">'+label(key)+'</span><div class="values">'+
        values.map(function(v){
          var isActive = SELECTED_SYMPTOMS.some(function(item){ return item.key === key && item.value === v; });
          return '<span class="symptom-pill'+(isActive ? ' active' : '')+'" data-key="'+key+'" data-value="'+v+'">'+label(v)+'</span>';
        }).join("") +
        '</div></div>';
    }).join("");
    el.innerHTML = '<h3>Recorded plant symptoms</h3><div class="symptom-groups">'+groups+'</div>';
    el.querySelectorAll(".symptom-pill").forEach(function(pill){
      pill.addEventListener("click", function(){
        var key = pill.dataset.key;
        var value = pill.dataset.value;
        var exists = SELECTED_SYMPTOMS.some(function(item){ return item.key === key && item.value === value; });
        if (exists) {
          SELECTED_SYMPTOMS = SELECTED_SYMPTOMS.filter(function(item){ return !(item.key === key && item.value === value); });
        } else {
          SELECTED_SYMPTOMS.push({ key: key, value: value });
        }
        renderSymptomLexicon();
        renderSelectedSymptoms();
        buildSymptomQueryFromSelection();
      });
    });
  }

  var SELECTED_IMAGE_ID = null;
  var UPLOADED_FILE = null;
  var LAST_TREATMENT_RESULT = null;

  function renderPhotoSelection(){
    document.querySelectorAll(".photo-thumb").forEach(function(el){
      el.classList.toggle("active", +el.dataset.id === SELECTED_IMAGE_ID);
    });
    var box = document.getElementById("photoPreview");
    if(UPLOADED_FILE) return; // upload listener owns the preview in that case
    if(SELECTED_IMAGE_ID == null){ box.style.display = "none"; return; }
    var p = EXAMPLES_CACHE.photos.filter(function(x){return x.id===SELECTED_IMAGE_ID;})[0];
    if(!p) return;
    box.style.display = "flex";
    document.getElementById("photoPreviewImg").src = "/media/"+p.id;
    document.getElementById("photoPreviewMeta").innerHTML = "<b>"+label(p.label)+"</b>";
  }

  function renderPhotoChips(photos){
    var wrap = document.getElementById("photoChips");
    wrap.innerHTML = photos.map(function(p){
      return '<span class="photo-thumb" data-id="'+p.id+'" title="'+label(p.label)+' ('+label(p.species)+', '+p.split+')">'+
        '<img src="/media/'+p.id+'" loading="lazy" alt="'+label(p.label)+'"></span>';
    }).join("");
    wrap.querySelectorAll(".photo-thumb").forEach(function(el){
      el.addEventListener("click", function(){
        var id = +el.dataset.id;
        SELECTED_IMAGE_ID = (SELECTED_IMAGE_ID === id) ? null : id;
        if(SELECTED_IMAGE_ID !== null){
          UPLOADED_FILE = null;
          document.getElementById("imageUpload").value = "";
        }
        renderPhotoSelection();
      });
    });
  }

  function populatePickers(data){
    EXAMPLES_CACHE = data;
    var sp = document.getElementById("speciesSelect");
    data.species_list.forEach(function(s){
      var o = document.createElement("option"); o.value = s; o.textContent = label(s);
      sp.appendChild(o);
    });

    var chipRow = document.getElementById("exampleChips");
    chipRow.innerHTML = data.texts.map(function(t,i){
      var short = t.text.length>48 ? t.text.slice(0,48)+"…" : t.text;
      return '<span class="chip" data-i="'+i+'">'+short+'</span>';
    }).join("");
    chipRow.querySelectorAll(".chip").forEach(function(chip){
      chip.addEventListener("click", function(){
        var text = data.texts[+chip.dataset.i].text;
        document.getElementById("queryInput").value = text;
        document.getElementById("queryInput").dataset.userTyped = "true";
      });
    });

    renderPhotoChips(data.photos || []);
  }

  function routeBlock(title, items, cls){
    var bars = items.map(function(it){
      var pct = Math.max(Math.round(it.p*100),1);
      return '<div class="route-bar-row"><span class="cls-mono">'+label(it.cls)+
        '</span><span class="route-bar-track"><span class="route-bar-fill '+cls+'" style="width:'+pct+'%"></span></span>'+
        '<span class="val">'+pct+'%</span></div>';
    }).join("");
    return '<div class="route-block"><h4>'+title+'</h4>'+bars+'</div>';
  }

  function qualityWarningHtml(quality, whatLabel){
    if(!quality || !quality.issues || !quality.issues.length) return "";
    var items = quality.issues.map(function(i){ return '<li>'+i.message+'</li>'; }).join("");
    return '<div class="quality-warning"><b>Photo quality check:</b> '+whatLabel+
      ' may not be reliable for diagnosis.<ul>'+items+'</ul></div>';
  }

  function runDecode(){
    var text = document.getElementById("queryInput").value.trim();
    var card = document.getElementById("resultCard");
    if(!text){ card.innerHTML = '<div class="placeholder-note">Type a message first, or click symptom chips to build a query.</div>'; return; }
    var species = document.getElementById("speciesSelect").value;
    var btn = document.getElementById("diagnoseBtn");
    btn.disabled = true; btn.textContent = "Running MuRIL…";
    card.innerHTML = '<div class="placeholder-note">Running live inference…</div>';

    var payload = new FormData();
    payload.append("text", text);
    payload.append("species", species);
    if(SELECTED_IMAGE_ID != null) payload.append("image_id", SELECTED_IMAGE_ID);
    if(UPLOADED_FILE) payload.append("image", UPLOADED_FILE);

    fetch("/api/decode", {
      method:"POST",
      body: payload
    }).then(function(r){return r.json();}).then(function(res){
      btn.disabled=false; btn.textContent="Decode & diagnose";
      if(res.error){ card.innerHTML = '<div class="placeholder-note">'+res.error+'</div>'; return; }

      var chips = Object.keys(res.detected).map(function(h){
        var d = res.detected[h];
        return '<span class="attr-tag">'+h+': <b>'+label(d.value)+'</b> ('+Math.round(d.confidence*100)+'%)</span>';
      }).join("") || '<span class="attr-tag" style="color:var(--muted)">no strong symptom signal detected in the text</span>';
      var healthyChip = res.is_healthy_prob>0.5 ? '<span class="attr-tag">reads as: <b>healthy</b> ('+Math.round(res.is_healthy_prob*100)+'%)</span>' : "";
      var plantLine = res.species_known ? '<div class="summary-item"><span class="label">Plant</span><span class="value">'+label(res.species_used || species)+'</span></div>' : '<div class="summary-item"><span class="label">Plant</span><span class="value">Not specified</span></div>';
      var imageLine = '<div class="summary-item"><span class="label">Input mode</span><span class="value">'+
        (res.image_used ? "Photo + text (fused)" : "Symptom text only")+'</span></div>';
      var gateLine = res.image_used ? ' &middot; image weight in fusion: '+Math.round(res.gate_image_weight*100)+'%' : "";
      var severityLabels = SYMPTOM_SCHEMA.severity;
      var severityText = (res.severity!==undefined && severityLabels[res.severity]!==undefined)
        ? label(severityLabels[res.severity]) : "unknown";

      var learnedAvailable = !!(res.routes.fusion_learned && res.routes.fusion_learned.length);
      var topList = learnedAvailable ? res.routes.fusion_learned : res.routes.fusion_prototype;
      var top = topList[0];
      var others = topList.slice(1, 3);
      var photoNoteHtml = res.photo_note ? '<div class="upload-help">'+res.photo_note+'</div>' : "";
      var qualityHtml = qualityWarningHtml(res.photo_quality, "the selected photo") +
        qualityWarningHtml(res.uploaded_image && res.uploaded_image.quality, "your uploaded photo");

      card.innerHTML =
        '<div><label class="field-label">Detected symptoms</label>'+
        '<div class="tag-row">'+chips+healthyChip+'</div></div>'+
        photoNoteHtml+
        qualityHtml+
        '<div class="summary-panel"><div class="summary-grid">'+plantLine+imageLine+'</div>'+
        '<div class="gate-readout">Severity: <b>'+severityText+'</b> &middot; health score: '+Math.round(res.is_healthy_prob*100)+'% &middot; coverage: '+Math.round(res.coverage*100)+'%'+gateLine+'</div>'+
        '</div>'+
        '<div class="diag-final"><div class="dl">Most likely</div>'+
        '<div class="dv">'+label(top.cls)+'</div>'+
        '<div class="dv-pct">'+Math.round(top.p*100)+'% likely</div>'+
        '</div>'+
        (others.length ? routeBlock("Other possibilities", others, learnedAvailable?"learned":"proto") : "")+
        '<div class="treatment-block">'+
        '<div class="treatment-row">'+
        '<button class="go-btn secondary" id="treatmentBtn" data-disease="'+top.cls+'" data-confidence="'+top.p+'">Get treatment advice for '+label(top.cls)+'</button>'+
        '<select id="treatmentLang" class="select lang-select" title="Output language">'+
        '<option value="en">English</option>'+
        '<option value="hi">हिन्दी (Hindi)</option>'+
        '</select>'+
        '</div>'+
        '<div id="treatmentResult"></div>'+
        '</div>';
    }).catch(function(err){
      btn.disabled=false; btn.textContent="Decode & diagnose";
      card.innerHTML = '<div class="placeholder-note">request failed: '+err+'</div>';
    });
  }

  /* ================= TREATMENT ADVICE (MODULE 4 RAG) ================= */
  function fetchTreatment(disease, btn){
    var out = document.getElementById("treatmentResult");
    btn.disabled = true; btn.textContent = "Getting treatment advice…";
    out.innerHTML = '<div class="placeholder-note">Looking up treatment advice&hellip;</div>';

    var followUp = document.getElementById("queryInput").value.trim();
    var confidence = parseFloat(btn.dataset.confidence);
    var langSel = document.getElementById("treatmentLang");
    var language = langSel ? langSel.value : "en";
    if(language === "hi"){
      btn.textContent = "Getting advice & translating…";
    }
    fetch("/api/treatment", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({disease: disease, query: followUp, gen_model: "qwen1.5b",
                            confidence: isNaN(confidence) ? null : confidence,
                            language: language})
    }).then(function(r){return r.json();}).then(function(res){
      btn.disabled = false; btn.textContent = "Get treatment advice for " + label(disease);
      if(res.error){ out.innerHTML = '<div class="placeholder-note">'+res.error+'</div>'; return; }
      LAST_TREATMENT_RESULT = res;
      out.innerHTML = treatmentAnswerHtml(res) +
        '<button class="go-btn secondary" id="logCaseBtn">Save to follow-up log</button>'+
        '<div id="logCaseMsg"></div>';
    }).catch(function(err){
      btn.disabled = false; btn.textContent = "Get treatment advice for " + label(disease);
      out.innerHTML = '<div class="placeholder-note">request failed: '+err+'</div>';
    });
  }

  /* Renders a RAG answer (treatment advice or a follow-up Q&A) with only
     what a farmer needs to act on it -- no module names, file names, or
     model internals. Sourcing is one quiet, collapsed disclosure. */
  function treatmentAnswerHtml(res){
    var passageHtml = (res.passages||[]).map(function(p){
      return '<div class="treatment-passage"><span class="attr-tag">'+p.section+'</span> '+p.text+'</div>';
    }).join("");
    var seenUrls = {};
    var citeHtml = (res.passages||[]).reduce(function(acc, p){
      (p.sources||[]).forEach(function(s){
        if(seenUrls[s.url]) return;
        seenUrls[s.url] = true;
        acc.push('<li><a href="'+s.url+'" target="_blank" rel="noopener">'+s.title+'</a></li>');
      });
      return acc;
    }, []).join("");
    var alikeHtml = (res.look_alikes||[]).map(function(n){
      return '<div class="treatment-passage"><span class="attr-tag">look-alike</span> '+
        n.look_alike_common_name+': '+n.look_alike_symptoms+'</div>';
    }).join("");
    var alikeBlock = alikeHtml
      ? '<details class="treatment-sources" open><summary>Could also be one of these look-alikes</summary>'+alikeHtml+'</details>'
      : "";
    var originalBlock = (res.language === "hi" && res.answer_en)
      ? '<details class="treatment-sources"><summary>View in English</summary>'+
        '<div class="treatment-passage">'+res.answer_en+'</div></details>'
      : "";
    var sourcesInner = (passageHtml || citeHtml)
      ? passageHtml + (citeHtml ? '<ul class="source-list">'+citeHtml+'</ul>' : "")
      : '<div class="treatment-passage">No reference material for this one.</div>';
    var sourcesBlock = (passageHtml || citeHtml)
      ? '<details class="treatment-sources"><summary>Where this came from</summary>'+sourcesInner+'</details>'
      : "";
    return (
      '<div class="treatment-answer"><label class="field-label">'+
      (res.common_name ? label(res.common_name) : "Treatment advice")+'</label>'+
      '<p lang="'+res.language+'">'+res.answer.replace(/\n/g, "<br>")+'</p></div>'+
      alikeBlock+ originalBlock+ sourcesBlock
    );
  }

  /* ================= TREATMENT FOLLOW-UP LOG ================= */
  function logCurrentTreatment(btn){
    if(!LAST_TREATMENT_RESULT){ return; }
    var res = LAST_TREATMENT_RESULT;
    var species = document.getElementById("speciesSelect").value;
    var msg = document.getElementById("logCaseMsg");
    btn.disabled = true; btn.textContent = "Saving…";
    fetch("/api/cases", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        disease: res.disease, common_name: res.common_name,
        species: species, treatment_answer: res.answer, language: res.language
      })
    }).then(function(r){return r.json();}).then(function(created){
      btn.textContent = "Saved ✓";
      msg.innerHTML = '<div class="upload-help">Saved to your follow-up log. '+
        'Open the "Treatment follow-up" tab any time to check in on it.</div>';
    }).catch(function(err){
      btn.disabled = false; btn.textContent = "Save to follow-up log";
      msg.innerHTML = '<div class="placeholder-note">failed to save: '+err+'</div>';
    });
  }

  var CASES_CACHE = [];
  var STATUS_FILTER = "all";
  var STATUS_META = {
    better: {label:"Better", cls:"good", icon:"✓"},
    same:   {label:"No change", cls:"", icon:"•"},
    worse:  {label:"Worse", cls:"bad", icon:"✕"}
  };

  function loadCases(){
    var el = document.getElementById("casesList");
    fetch("/api/cases").then(function(r){return r.json();}).then(function(cases){
      CASES_CACHE = cases;
      renderCases();
    }).catch(function(err){
      el.innerHTML = '<div class="placeholder-note">couldn\'t load your log: '+err+'</div>';
    });
  }

  function latestStatus(c){
    var statuses = (c.updates||[]).filter(function(u){return u.kind==="status";});
    return statuses.length ? statuses[statuses.length-1] : null;
  }

  function relativeTime(iso){
    var d = new Date(iso), diff = (Date.now() - d.getTime())/1000;
    if(diff < 60) return "just now";
    if(diff < 3600) return Math.floor(diff/60)+"m ago";
    if(diff < 86400) return Math.floor(diff/3600)+"h ago";
    if(diff < 86400*30) return Math.floor(diff/86400)+"d ago";
    return d.toLocaleDateString();
  }

  function renderFilterBar(){
    var counts = {all: CASES_CACHE.length, better:0, same:0, worse:0, open:0};
    CASES_CACHE.forEach(function(c){
      var s = latestStatus(c);
      if(s) counts[s.result] = (counts[s.result]||0)+1; else counts.open++;
    });
    var defs = [["all","All"],["open","Awaiting check-in"],["better","Better"],["same","No change"],["worse","Worse"]];
    return '<div class="case-filter-bar">'+defs.map(function(d){
      var key = d[0];
      return '<button class="chip case-filter'+(STATUS_FILTER===key?' active':'')+'" data-filter="'+key+'">'+
        d[1]+' <span class="count">'+(counts[key]||0)+'</span></button>';
    }).join("")+'</div>';
  }

  function updateEntryHtml(u){
    if(u.kind === "status"){
      var meta = STATUS_META[u.result] || {label:u.result, cls:"", icon:"•"};
      return '<div class="timeline-entry status '+meta.cls+'">'+
        '<span class="timeline-icon">'+meta.icon+'</span>'+
        '<div class="timeline-body"><span class="timeline-kind">Marked '+meta.label.toLowerCase()+'</span>'+
        (u.text ? '<p class="timeline-text">'+u.text+'</p>' : '')+
        '<span class="timeline-time">'+relativeTime(u.created_at)+'</span></div></div>';
    }
    if(u.kind === "question"){
      return '<div class="timeline-entry question">'+
        '<span class="timeline-icon">?</span>'+
        '<div class="timeline-body"><span class="timeline-kind">Asked</span>'+
        '<p class="timeline-text timeline-question">'+u.text+'</p>'+
        '<p class="timeline-text">'+(u.answer||"").replace(/\n/g,"<br>")+'</p>'+
        '<span class="timeline-time">'+relativeTime(u.created_at)+'</span></div></div>';
    }
    return '<div class="timeline-entry comment">'+
      '<span class="timeline-icon">&#9998;</span>'+
      '<div class="timeline-body"><span class="timeline-kind">Note</span>'+
      '<p class="timeline-text">'+u.text+'</p>'+
      '<span class="timeline-time">'+relativeTime(u.created_at)+'</span></div></div>';
  }

  function caseCardHtml(c){
    var status = latestStatus(c);
    var statusBadge = status
      ? '<span class="truth-pill '+(STATUS_META[status.result]||{}).cls+'">'+(STATUS_META[status.result]||{}).label+'</span>'
      : '<span class="truth-pill pending">Awaiting check-in</span>';
    var timeline = (c.updates||[]).map(updateEntryHtml).join("");
    return '<div class="case-card" data-case-id="'+c.id+'">'+
      '<div class="case-head">'+
        '<div class="case-head-main"><b>'+label(c.common_name || c.disease)+'</b>'+
          (c.species ? ' <span class="case-species">&middot; '+label(c.species)+'</span>' : '')+'</div>'+
        '<div class="case-head-side">'+statusBadge+
          '<span class="case-date">logged '+relativeTime(c.created_at)+'</span></div>'+
      '</div>'+
      (c.treatment_answer
        ? '<details class="treatment-sources case-advice"><summary>Original advice</summary>'+
          '<div class="treatment-passage">'+String(c.treatment_answer).replace(/\n/g,"<br>")+'</div></details>'
        : '')+
      (timeline ? '<div class="case-timeline">'+timeline+'</div>' : '')+
      '<div class="case-actions">'+
        '<div class="case-action-row">'+
          '<span class="case-action-label">How is it doing?</span>'+
          '<div class="chip-row">'+
            '<button class="chip status-btn" data-result="better">Better</button>'+
            '<button class="chip status-btn" data-result="same">No change</button>'+
            '<button class="chip status-btn" data-result="worse">Worse</button>'+
          '</div>'+
        '</div>'+
        '<form class="case-comment-form">'+
          '<input type="text" class="case-input comment-input" placeholder="Add a note or update&hellip;" maxlength="400">'+
          '<button type="submit" class="chip-submit">Add</button>'+
        '</form>'+
        '<form class="case-ask-form">'+
          '<input type="text" class="case-input ask-input" placeholder="Ask a follow-up question about this case&hellip;" maxlength="400">'+
          '<button type="submit" class="chip-submit accent">Ask</button>'+
        '</form>'+
        '<div class="case-action-msg"></div>'+
      '</div>'+
      '</div>';
  }

  function renderCases(){
    var el = document.getElementById("casesList");
    if(!CASES_CACHE.length){
      el.innerHTML = '<div class="placeholder-note">Nothing logged yet. After getting treatment '+
        'advice in the Symptom decoder tab, click "Save to follow-up log" to start tracking a case '+
        'here &mdash; you can check in on it, leave notes, and keep asking questions over time.</div>';
      return;
    }
    var visible = CASES_CACHE.filter(function(c){
      if(STATUS_FILTER === "all") return true;
      var s = latestStatus(c);
      if(STATUS_FILTER === "open") return !s;
      return s && s.result === STATUS_FILTER;
    });
    var body = visible.length
      ? '<div class="case-list">'+visible.map(caseCardHtml).join("")+'</div>'
      : '<div class="placeholder-note">No cases match this filter.</div>';
    el.innerHTML = renderFilterBar() + body;
  }

  function setCaseMsg(card, html, isError){
    var msg = card.querySelector(".case-action-msg");
    msg.innerHTML = html;
    msg.classList.toggle("is-error", !!isError);
  }

  /* Ticks a "still working" message with elapsed seconds so a slow (RAG)
     request reads as in-progress rather than stuck. Returns a stop function. */
  function startWaitTicker(card, baseText, slowAfterSecs){
    var msg = card.querySelector(".case-action-msg");
    var start = Date.now();
    function tick(){
      var secs = Math.round((Date.now() - start) / 1000);
      var extra = slowAfterSecs && secs >= slowAfterSecs
        ? " &mdash; still working, this can take up to a minute on this device" : "";
      msg.innerHTML = baseText + " (" + secs + "s)" + extra;
      msg.classList.remove("is-error");
    }
    tick();
    var id = setInterval(tick, 1000);
    return function(){ clearInterval(id); };
  }

  function reenable(){
    for(var i=0;i<arguments.length;i++) if(arguments[i]) arguments[i].disabled = false;
  }

  function submitFollowup(caseId, result, card){
    fetch("/api/cases/"+caseId+"/followup", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({result: result, notes: null})
    }).then(function(r){return r.json();}).then(function(updated){
      var i = CASES_CACHE.findIndex(function(c){return c.id===updated.id;});
      if(i>=0) CASES_CACHE[i] = updated;
      renderCases();
    }).catch(function(err){
      setCaseMsg(card, "couldn't save that check-in: "+err, true);
      card.querySelectorAll(".status-btn").forEach(function(b){b.disabled = false;});
    });
  }

  function submitComment(caseId, text, card, input, btn){
    var stop = startWaitTicker(card, "Saving");
    fetch("/api/cases/"+caseId+"/comments", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: text})
    }).then(function(r){return r.json();}).then(function(updated){
      stop();
      var i = CASES_CACHE.findIndex(function(c){return c.id===updated.id;});
      if(i>=0) CASES_CACHE[i] = updated;
      renderCases();
    }).catch(function(err){
      stop();
      setCaseMsg(card, "couldn't save that note: "+err, true);
      reenable(input, btn);
    });
  }

  function submitAsk(caseId, question, card, input, btn){
    var stop = startWaitTicker(card, "Getting your answer", 8);
    fetch("/api/cases/"+caseId+"/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: question})
    }).then(function(r){return r.json();}).then(function(res){
      stop();
      if(res.error){ setCaseMsg(card, res.error, true); reenable(input, btn); return; }
      var i = CASES_CACHE.findIndex(function(c){return c.id===res.case.id;});
      if(i>=0) CASES_CACHE[i] = res.case;
      renderCases();
    }).catch(function(err){
      stop();
      setCaseMsg(card, "couldn't get an answer: "+err, true);
      reenable(input, btn);
    });
  }

  /* ================= INIT ================= */
  function init(){
    initTabs();
    var casesList = document.getElementById("casesList");
    casesList.addEventListener("click", function(e){
      var filterBtn = e.target.closest(".case-filter");
      if(filterBtn){
        STATUS_FILTER = filterBtn.dataset.filter;
        renderCases();
        return;
      }
      var statusBtn = e.target.closest(".status-btn");
      if(statusBtn){
        var card = statusBtn.closest(".case-card");
        card.querySelectorAll(".status-btn").forEach(function(b){b.disabled = true;});
        submitFollowup(card.dataset.caseId, statusBtn.dataset.result, card);
      }
    });
    casesList.addEventListener("submit", function(e){
      var card = e.target.closest(".case-card");
      if(!card) return;
      if(e.target.classList.contains("case-comment-form")){
        e.preventDefault();
        var input = e.target.querySelector(".comment-input");
        var submitBtn = e.target.querySelector(".chip-submit");
        var text = input.value.trim();
        if(!text) return;
        input.disabled = true; submitBtn.disabled = true;
        submitComment(card.dataset.caseId, text, card, input, submitBtn);
      } else if(e.target.classList.contains("case-ask-form")){
        e.preventDefault();
        var aInput = e.target.querySelector(".ask-input");
        var askBtn = e.target.querySelector(".chip-submit");
        var question = aInput.value.trim();
        if(!question) return;
        aInput.disabled = true; askBtn.disabled = true;
        submitAsk(card.dataset.caseId, question, card, aInput, askBtn);
      }
    });
    renderSymptomLexicon();
    renderSelectedSymptoms();
    fetch("/api/report").then(function(r){return r.json();}).then(function(data){
      REPORT = data;
      renderRouteChart();
      renderRobustnessChart();
      renderAttrChart();
      renderCorpusChart();
      renderGateChart();
      renderConfusion();
      renderGallery();
    });
    fetch("/api/examples").then(function(r){return r.json();}).then(populatePickers);
    document.getElementById("diagnoseBtn").addEventListener("click", runDecode);
    document.getElementById("resultCard").addEventListener("click", function(e){
      var btn = e.target.closest("#treatmentBtn");
      if(btn) fetchTreatment(btn.dataset.disease, btn);
      var logBtn = e.target.closest("#logCaseBtn");
      if(logBtn) logCurrentTreatment(logBtn);
    });
    document.getElementById("queryInput").addEventListener("input", function(){
      this.dataset.userTyped = "true";
    });
    document.getElementById("imageUpload").addEventListener("change", function(){
      UPLOADED_FILE = this.files[0] || null;
      if(UPLOADED_FILE){
        SELECTED_IMAGE_ID = null;
        renderPhotoSelection();
        var box = document.getElementById("photoPreview");
        box.style.display = "flex";
        document.getElementById("photoPreviewImg").src = URL.createObjectURL(UPLOADED_FILE);
        document.getElementById("photoPreviewMeta").innerHTML =
          "<b>"+UPLOADED_FILE.name+"</b><br>uploaded &mdash; preview only, the live pipeline reads image evidence from the indexed dataset photos below, not from arbitrary uploads.";
      }
    });
  }
  document.addEventListener("DOMContentLoaded", init);
})();
