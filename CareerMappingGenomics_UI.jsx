import { useState, useRef, useEffect } from "react";

// ─────────────────────────────────────────────
// DATA DEFINITIONS (mirrors the Python project)
// ─────────────────────────────────────────────

const BIG_FIVE = [
  { key:"O", name:"Openness",          desc:"Curiosity, creativity, appreciation for art & new ideas",        color:"#7C9EFF", asd:"↑ Often elevated in ASD" },
  { key:"C", name:"Conscientiousness", desc:"Organisation, discipline, goal-directed behaviour",               color:"#4ECDC4", asd:"↑ Slightly elevated in ASD" },
  { key:"E", name:"Extraversion",      desc:"Social energy, assertiveness, positive emotionality",             color:"#FFD166", asd:"↓ Often reduced in ASD" },
  { key:"A", name:"Agreeableness",     desc:"Cooperation, empathy, trust, prosocial orientation",              color:"#F8961E", asd:"↓ Often reduced in ASD" },
  { key:"N", name:"Neuroticism",       desc:"Emotional instability, anxiety, stress sensitivity",              color:"#FF6B6B", asd:"↑ Slightly elevated in ASD" },
];

const BIOMARKER_CATEGORIES = {
  "Hematology": {
    icon:"🩸", color:"#FF6B6B", importance:"medium",
    desc:"Basic blood cell composition. Abnormal values can indicate anaemia or infection affecting cognitive performance.",
    markers:[
      { key:"hemoglobin_g_dl",    label:"Hemoglobin",    unit:"g/dL",   normal:[12,17],   asd_note:null },
      { key:"wbc_count_k_ul",     label:"WBC Count",     unit:"K/µL",   normal:[4,11],    asd_note:null },
      { key:"platelet_count_k_ul",label:"Platelets",     unit:"K/µL",   normal:[150,400], asd_note:null },
      { key:"hematocrit_pct",     label:"Hematocrit",    unit:"%",      normal:[36,50],   asd_note:null },
      { key:"mcv_fl",             label:"MCV",           unit:"fL",     normal:[80,100],  asd_note:null },
    ]
  },
  "Hormones": {
    icon:"⚗️", color:"#C77DFF", importance:"high",
    desc:"Hormonal environment shapes mood, cognition, and social behaviour. Cortisol and oxytocin are especially important for ASD profiling.",
    markers:[
      { key:"cortisol_ug_dl",     label:"Cortisol",      unit:"µg/dL",  normal:[6,23],    asd_note:"↑ Higher in ASD — stress dysregulation" },
      { key:"testosterone_ng_dl", label:"Testosterone",  unit:"ng/dL",  normal:[300,900], asd_note:"Studied in extreme male brain theory" },
      { key:"estradiol_pg_ml",    label:"Estradiol",     unit:"pg/mL",  normal:[20,150],  asd_note:null },
      { key:"dhea_s_ug_dl",       label:"DHEA-S",        unit:"µg/dL",  normal:[100,400], asd_note:"Cortisol/DHEA ratio important" },
      { key:"tsh_miu_l",          label:"TSH",           unit:"mIU/L",  normal:[0.4,4.0], asd_note:null },
      { key:"free_t3_pg_ml",      label:"Free T3",       unit:"pg/mL",  normal:[2.3,4.2], asd_note:null },
      { key:"free_t4_ng_dl",      label:"Free T4",       unit:"ng/dL",  normal:[0.8,1.8], asd_note:null },
    ]
  },
  "Metabolic": {
    icon:"🔬", color:"#4ECDC4", importance:"medium",
    desc:"Metabolic panel reflects energy regulation and cardiovascular risk, influencing sustained cognitive effort and career endurance.",
    markers:[
      { key:"glucose_mg_dl",      label:"Glucose",       unit:"mg/dL",  normal:[70,100],  asd_note:null },
      { key:"hba1c_pct",          label:"HbA1c",         unit:"%",      normal:[4.0,5.6], asd_note:null },
      { key:"total_cholesterol",  label:"Cholesterol",   unit:"mg/dL",  normal:[150,200], asd_note:null },
      { key:"hdl_mg_dl",          label:"HDL",           unit:"mg/dL",  normal:[40,80],   asd_note:null },
      { key:"ldl_mg_dl",          label:"LDL",           unit:"mg/dL",  normal:[0,130],   asd_note:null },
      { key:"triglycerides_mg_dl",label:"Triglycerides", unit:"mg/dL",  normal:[0,150],   asd_note:null },
      { key:"uric_acid_mg_dl",    label:"Uric Acid",     unit:"mg/dL",  normal:[2.4,7.0], asd_note:null },
      { key:"insulin_uiu_ml",     label:"Insulin",       unit:"µIU/mL", normal:[2,25],    asd_note:null },
    ]
  },
  "Neurotransmitter Precursors": {
    icon:"🧠", color:"#7C9EFF", importance:"critical",
    desc:"These are the MOST CRITICAL for personality and career mapping. They directly affect dopamine, serotonin, and GABA pathways — the chemical basis of personality traits.",
    markers:[
      { key:"tryptophan_umol_l",  label:"Tryptophan",    unit:"µmol/L", normal:[40,80],   asd_note:"↓ Serotonin precursor — lower in ASD" },
      { key:"tyrosine_umol_l",    label:"Tyrosine",      unit:"µmol/L", normal:[40,100],  asd_note:"Dopamine precursor — drives motivation" },
      { key:"serotonin_ng_ml",    label:"Serotonin",     unit:"ng/mL",  normal:[100,300], asd_note:"↓ Reduced in ASD; affects agreeableness" },
      { key:"dopamine_pg_ml",     label:"Dopamine",      unit:"pg/mL",  normal:[10,40],   asd_note:"Reward system — links to conscientiousness" },
      { key:"gaba_nmol_ml",       label:"GABA",          unit:"nmol/mL",normal:[0.2,0.7], asd_note:"↓ Lower in ASD — inhibitory control" },
      { key:"glutamate_nmol_ml",  label:"Glutamate",     unit:"nmol/mL",normal:[1.5,4.0], asd_note:"↑ Elevated in ASD — excitatory excess" },
      { key:"gaba_glutamate_ratio",label:"GABA/Glut Ratio",unit:"ratio",normal:[0.1,0.3], asd_note:"⚑ KEY ASD BIOMARKER — imbalance = ASD signal" },
    ]
  },
  "Inflammatory": {
    icon:"🔥", color:"#FF6B6B", importance:"high",
    desc:"Neuroinflammation is strongly implicated in autism. Elevated IL-6 and CRP indicate immune activation that can impair social cognition.",
    markers:[
      { key:"hs_crp_mg_l",        label:"hsCRP",         unit:"mg/L",   normal:[0,3],     asd_note:"↑ Elevated in ASD — neuroinflammation" },
      { key:"il6_pg_ml",          label:"IL-6",          unit:"pg/mL",  normal:[0,7],     asd_note:"↑ KEY ASD MARKER — elevated by 40%" },
      { key:"tnf_alpha_pg_ml",    label:"TNF-α",         unit:"pg/mL",  normal:[0,8],     asd_note:"↑ Elevated in ASD — immune dysregulation" },
      { key:"homocysteine_umol_l",label:"Homocysteine",  unit:"µmol/L", normal:[5,15],    asd_note:"Methylation pathway marker" },
    ]
  },
  "Nutritional": {
    icon:"🌿", color:"#4CAF50", importance:"medium",
    desc:"Nutritional deficiencies compound ASD traits and reduce cognitive performance. Vitamin D and B12 deficiencies are common in ASD populations.",
    markers:[
      { key:"vitamin_d_ng_ml",    label:"Vitamin D",     unit:"ng/mL",  normal:[30,100],  asd_note:"↓ Often deficient in ASD (-25%)" },
      { key:"b12_pg_ml",          label:"Vitamin B12",   unit:"pg/mL",  normal:[200,900], asd_note:"Critical for myelin and methylation" },
      { key:"folate_ng_ml",       label:"Folate",        unit:"ng/mL",  normal:[5,25],    asd_note:"1-carbon metabolism — ASD risk factor" },
      { key:"ferritin_ng_ml",     label:"Ferritin",      unit:"ng/mL",  normal:[15,200],  asd_note:"Iron storage — affects attention" },
      { key:"zinc_ug_dl",         label:"Zinc",          unit:"µg/dL",  normal:[60,120],  asd_note:"↓ Often reduced in ASD (-15%)" },
      { key:"omega3_index_pct",   label:"Omega-3 Index", unit:"%",      normal:[4,12],    asd_note:"Brain lipid membrane integrity" },
    ]
  },
  "Autism-Specific": {
    icon:"⭐", color:"#FFD166", importance:"critical",
    desc:"These are the HIGHEST PRIORITY biomarkers for this project. They are specifically associated with autism spectrum traits and directly feed into the ASD probability model.",
    markers:[
      { key:"oxytocin_pg_ml",     label:"Oxytocin",      unit:"pg/mL",  normal:[10,40],   asd_note:"⭐ PRIMARY ASD MARKER — ↓30% in ASD; social bonding" },
      { key:"avp_pg_ml",          label:"Vasopressin",   unit:"pg/mL",  normal:[2,8],     asd_note:"⭐ PRIMARY ASD MARKER — ↓20% in ASD; social memory" },
      { key:"bdnf_ng_ml",         label:"BDNF",          unit:"ng/mL",  normal:[15,40],   asd_note:"⭐ Neural plasticity — ↓20% in ASD" },
      { key:"melatonin_pg_ml",    label:"Melatonin",     unit:"pg/mL",  normal:[10,50],   asd_note:"↓30% in ASD — sleep disruption common" },
    ]
  },
};

const VITAL_PRIORITY = {
  critical: { label:"CRITICAL", color:"#FF6B6B", bg:"#2d1515", desc:"Directly drives ASD probability and personality inference" },
  high:     { label:"HIGH",     color:"#FFD166", bg:"#2d2515", desc:"Strongly influences trait scores and career matching" },
  medium:   { label:"MEDIUM",   color:"#4ECDC4", bg:"#152d2b", desc:"Provides supporting context for the overall profile" },
};

const CAREERS = [
  { id:"C001", title:"Software Engineer",           domain:"Technology",   autism:true,  O:0.75, C:0.80, E:0.40, A:0.55, N:0.30, intel:0.85 },
  { id:"C002", title:"Data Scientist / ML Engineer",domain:"Technology",   autism:true,  O:0.85, C:0.78, E:0.38, A:0.52, N:0.28, intel:0.90 },
  { id:"C003", title:"Cybersecurity Analyst",       domain:"Technology",   autism:true,  O:0.72, C:0.88, E:0.35, A:0.48, N:0.32, intel:0.83 },
  { id:"C004", title:"Research Scientist",          domain:"Research",     autism:true,  O:0.92, C:0.82, E:0.42, A:0.60, N:0.25, intel:0.92 },
  { id:"C005", title:"Bioinformatics Scientist",    domain:"Research",     autism:true,  O:0.88, C:0.85, E:0.38, A:0.55, N:0.22, intel:0.90 },
  { id:"C006", title:"Actuary",                     domain:"Finance",      autism:true,  O:0.65, C:0.90, E:0.40, A:0.55, N:0.25, intel:0.88 },
  { id:"C007", title:"Database Administrator",      domain:"Technology",   autism:true,  O:0.62, C:0.88, E:0.35, A:0.52, N:0.28, intel:0.78 },
  { id:"C008", title:"QA Engineer",                 domain:"Technology",   autism:true,  O:0.60, C:0.92, E:0.38, A:0.58, N:0.28, intel:0.75 },
  { id:"C009", title:"Pharmacist",                  domain:"Healthcare",   autism:false, O:0.65, C:0.92, E:0.55, A:0.72, N:0.28, intel:0.85 },
  { id:"C010", title:"Medical Laboratory Scientist",domain:"Healthcare",   autism:true,  O:0.70, C:0.88, E:0.42, A:0.62, N:0.28, intel:0.82 },
  { id:"C011", title:"Radiologist",                 domain:"Healthcare",   autism:true,  O:0.72, C:0.88, E:0.45, A:0.65, N:0.25, intel:0.92 },
  { id:"C012", title:"Graphic Designer",            domain:"Creative",     autism:false, O:0.90, C:0.68, E:0.52, A:0.65, N:0.38, intel:0.70 },
  { id:"C013", title:"Animator / VFX Artist",       domain:"Creative",     autism:true,  O:0.88, C:0.72, E:0.45, A:0.60, N:0.35, intel:0.75 },
  { id:"C014", title:"Financial Analyst",           domain:"Finance",      autism:false, O:0.68, C:0.85, E:0.55, A:0.58, N:0.30, intel:0.85 },
  { id:"C015", title:"Accountant / Auditor",        domain:"Finance",      autism:true,  O:0.55, C:0.92, E:0.45, A:0.62, N:0.28, intel:0.78 },
  { id:"C016", title:"Technical Writer",            domain:"Communication",autism:true,  O:0.75, C:0.80, E:0.38, A:0.62, N:0.30, intel:0.78 },
  { id:"C017", title:"Translator / Linguist",       domain:"Communication",autism:true,  O:0.80, C:0.75, E:0.40, A:0.68, N:0.32, intel:0.82 },
  { id:"C018", title:"Librarian / Archivist",       domain:"Information",  autism:true,  O:0.72, C:0.85, E:0.38, A:0.68, N:0.28, intel:0.75 },
  { id:"C019", title:"Electrical Engineer",         domain:"Engineering",  autism:true,  O:0.72, C:0.85, E:0.48, A:0.58, N:0.28, intel:0.87 },
  { id:"C020", title:"Mechanical Engineer",         domain:"Engineering",  autism:false, O:0.70, C:0.83, E:0.50, A:0.60, N:0.28, intel:0.85 },
  { id:"C021", title:"Mathematician / Statistician",domain:"Research",     autism:true,  O:0.90, C:0.82, E:0.35, A:0.55, N:0.25, intel:0.95 },
  { id:"C022", title:"Cryptographer",               domain:"Technology",   autism:true,  O:0.82, C:0.88, E:0.32, A:0.52, N:0.25, intel:0.93 },
  { id:"C023", title:"Special Education Teacher",   domain:"Education",    autism:false, O:0.75, C:0.80, E:0.68, A:0.88, N:0.35, intel:0.72 },
  { id:"C024", title:"University Professor",        domain:"Education",    autism:false, O:0.88, C:0.80, E:0.60, A:0.65, N:0.28, intel:0.90 },
  { id:"C025", title:"Ecologist / Env. Scientist",  domain:"Science",      autism:false, O:0.82, C:0.78, E:0.52, A:0.72, N:0.28, intel:0.80 },
  { id:"C026", title:"Astronomer / Astrophysicist", domain:"Science",      autism:true,  O:0.92, C:0.82, E:0.38, A:0.55, N:0.25, intel:0.95 },
  { id:"C027", title:"Composer / Music Producer",   domain:"Creative",     autism:true,  O:0.92, C:0.68, E:0.48, A:0.60, N:0.45, intel:0.78 },
  { id:"C028", title:"Architect",                   domain:"Design",       autism:false, O:0.85, C:0.80, E:0.52, A:0.62, N:0.30, intel:0.85 },
  { id:"C029", title:"Clinical Psychologist",       domain:"Healthcare",   autism:false, O:0.80, C:0.78, E:0.65, A:0.82, N:0.32, intel:0.85 },
  { id:"C030", title:"Supply Chain Analyst",        domain:"Business",     autism:true,  O:0.62, C:0.88, E:0.50, A:0.62, N:0.28, intel:0.78 },
];

const PIPELINE_STEPS = [
  { id:"genomics",    icon:"🧬", title:"SNP Genotyping",       model:"SNP-Transformer",  params:"34,664",  desc:"1,000+ SNP markers processed through block-wise Transformer encoder to produce 8 polygenic scores (Big Five + Autism PGS + Intelligence + Education)." },
  { id:"facial",      icon:"👤", title:"Facial Analysis",       model:"FaceGenome-CNN",   params:"230,548", desc:"468 facial landmarks from MediaPipe face mesh fed into CNN morphometry encoder + facial attention module → Big Five embeddings + 7-class emotion recognition." },
  { id:"blood",       icon:"🩸", title:"Blood Biomarkers",      model:"BiomarkerNet",     params:"84,455",  desc:"80 blood markers in 7 clinical categories processed with per-category encoders and cross-attention → ASD probability + personality embeddings." },
  { id:"questionnaire",icon:"📋",title:"Questionnaire",         model:"QuestionnaireEncoder",params:"~5,000",desc:"Big Five self-report responses (0–100 sliders) encoded into a 32-dimensional embedding representing the individual's self-perceived personality." },
  { id:"fusion",      icon:"⚡", title:"Multi-Modal Fusion",    model:"FusionNet",        params:"268,746", desc:"Cross-modal attention transformer fuses all 4 modality embeddings. Each modality attends to all others to capture genomic-biomarker correlations and facial-personality links." },
  { id:"output",      icon:"🗂️", title:"Career Recommendations",model:"Career Engine",    params:"30 careers", desc:"Cosine similarity between fused personality vector and learned O*NET career prototypes. Top-K careers returned with compatibility %, autism-strength flags, and education pathways." },
];

// ─────────────────────────────────────────────
// COMPUTE career scores from Big Five input
// ─────────────────────────────────────────────
function computeCareerScores(bf) {
  return CAREERS.map(c => {
    const diff = Math.abs(bf.O-c.O)+Math.abs(bf.C-c.C)+Math.abs(bf.E-c.E)+Math.abs(bf.A-c.A)+Math.abs(bf.N-c.N);
    const boost = (bf.asd > 0.5 && c.autism) ? -0.12 : 0;
    const raw = 1 - (diff + boost) / 5;
    return { ...c, score: Math.min(100, Math.max(0, raw * 100)) };
  }).sort((a,b)=>b.score-a.score);
}

function computeASD(bm) {
  let score = 0;
  if (bm.oxytocin_pg_ml !== undefined) score += (20 - Math.min(bm.oxytocin_pg_ml, 20)) / 20 * 0.20;
  if (bm.il6_pg_ml !== undefined)      score += Math.min(bm.il6_pg_ml, 14) / 14 * 0.15;
  if (bm.bdnf_ng_ml !== undefined)     score += (25 - Math.min(bm.bdnf_ng_ml, 25)) / 25 * 0.15;
  if (bm.gaba_glutamate_ratio !== undefined) score += (0.16 - Math.min(bm.gaba_glutamate_ratio, 0.16)) / 0.16 * 0.20;
  if (bm.serotonin_ng_ml !== undefined) score += (160 - Math.min(bm.serotonin_ng_ml, 160)) / 160 * 0.15;
  if (bm.vitamin_d_ng_ml !== undefined) score += (30 - Math.min(bm.vitamin_d_ng_ml, 30)) / 30 * 0.15;
  return Math.min(0.95, Math.max(0.05, score));
}

// ─────────────────────────────────────────────
// COLOUR UTILITIES
// ─────────────────────────────────────────────
function getStatusColor(value, normal) {
  if (!normal) return "#8899aa";
  const [lo, hi] = normal;
  if (value < lo * 0.8 || value > hi * 1.2) return "#FF6B6B";
  if (value < lo || value > hi) return "#FFD166";
  return "#4ECDC4";
}
function getStatusLabel(value, normal) {
  if (!normal) return "";
  const [lo, hi] = normal;
  if (value < lo * 0.8) return "⬇ Low";
  if (value > hi * 1.2) return "⬆ High";
  if (value < lo)       return "↓ Low-normal";
  if (value > hi)       return "↑ High-normal";
  return "✓ Normal";
}

// ─────────────────────────────────────────────
// RADAR SVG component
// ─────────────────────────────────────────────
function Radar({ values, size=180 }) {
  const labels = ["O","C","E","A","N"];
  const n = labels.length;
  const cx = size/2, cy = size/2, r = size*0.38;
  const angles = labels.map((_,i) => (i/n)*2*Math.PI - Math.PI/2);
  const pts = values.map((v,i)=>({
    x: cx + r*v*Math.cos(angles[i]),
    y: cy + r*v*Math.sin(angles[i]),
  }));
  const poly = pts.map(p=>`${p.x},${p.y}`).join(" ");
  const gridLevels = [0.25,0.5,0.75,1.0];
  return (
    <svg width={size} height={size}>
      {gridLevels.map(g=>(
        <polygon key={g}
          points={angles.map(a=>`${cx+r*g*Math.cos(a)},${cy+r*g*Math.sin(a)}`).join(" ")}
          fill="none" stroke="#2a3a50" strokeWidth="1"/>
      ))}
      {angles.map((a,i)=>(
        <line key={i} x1={cx} y1={cy} x2={cx+r*Math.cos(a)} y2={cy+r*Math.sin(a)} stroke="#2a3a50" strokeWidth="1"/>
      ))}
      <polygon points={poly} fill="#7C9EFF33" stroke="#7C9EFF" strokeWidth="2"/>
      {pts.map((p,i)=><circle key={i} cx={p.x} cy={p.y} r={4} fill="#7C9EFF"/>)}
      {angles.map((a,i)=>{
        const lx = cx+(r+18)*Math.cos(a), ly = cy+(r+18)*Math.sin(a);
        return <text key={i} x={lx} y={ly+4} textAnchor="middle" fill={BIG_FIVE[i].color} fontSize="11" fontWeight="bold">{labels[i]}</text>;
      })}
    </svg>
  );
}

// ─────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("overview");

  // Big Five sliders (0–1)
  const [bf, setBf] = useState({ O:0.70, C:0.78, E:0.42, A:0.55, N:0.30, asd:0.0 });

  // Biomarker inputs (key → value string)
  const [bm, setBm] = useState({
    oxytocin_pg_ml:"20", il6_pg_ml:"2.5", bdnf_ng_ml:"25",
    gaba_glutamate_ratio:"0.16", serotonin_ng_ml:"160",
    vitamin_d_ng_ml:"30", cortisol_ug_dl:"15",
    tryptophan_umol_l:"55", dopamine_pg_ml:"25",
    hs_crp_mg_l:"1.5", zinc_ug_dl:"90", avp_pg_ml:"4",
  });

  // Computed
  const asdProb = computeASD(Object.fromEntries(Object.entries(bm).map(([k,v])=>[k,parseFloat(v)||0])));
  const bfWithAsd = { ...bf, asd: asdProb };
  const careers = computeCareerScores(bfWithAsd);

  const tabs = [
    { id:"overview",    label:"📖 Overview" },
    { id:"pipeline",    label:"⚙️ Pipeline" },
    { id:"vitals",      label:"🩸 Vital Indicators" },
    { id:"input",       label:"🎛️ Data Input" },
    { id:"results",     label:"📊 Results" },
  ];

  return (
    <div style={{ fontFamily:"'Palatino Linotype','Book Antiqua',Palatino,serif", background:"#070b14", color:"#e2e8f0", minHeight:"100vh", display:"flex", flexDirection:"column" }}>

      {/* Header */}
      <header style={{ background:"linear-gradient(135deg,#0d1627 0%,#0a1020 100%)", borderBottom:"1px solid #1e2a45", padding:"20px 32px" }}>
        <div style={{ display:"flex", alignItems:"center", gap:16, marginBottom:4 }}>
          <div style={{ fontSize:28 }}>🧬</div>
          <div>
            <div style={{ fontSize:22, fontWeight:"bold", color:"#fff", letterSpacing:1 }}>CareerMapping<span style={{color:"#7C9EFF"}}>Genomics</span></div>
            <div style={{ fontSize:12, color:"#8899aa", letterSpacing:2, textTransform:"uppercase" }}>Deep Learning · Genomics · Blood Biomarkers · Facial Analysis</div>
          </div>
          <div style={{ marginLeft:"auto", textAlign:"right" }}>
            <div style={{ fontSize:11, color:"#4ECDC4" }}>618,413 parameters · 4 modalities · 30 careers</div>
            <div style={{ fontSize:11, color:"#8899aa" }}>Abdallah El-Daly — Research Prototype v1.0</div>
          </div>
        </div>
      </header>

      {/* Tab bar */}
      <nav style={{ display:"flex", background:"#0d1220", borderBottom:"1px solid #1e2a45", overflowX:"auto" }}>
        {tabs.map(t=>(
          <button key={t.id} onClick={()=>setTab(t.id)}
            style={{ padding:"12px 22px", background:"none", border:"none", borderBottom: tab===t.id?"3px solid #7C9EFF":"3px solid transparent",
              color: tab===t.id?"#7C9EFF":"#8899aa", cursor:"pointer", fontSize:13, fontFamily:"inherit", whiteSpace:"nowrap", transition:"all .2s" }}>
            {t.label}
          </button>
        ))}
      </nav>

      {/* Content */}
      <div style={{ flex:1, overflowY:"auto" }}>

        {/* ─── OVERVIEW ─── */}
        {tab==="overview" && (
          <div style={{ maxWidth:900, margin:"0 auto", padding:"36px 32px" }}>
            <div style={{ fontSize:11, letterSpacing:3, color:"#7C9EFF", textTransform:"uppercase", marginBottom:8 }}>Full Project Explanation</div>
            <h1 style={{ fontSize:32, color:"#fff", marginBottom:4, fontWeight:"bold" }}>What is CareerMappingGenomics?</h1>
            <p style={{ color:"#8899aa", fontSize:15, lineHeight:1.9, marginBottom:32 }}>
              CareerMappingGenomics is a multi-modal deep learning research system that combines <strong style={{color:"#e2e8f0"}}>four biological and psychological data sources</strong> to predict a person's personality traits and recommend the most suitable career paths for them — with special focus on neurodivergent individuals, particularly those on the <strong style={{color:"#FFD166"}}>autism spectrum</strong>.
            </p>

            {/* What problem does it solve */}
            <div style={{ background:"#0d1220", border:"1px solid #1e2a45", borderRadius:12, padding:24, marginBottom:24, borderLeft:"4px solid #7C9EFF" }}>
              <div style={{ fontSize:14, color:"#7C9EFF", fontWeight:"bold", marginBottom:10 }}>🎯 THE CORE PROBLEM IT SOLVES</div>
              <p style={{ color:"#c8d4e0", lineHeight:1.8, margin:0 }}>
                Traditional career guidance relies entirely on self-reported questionnaires. But questionnaires miss the biological reality beneath — the hormones shaping your mood, the genes influencing your temperament, the facial expressions revealing your emotional world. CareerMappingGenomics reads <em>all four layers at once</em> and produces a richer, more scientifically grounded career profile. For autistic individuals who may not represent themselves accurately in written tests, the biological signals become especially valuable.
              </p>
            </div>

            {/* Four modalities */}
            <h2 style={{ color:"#fff", fontSize:20, marginBottom:16 }}>The Four Data Modalities</h2>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginBottom:32 }}>
              {[
                { icon:"🧬", title:"Genomics (SNP-Transformer)", color:"#7C9EFF",
                  what:"A saliva or blood sample is genotyped. 1,000–500,000 SNP markers are extracted.",
                  how:"The SNP-Transformer processes SNP blocks through a Transformer encoder to produce 8 Polygenic Scores.",
                  output:"Scores for Big Five traits, Autism PGS, Intelligence PGS, Education PGS",
                  accuracy:"Explains 5–15% of personality variance (GWAS ceiling)" },
                { icon:"👤", title:"Facial Analysis (FaceGenome-CNN)", color:"#FFD166",
                  what:"A camera captures the person's face. MediaPipe extracts 468 3D landmarks.",
                  how:"A CNN processes morphometry (bone structure, symmetry) + emotion tracking (FER) while watching stimuli.",
                  output:"Big Five embedding + 7-class emotion probabilities (happy, sad, neutral, etc.)",
                  accuracy:"Pearson r ≈ 0.3–0.5 per Big Five trait" },
                { icon:"🩸", title:"Blood Biomarkers (BiomarkerNet)", color:"#FF6B6B",
                  what:"A fingerprick blood sample (~0.5mL) in a portable lab-on-chip device.",
                  how:"80 markers across 7 categories fed into per-category encoders + cross-attention layer.",
                  output:"ASD probability, personality embedding, biomarker importance weights",
                  accuracy:"AUC 0.72–0.85 for ASD classification" },
                { icon:"📋", title:"Questionnaire (QuestionnaireEncoder)", color:"#4ECDC4",
                  what:"A short Big Five self-report questionnaire (5 trait sliders, 0–100).",
                  how:"Responses encoded into a 32-dimensional embedding via a small MLP.",
                  output:"Questionnaire personality embedding fused with biological signals",
                  accuracy:"Self-report r = 0.6–0.8 with validated instruments" },
              ].map((m,i)=>(
                <div key={i} style={{ background:"#0d1220", border:`1px solid ${m.color}33`, borderRadius:10, padding:18, borderTop:`3px solid ${m.color}` }}>
                  <div style={{ fontSize:22, marginBottom:6 }}>{m.icon}</div>
                  <div style={{ fontSize:14, fontWeight:"bold", color:m.color, marginBottom:8 }}>{m.title}</div>
                  <div style={{ marginBottom:6 }}><span style={{color:"#7C9EFF",fontSize:11,fontWeight:"bold"}}>WHAT: </span><span style={{color:"#c8d4e0",fontSize:12}}>{m.what}</span></div>
                  <div style={{ marginBottom:6 }}><span style={{color:"#4ECDC4",fontSize:11,fontWeight:"bold"}}>HOW: </span><span style={{color:"#c8d4e0",fontSize:12}}>{m.how}</span></div>
                  <div style={{ marginBottom:6 }}><span style={{color:"#FFD166",fontSize:11,fontWeight:"bold"}}>OUTPUT: </span><span style={{color:"#c8d4e0",fontSize:12}}>{m.output}</span></div>
                  <div style={{ fontSize:11, color:"#8899aa", marginTop:8, fontStyle:"italic" }}>Accuracy: {m.accuracy}</div>
                </div>
              ))}
            </div>

            {/* How fusion works */}
            <h2 style={{ color:"#fff", fontSize:20, marginBottom:12 }}>How the Fusion Works</h2>
            <div style={{ background:"#0d1220", border:"1px solid #1e2a45", borderRadius:12, padding:24, marginBottom:24 }}>
              <p style={{ color:"#c8d4e0", lineHeight:1.8, marginBottom:12 }}>
                FusionNet takes the four modality embeddings and stacks them as a sequence of tokens: <code style={{color:"#7C9EFF", background:"#0a0e1a", padding:"2px 6px", borderRadius:4}}>[genomic, facial, biomarker, questionnaire]</code>. Two layers of <strong style={{color:"#fff"}}>cross-modal attention</strong> allow each modality to attend to all the others — so the model learns that a high cortisol reading is more meaningful when the face also shows stress, and that an autism PGS is more predictive when combined with low oxytocin.
              </p>
              <div style={{ display:"flex", alignItems:"center", gap:8, flexWrap:"wrap" }}>
                {["Genomic embed (64d)","Facial embed (128d)","Biomarker embed (64d)","Questionnaire (32d)"].map((s,i)=>(
                  <span key={i} style={{ background:"#0a0e1a", border:"1px solid #2a3a50", borderRadius:20, padding:"4px 12px", fontSize:12, color:"#c8d4e0" }}>{s}</span>
                ))}
                <span style={{ color:"#7C9EFF", fontSize:20 }}>→</span>
                <span style={{ background:"#7C9EFF22", border:"1px solid #7C9EFF", borderRadius:20, padding:"4px 14px", fontSize:12, color:"#7C9EFF", fontWeight:"bold" }}>Cross-Modal Attention (256d)</span>
                <span style={{ color:"#7C9EFF", fontSize:20 }}>→</span>
                <span style={{ background:"#4ECDC422", border:"1px solid #4ECDC4", borderRadius:20, padding:"4px 14px", fontSize:12, color:"#4ECDC4", fontWeight:"bold" }}>Career Scores</span>
              </div>
            </div>

            {/* ASD focus */}
            <h2 style={{ color:"#fff", fontSize:20, marginBottom:12 }}>Special Focus: Autism Spectrum</h2>
            <div style={{ background:"#1a1a0d", border:"1px solid #FFD16644", borderRadius:12, padding:24, marginBottom:24 }}>
              <p style={{ color:"#c8d4e0", lineHeight:1.8, margin:0 }}>
                Research shows that <strong style={{color:"#FFD166"}}>12,000+ genetic variants are shared between autism and intelligence</strong> (Dice coefficient = 0.91). This means autism is not a deficit but a <em>different cognitive style</em> with documented strengths: hyper-systemising, pattern recognition, deep focus, reduced conformity bias, exceptional memory in specialist domains. The system maps these strengths directly to careers marked with ⭐ <span style={{color:"#FFD166"}}>Autism Strength</span> — 21 out of 30 careers where neurodivergent traits create genuine competitive advantage.
              </p>
            </div>

            {/* Ethics box */}
            <div style={{ background:"#1a0d0d", border:"1px solid #FF6B6B44", borderRadius:12, padding:20 }}>
              <div style={{ fontSize:13, color:"#FF6B6B", fontWeight:"bold", marginBottom:8 }}>⚠ ETHICAL FRAMEWORK</div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
                {["Personality PGS explain only 5–15% of variance — never deterministic","IRB approval required before any clinical deployment","GDPR Article 9: genomic data = special category","Facial recognition has known demographic bias — must audit","Career recommendations only — never disqualification","Right to erasure of all biological data","Diverse training cohort needed (MENA ancestry)","Informed consent for every data modality"].map((s,i)=>(
                  <div key={i} style={{ fontSize:12, color:"#c8d4e0", padding:"5px 10px", background:"#0a0e1a", borderRadius:5 }}>⚖ {s}</div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ─── PIPELINE ─── */}
        {tab==="pipeline" && (
          <div style={{ maxWidth:860, margin:"0 auto", padding:"36px 32px" }}>
            <div style={{ fontSize:11, letterSpacing:3, color:"#7C9EFF", textTransform:"uppercase", marginBottom:8 }}>End-to-End Architecture</div>
            <h1 style={{ fontSize:28, color:"#fff", marginBottom:24 }}>The Deep Learning Pipeline</h1>
            <div style={{ display:"flex", flexDirection:"column", gap:2 }}>
              {PIPELINE_STEPS.map((step, i)=>(
                <div key={step.id} style={{ display:"flex", gap:0, alignItems:"stretch" }}>
                  {/* Left spine */}
                  <div style={{ display:"flex", flexDirection:"column", alignItems:"center", width:50, flexShrink:0 }}>
                    <div style={{ width:36, height:36, borderRadius:"50%", background:"#0d1220", border:"2px solid #7C9EFF", display:"flex", alignItems:"center", justifyContent:"center", fontSize:18, flexShrink:0 }}>{step.icon}</div>
                    {i<PIPELINE_STEPS.length-1 && <div style={{ width:2, flex:1, background:"#1e2a45", marginTop:4 }}/>}
                  </div>
                  {/* Card */}
                  <div style={{ flex:1, background:"#0d1220", border:"1px solid #1e2a45", borderRadius:10, padding:18, marginLeft:12, marginBottom:12 }}>
                    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:8 }}>
                      <div>
                        <div style={{ fontSize:15, fontWeight:"bold", color:"#fff" }}>{step.title}</div>
                        <div style={{ fontSize:11, color:"#7C9EFF", marginTop:2 }}>Model: <strong>{step.model}</strong></div>
                      </div>
                      <div style={{ background:"#0a0e1a", border:"1px solid #2a3a50", borderRadius:6, padding:"4px 10px", fontSize:11, color:"#4ECDC4" }}>{step.params} params</div>
                    </div>
                    <p style={{ fontSize:13, color:"#8899aa", lineHeight:1.7, margin:0 }}>{step.desc}</p>
                  </div>
                </div>
              ))}
            </div>
            {/* Total */}
            <div style={{ background:"#0d1220", border:"2px solid #7C9EFF", borderRadius:12, padding:20, marginTop:8, textAlign:"center" }}>
              <div style={{ fontSize:28, color:"#7C9EFF", fontWeight:"bold" }}>618,413</div>
              <div style={{ color:"#8899aa", fontSize:13 }}>Total trainable parameters across all four models</div>
              <div style={{ display:"flex", justifyContent:"center", gap:20, marginTop:12 }}>
                {[["SNP-Transformer","34,664","#7C9EFF"],["FaceGenome-CNN","230,548","#FFD166"],["BiomarkerNet","84,455","#FF6B6B"],["FusionNet","268,746","#4ECDC4"]].map(([name,p,c])=>(
                  <div key={name} style={{ textAlign:"center" }}>
                    <div style={{ color:c, fontWeight:"bold", fontSize:14 }}>{p}</div>
                    <div style={{ color:"#8899aa", fontSize:11 }}>{name}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ─── VITAL INDICATORS ─── */}
        {tab==="vitals" && (
          <div style={{ maxWidth:920, margin:"0 auto", padding:"36px 32px" }}>
            <div style={{ fontSize:11, letterSpacing:3, color:"#FF6B6B", textTransform:"uppercase", marginBottom:8 }}>Biomarker Guide</div>
            <h1 style={{ fontSize:28, color:"#fff", marginBottom:4 }}>Vital Indicators: Where They Are & Why They Matter</h1>
            <p style={{ color:"#8899aa", marginBottom:24, lineHeight:1.8 }}>
              The 80 blood biomarkers are organised into 7 clinical categories. Two categories are marked <strong style={{color:"#FF6B6B"}}>CRITICAL</strong> — these directly drive the ASD probability and personality scores. Below is a complete guide to every biomarker, its normal range, and its specific role in the model.
            </p>

            {/* Priority legend */}
            <div style={{ display:"flex", gap:12, marginBottom:28, flexWrap:"wrap" }}>
              {Object.entries(VITAL_PRIORITY).map(([k,v])=>(
                <div key={k} style={{ background:v.bg, border:`1px solid ${v.color}`, borderRadius:8, padding:"8px 14px", display:"flex", alignItems:"center", gap:8 }}>
                  <span style={{ background:v.color, color:"#0a0e1a", borderRadius:4, padding:"2px 8px", fontSize:11, fontWeight:"bold" }}>{v.label}</span>
                  <span style={{ fontSize:12, color:"#c8d4e0" }}>{v.desc}</span>
                </div>
              ))}
            </div>

            {/* How to add biomarkers */}
            <div style={{ background:"#0d1220", border:"1px solid #4ECDC444", borderRadius:12, padding:20, marginBottom:28 }}>
              <div style={{ fontSize:13, color:"#4ECDC4", fontWeight:"bold", marginBottom:10 }}>🔧 HOW TO ADD NEW BIOMARKERS TO THE SYSTEM</div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12 }}>
                {[
                  { step:"1. Add to BIOMARKER_SPECS", code:"src/utils/synthetic_data.py", detail:"Add name: (mean, std) to the BIOMARKER_SPECS dictionary. This defines the reference range." },
                  { step:"2. Assign to a category", code:"src/models/biomarker_model.py", detail:"Update CATEGORY_SLICES with a new slice or extend an existing category's slice range." },
                  { step:"3. Add ASD shift (if relevant)", code:"src/utils/synthetic_data.py", detail:"In _autism_biomarker_shift(), add the marker key with a z_delta value (+ = higher in ASD, - = lower)." },
                  { step:"4. Retrain the model", code:"python main.py --epochs 30", detail:"Run main.py after updating the data. BiomarkerNet will automatically adapt to the new feature count." },
                ].map((s,i)=>(
                  <div key={i} style={{ background:"#0a0e1a", borderRadius:8, padding:12 }}>
                    <div style={{ fontSize:12, color:"#4ECDC4", fontWeight:"bold", marginBottom:4 }}>{s.step}</div>
                    <code style={{ fontSize:11, color:"#7C9EFF", display:"block", marginBottom:4 }}>{s.code}</code>
                    <div style={{ fontSize:11, color:"#8899aa" }}>{s.detail}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Category cards */}
            {Object.entries(BIOMARKER_CATEGORIES).map(([catName, cat])=>(
              <div key={catName} style={{ background:"#0d1220", border:`1px solid ${cat.color}33`, borderRadius:12, padding:20, marginBottom:16, borderTop:`3px solid ${cat.color}` }}>
                <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:8 }}>
                  <span style={{ fontSize:22 }}>{cat.icon}</span>
                  <div>
                    <div style={{ fontSize:15, fontWeight:"bold", color:cat.color }}>{catName}</div>
                    <span style={{ background:VITAL_PRIORITY[cat.importance].color, color:"#0a0e1a", borderRadius:4, padding:"1px 8px", fontSize:10, fontWeight:"bold" }}>{VITAL_PRIORITY[cat.importance].label}</span>
                  </div>
                </div>
                <p style={{ fontSize:12, color:"#8899aa", lineHeight:1.6, marginBottom:12 }}>{cat.desc}</p>
                <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill, minmax(260px, 1fr))", gap:8 }}>
                  {cat.markers.map(m=>(
                    <div key={m.key} style={{ background:"#0a0e1a", borderRadius:8, padding:10, borderLeft:`3px solid ${m.asd_note ? cat.color : "#2a3a50"}` }}>
                      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                        <span style={{ fontSize:13, color:"#e2e8f0", fontWeight:"bold" }}>{m.label}</span>
                        <span style={{ fontSize:11, color:"#4a90d9" }}>{m.unit}</span>
                      </div>
                      <div style={{ fontSize:11, color:"#8899aa", marginTop:2 }}>Normal: {m.normal ? `${m.normal[0]}–${m.normal[1]} ${m.unit}` : "varies"}</div>
                      {m.asd_note && <div style={{ fontSize:11, color:cat.color, marginTop:4 }}>{m.asd_note}</div>}
                      <code style={{ fontSize:10, color:"#4a6080", display:"block", marginTop:4 }}>{m.key}</code>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ─── DATA INPUT ─── */}
        {tab==="input" && (
          <div style={{ maxWidth:920, margin:"0 auto", padding:"36px 32px" }}>
            <div style={{ fontSize:11, letterSpacing:3, color:"#4ECDC4", textTransform:"uppercase", marginBottom:8 }}>Interactive Data Entry</div>
            <h1 style={{ fontSize:28, color:"#fff", marginBottom:4 }}>Enter Your Data</h1>
            <p style={{ color:"#8899aa", marginBottom:24, lineHeight:1.7 }}>
              Adjust the Big Five personality sliders and enter blood biomarker values from a lab report. The model will update career recommendations in real time. <strong style={{color:"#4ECDC4"}}>Key ASD biomarkers</strong> have the greatest impact on the ASD probability score.
            </p>

            {/* ASD probability live badge */}
            <div style={{ background:"#0d1220", border:"2px solid #7C9EFF", borderRadius:12, padding:20, marginBottom:28, display:"flex", alignItems:"center", gap:24 }}>
              <Radar values={[bf.O, bf.C, bf.E, bf.A, bf.N]} size={160} />
              <div style={{ flex:1 }}>
                <div style={{ fontSize:12, color:"#8899aa", marginBottom:8 }}>LIVE PERSONALITY PROFILE</div>
                {BIG_FIVE.map(t=>(
                  <div key={t.key} style={{ marginBottom:8 }}>
                    <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3 }}>
                      <span style={{ fontSize:12, color:t.color }}>{t.name} ({t.key})</span>
                      <span style={{ fontSize:12, color:"#fff" }}>{Math.round(bf[t.key]*100)}%</span>
                    </div>
                    <div style={{ height:6, background:"#1e2a45", borderRadius:3, overflow:"hidden" }}>
                      <div style={{ width:`${bf[t.key]*100}%`, height:"100%", background:t.color, transition:"width .2s", borderRadius:3 }}/>
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ textAlign:"center", minWidth:100 }}>
                <div style={{ fontSize:36, fontWeight:"bold", color: asdProb>0.6?"#FF6B6B":asdProb>0.35?"#FFD166":"#4ECDC4" }}>{Math.round(asdProb*100)}%</div>
                <div style={{ fontSize:11, color:"#8899aa" }}>ASD Probability</div>
                <div style={{ fontSize:11, color: asdProb>0.6?"#FF6B6B":asdProb>0.35?"#FFD166":"#4ECDC4", marginTop:4 }}>
                  {asdProb>0.6?"Likely ASD traits":asdProb>0.35?"Possible ASD traits":"Neurotypical"}
                </div>
              </div>
            </div>

            {/* Big Five sliders */}
            <h3 style={{ color:"#fff", fontSize:16, marginBottom:14 }}>📋 Step 1 — Questionnaire (Big Five Self-Report)</h3>
            <div style={{ background:"#0d1220", border:"1px solid #1e2a45", borderRadius:12, padding:20, marginBottom:24 }}>
              {BIG_FIVE.map(t=>(
                <div key={t.key} style={{ marginBottom:16 }}>
                  <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
                    <div>
                      <span style={{ fontSize:14, color:t.color, fontWeight:"bold" }}>{t.name}</span>
                      <span style={{ fontSize:12, color:"#8899aa", marginLeft:8 }}>{t.desc}</span>
                    </div>
                    <span style={{ fontSize:14, color:"#fff", fontWeight:"bold", minWidth:36, textAlign:"right" }}>{Math.round(bf[t.key]*100)}</span>
                  </div>
                  <input type="range" min="0" max="100" value={Math.round(bf[t.key]*100)}
                    onChange={e=>setBf(p=>({...p,[t.key]:+e.target.value/100}))}
                    style={{ width:"100%", accentColor:t.color, cursor:"pointer" }}/>
                  <div style={{ fontSize:11, color:"#4ECDC4", marginTop:2 }}>{t.asd}</div>
                </div>
              ))}
            </div>

            {/* Biomarker inputs */}
            <h3 style={{ color:"#fff", fontSize:16, marginBottom:6 }}>🩸 Step 2 — Key Blood Biomarkers</h3>
            <p style={{ color:"#8899aa", fontSize:12, marginBottom:14 }}>Enter values from a blood test report. Fields marked ⭐ have the highest impact on the ASD probability score.</p>

            {Object.entries(BIOMARKER_CATEGORIES).filter(([,c])=>c.importance==="critical").map(([catName, cat])=>(
              <div key={catName} style={{ background:"#0d1220", border:`2px solid ${cat.color}`, borderRadius:12, padding:20, marginBottom:16 }}>
                <div style={{ fontSize:13, color:cat.color, fontWeight:"bold", marginBottom:14 }}>{cat.icon} {catName} — <span style={{background:cat.color,color:"#0a0e1a",borderRadius:4,padding:"1px 8px",fontSize:10}}>CRITICAL</span></div>
                <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill, minmax(220px, 1fr))", gap:12 }}>
                  {cat.markers.filter(m=>bm[m.key]!==undefined||true).map(m=>{
                    const val = parseFloat(bm[m.key]||"0");
                    const statusColor = getStatusColor(val, m.normal);
                    const statusLabel = getStatusLabel(val, m.normal);
                    return (
                      <div key={m.key} style={{ background:"#0a0e1a", borderRadius:8, padding:12, borderLeft:`3px solid ${statusColor}` }}>
                        <div style={{ fontSize:12, color:"#e2e8f0", fontWeight:"bold", marginBottom:2 }}>{m.label}</div>
                        {m.asd_note && <div style={{ fontSize:10, color:cat.color, marginBottom:6 }}>{m.asd_note}</div>}
                        <div style={{ display:"flex", gap:6, alignItems:"center" }}>
                          <input type="number" value={bm[m.key]||""} placeholder={`${m.normal?m.normal[0]:0}`}
                            onChange={e=>setBm(p=>({...p,[m.key]:e.target.value}))}
                            style={{ flex:1, background:"#0d1220", border:`1px solid ${statusColor}55`, borderRadius:6, padding:"6px 10px", color:"#fff", fontSize:13, fontFamily:"monospace", outline:"none", width:0 }}/>
                          <span style={{ fontSize:11, color:"#4a90d9", minWidth:50 }}>{m.unit}</span>
                        </div>
                        <div style={{ fontSize:11, color:statusColor, marginTop:4 }}>{statusLabel || `Normal: ${m.normal?m.normal.join("–"):""}`}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}

            {/* High importance */}
            {Object.entries(BIOMARKER_CATEGORIES).filter(([,c])=>c.importance==="high").map(([catName, cat])=>(
              <div key={catName} style={{ background:"#0d1220", border:`1px solid ${cat.color}55`, borderRadius:12, padding:20, marginBottom:16 }}>
                <div style={{ fontSize:13, color:cat.color, fontWeight:"bold", marginBottom:14 }}>{cat.icon} {catName} — <span style={{background:VITAL_PRIORITY.high.color,color:"#0a0e1a",borderRadius:4,padding:"1px 8px",fontSize:10}}>HIGH</span></div>
                <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill, minmax(200px, 1fr))", gap:10 }}>
                  {cat.markers.map(m=>{
                    const val = parseFloat(bm[m.key]||"0");
                    const statusColor = bm[m.key] ? getStatusColor(val, m.normal) : "#2a3a50";
                    return (
                      <div key={m.key} style={{ background:"#0a0e1a", borderRadius:8, padding:10 }}>
                        <div style={{ fontSize:12, color:"#c8d4e0", marginBottom:4 }}>{m.label}{m.asd_note?<span style={{color:cat.color,fontSize:10,marginLeft:4}}>●</span>:""}</div>
                        <div style={{ display:"flex", gap:6 }}>
                          <input type="number" value={bm[m.key]||""} placeholder={m.normal?`${m.normal[0]}`:""}
                            onChange={e=>setBm(p=>({...p,[m.key]:e.target.value}))}
                            style={{ flex:1, background:"#0d1220", border:`1px solid ${statusColor}`, borderRadius:5, padding:"5px 8px", color:"#fff", fontSize:12, fontFamily:"monospace", outline:"none", width:0 }}/>
                          <span style={{ fontSize:10, color:"#4a90d9", minWidth:44 }}>{m.unit}</span>
                        </div>
                        {bm[m.key] && <div style={{ fontSize:10, color:statusColor, marginTop:3 }}>{getStatusLabel(val,m.normal)}</div>}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}

            <button onClick={()=>setTab("results")}
              style={{ display:"block", width:"100%", padding:"14px", background:"linear-gradient(135deg,#7C9EFF,#4ECDC4)", border:"none", borderRadius:10, color:"#0a0e1a", fontSize:15, fontWeight:"bold", cursor:"pointer", marginTop:8, letterSpacing:1 }}>
              VIEW CAREER RECOMMENDATIONS →
            </button>
          </div>
        )}

        {/* ─── RESULTS ─── */}
        {tab==="results" && (
          <div style={{ maxWidth:920, margin:"0 auto", padding:"36px 32px" }}>
            <div style={{ fontSize:11, letterSpacing:3, color:"#4ECDC4", textTransform:"uppercase", marginBottom:8 }}>Live Analysis</div>
            <h1 style={{ fontSize:28, color:"#fff", marginBottom:24 }}>Career Recommendations</h1>

            {/* Summary row */}
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr 1fr", gap:12, marginBottom:28 }}>
              {[
                { label:"ASD Score", value:`${Math.round(asdProb*100)}%`, color: asdProb>0.6?"#FF6B6B":asdProb>0.35?"#FFD166":"#4ECDC4" },
                { label:"Top Trait", value:BIG_FIVE.reduce((a,b)=>bf[b.key]>bf[a.key]?b:a).name, color:"#7C9EFF" },
                { label:"Best Career", value:careers[0]?.title.slice(0,18)+"…", color:"#FFD166" },
                { label:"Match Score", value:`${Math.round(careers[0]?.score||0)}%`, color:"#4ECDC4" },
              ].map((s,i)=>(
                <div key={i} style={{ background:"#0d1220", border:"1px solid #1e2a45", borderRadius:10, padding:16, textAlign:"center" }}>
                  <div style={{ fontSize:22, fontWeight:"bold", color:s.color }}>{s.value}</div>
                  <div style={{ fontSize:11, color:"#8899aa" }}>{s.label}</div>
                </div>
              ))}
            </div>

            {/* Personality profile */}
            <div style={{ display:"grid", gridTemplateColumns:"auto 1fr", gap:24, background:"#0d1220", border:"1px solid #1e2a45", borderRadius:12, padding:20, marginBottom:24 }}>
              <Radar values={[bf.O,bf.C,bf.E,bf.A,bf.N]} size={200}/>
              <div>
                <div style={{ fontSize:13, color:"#7C9EFF", fontWeight:"bold", marginBottom:12 }}>PERSONALITY PROFILE</div>
                {BIG_FIVE.map(t=>{
                  const v = bf[t.key];
                  const level = v>0.66?"High":v>0.33?"Moderate":"Low";
                  return (
                    <div key={t.key} style={{ marginBottom:10 }}>
                      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3 }}>
                        <span style={{ fontSize:13, color:t.color }}>{t.name}</span>
                        <span style={{ fontSize:12, color:"#8899aa" }}>{level} ({Math.round(v*100)}%)</span>
                      </div>
                      <div style={{ height:8, background:"#1e2a45", borderRadius:4, overflow:"hidden" }}>
                        <div style={{ width:`${v*100}%`, height:"100%", background:`linear-gradient(90deg,${t.color}88,${t.color})`, borderRadius:4, transition:"width .3s" }}/>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Career rankings */}
            <h2 style={{ color:"#fff", fontSize:18, marginBottom:14 }}>Top Career Matches</h2>
            <div style={{ display:"flex", flexDirection:"column", gap:8, marginBottom:24 }}>
              {careers.slice(0,12).map((c,i)=>{
                const barW = Math.round(c.score);
                const isTop = i===0;
                return (
                  <div key={c.id} style={{ background: isTop?"#0f1a2e":"#0d1220", border:`1px solid ${isTop?"#7C9EFF":"#1e2a45"}`, borderRadius:10, padding:"12px 16px", display:"flex", alignItems:"center", gap:12 }}>
                    <div style={{ width:28, height:28, borderRadius:"50%", background: isTop?"#7C9EFF":"#1e2a45", color: isTop?"#0a0e1a":"#8899aa", display:"flex", alignItems:"center", justifyContent:"center", fontSize:12, fontWeight:"bold", flexShrink:0 }}>{i+1}</div>
                    <div style={{ flex:1, minWidth:0 }}>
                      <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:4 }}>
                        <span style={{ fontSize:14, color:"#fff", fontWeight: isTop?"bold":"normal" }}>{c.title}</span>
                        {c.autism && <span style={{ background:"#FFD16633", color:"#FFD166", borderRadius:4, padding:"1px 6px", fontSize:10, fontWeight:"bold" }}>⭐ ASD+</span>}
                        <span style={{ background:"#1e2a45", color:"#8899aa", borderRadius:4, padding:"1px 6px", fontSize:10 }}>{c.domain}</span>
                      </div>
                      <div style={{ height:5, background:"#1e2a45", borderRadius:3, overflow:"hidden" }}>
                        <div style={{ width:`${barW}%`, height:"100%", background: isTop?"linear-gradient(90deg,#7C9EFF,#4ECDC4)":"#2a4a6a", borderRadius:3, transition:"width .3s" }}/>
                      </div>
                    </div>
                    <div style={{ fontSize:15, fontWeight:"bold", color: isTop?"#7C9EFF":"#8899aa", minWidth:44, textAlign:"right" }}>{barW}%</div>
                  </div>
                );
              })}
            </div>

            {/* Top career detail */}
            {careers[0] && (
              <div style={{ background:"#0d1220", border:"2px solid #7C9EFF", borderRadius:12, padding:24, marginBottom:24 }}>
                <div style={{ fontSize:11, color:"#7C9EFF", letterSpacing:2, marginBottom:6 }}>TOP RECOMMENDATION</div>
                <div style={{ fontSize:22, color:"#fff", fontWeight:"bold", marginBottom:8 }}>{careers[0].title}</div>
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginBottom:12 }}>
                  {["O","C","E","A","N"].map(k=>{
                    const career_val = careers[0][k];
                    const you = bf[k];
                    const match = 1 - Math.abs(career_val - you);
                    return (
                      <div key={k} style={{ background:"#0a0e1a", borderRadius:8, padding:10 }}>
                        <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
                          <span style={{ color:BIG_FIVE.find(t=>t.key===k).color, fontSize:12 }}>{BIG_FIVE.find(t=>t.key===k).name}</span>
                          <span style={{ color:"#8899aa", fontSize:11 }}>Match: {Math.round(match*100)}%</span>
                        </div>
                        <div style={{ display:"flex", gap:4, alignItems:"center" }}>
                          <span style={{ fontSize:10, color:"#8899aa", width:28 }}>You</span>
                          <div style={{ flex:1, height:5, background:"#1e2a45", borderRadius:3, overflow:"hidden" }}>
                            <div style={{ width:`${you*100}%`, height:"100%", background:BIG_FIVE.find(t=>t.key===k).color, borderRadius:3 }}/>
                          </div>
                        </div>
                        <div style={{ display:"flex", gap:4, alignItems:"center", marginTop:2 }}>
                          <span style={{ fontSize:10, color:"#8899aa", width:28 }}>Job</span>
                          <div style={{ flex:1, height:5, background:"#1e2a45", borderRadius:3, overflow:"hidden" }}>
                            <div style={{ width:`${career_val*100}%`, height:"100%", background:"#4a90d9", borderRadius:3 }}/>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
                {careers[0].autism && (
                  <div style={{ background:"#1a1a0d", border:"1px solid #FFD16655", borderRadius:8, padding:12 }}>
                    <div style={{ fontSize:12, color:"#FFD166", fontWeight:"bold", marginBottom:6 }}>⭐ Autism Strength Career</div>
                    <div style={{ fontSize:12, color:"#c8d4e0" }}>This career is flagged as an <strong>autism-strength career</strong> where neurodivergent cognitive traits create genuine competitive advantage — systematic thinking, deep focus, pattern recognition, and reduced conformity bias.</div>
                  </div>
                )}
              </div>
            )}

            {/* Disclaimer */}
            <div style={{ background:"#1a0d0d", border:"1px solid #FF6B6B44", borderRadius:10, padding:16 }}>
              <div style={{ fontSize:12, color:"#FF6B6B", fontWeight:"bold", marginBottom:6 }}>⚠ Research Disclaimer</div>
              <p style={{ fontSize:12, color:"#8899aa", lineHeight:1.7, margin:0 }}>This is a research prototype. Personality polygenic scores explain 5–15% of trait variance. Results are probabilistic, not deterministic. Career recommendations should be considered alongside personal interests, cultural context, education, and professional guidance. All outputs carry uncertainty — biological signals are one layer of a complex human story.</p>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer style={{ background:"#0d1220", borderTop:"1px solid #1e2a45", padding:"12px 32px", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <span style={{ fontSize:11, color:"#4a6080" }}>CareerMappingGenomics v1.0 — Abdallah El-Daly</span>
        <span style={{ fontSize:11, color:"#4a6080" }}>Research Prototype · Not for clinical use</span>
      </footer>
    </div>
  );
}
