import { useState } from "react";

const sections = [
  { id: "overview", label: "Project Overview" },
  { id: "architecture", label: "System Architecture" },
  { id: "modules", label: "Deep Learning Modules" },
  { id: "data", label: "Data Pipeline" },
  { id: "genomics", label: "Genomic Analysis" },
  { id: "facial", label: "Facial Recognition" },
  { id: "blood", label: "Blood Biomarkers" },
  { id: "career", label: "Career Mapping" },
  { id: "ethics", label: "Ethics & Limitations" },
  { id: "roadmap", label: "Research Roadmap" },
];

const pillars = [
  { icon: "🧬", title: "Genomics", color: "#00C49F", desc: "SNP-based polygenic scoring, GWAS integration, rare variant analysis via WES/WGS" },
  { icon: "🧠", title: "Deep Learning", color: "#6C63FF", desc: "CNN for facial features, LSTM for sequence-based genomic data, Transformers for multi-modal fusion" },
  { icon: "🩸", title: "Blood Biomarkers", color: "#FF6B6B", desc: "Complete blood count, hormone panels, neurotransmitter precursors, metabolomics" },
  { icon: "👤", title: "Facial Recognition", color: "#FFD166", desc: "Morphometric analysis, facial emotion tracking, FER-based personality inference" },
  { icon: "🗂️", title: "Career Mapping", color: "#4ECDC4", desc: "O*NET job taxonomy, Big Five trait-to-career correlation, personalized recommendation engine" },
  { icon: "⚖️", title: "Ethics & Consent", color: "#F8961E", desc: "IRB compliance, GDPR/HIPAA alignment, bias auditing, informed consent framework" },
];

const architectureLayers = [
  {
    name: "INPUT LAYER",
    color: "#1e3a5f",
    border: "#4a90d9",
    items: [
      "Facial image capture (device camera)",
      "Blood sample → lab-on-chip sequencing",
      "Genomic SNP array / WGS output",
      "Psychometric questionnaires (Big Five, MBTI)",
    ],
  },
  {
    name: "PREPROCESSING & FEATURE EXTRACTION",
    color: "#1e3d2f",
    border: "#00C49F",
    items: [
      "Face alignment → 3D landmark detection",
      "Variant calling (GATK pipeline), QC, imputation",
      "Blood CBC parsing + metabolite normalization",
      "Text/response embedding (BERT fine-tuned)",
    ],
  },
  {
    name: "DEEP LEARNING CORE",
    color: "#2d1e5f",
    border: "#6C63FF",
    items: [
      "CNN branch: facial morphometry → trait embeddings",
      "Transformer branch: SNP sequences → polygenic embeddings",
      "GNN branch: gene interaction networks",
      "Multi-modal fusion layer (cross-attention)",
    ],
  },
  {
    name: "PERSONALITY INFERENCE ENGINE",
    color: "#3d2a10",
    border: "#FFD166",
    items: [
      "Big Five trait scores (O, C, E, A, N)",
      "Cognitive profile (fluid/crystallized intelligence markers)",
      "Autism spectrum trait quantification",
      "Confidence intervals & uncertainty estimates",
    ],
  },
  {
    name: "CAREER RECOMMENDATION ENGINE",
    color: "#1a3030",
    border: "#4ECDC4",
    items: [
      "O*NET occupational database mapping (900+ careers)",
      "Trait-to-career compatibility scoring",
      "Personalized career report generation",
      "Longitudinal tracking & feedback loop",
    ],
  },
];

const dlModels = [
  {
    name: "FaceGenome-CNN",
    type: "Convolutional Neural Network",
    purpose: "Extract morphometric features from facial images for personality trait inference",
    inputs: "RGB facial images (256×256)",
    outputs: "512-dim trait embedding vector",
    arch: "ResNet-50 backbone → GAP → FC(256) → FC(128) → 5-trait regression head",
    dataset: "CelebA + custom clinical dataset",
    metrics: "Pearson r ≈ 0.3–0.5 per Big Five trait (literature baseline)",
    note: "Retrain on autism-specific cohort for domain adaptation",
    color: "#FFD166",
  },
  {
    name: "SNP-Transformer",
    type: "Transformer / LSTM Hybrid",
    purpose: "Model long-range SNP interactions for polygenic trait prediction",
    inputs: "Genotype matrix (N × 500k SNPs), LD-pruned to ~50k",
    outputs: "Polygenic score vectors for Big Five + cognitive traits",
    arch: "SNP embedding → Positional encoding → 6-layer Transformer → MLP regression",
    dataset: "UK Biobank (N=500k), ABCD Study, SPARK autism cohort",
    metrics: "R² = 0.05–0.15 (current GWAS ceiling for personality traits)",
    note: "Integrate GWAS summary stats via LDpred2 for score calibration",
    color: "#6C63FF",
  },
  {
    name: "BiomarkerNet",
    type: "Multi-Layer Perceptron + Attention",
    purpose: "Map blood biomarker panels to neurological and behavioral phenotypes",
    inputs: "~80 blood markers (CBC, hormones, metabolites, proteins)",
    outputs: "Neurotype profile + energy/stress/cognitive load estimates",
    arch: "LayerNorm → 3×FC(256) → Self-Attention → FC(64) → output",
    dataset: "NHANES, custom clinical blood study",
    metrics: "AUC 0.72–0.85 for autism spectrum classification",
    note: "Use SHapley values for biomarker importance interpretation",
    color: "#FF6B6B",
  },
  {
    name: "FusionNet",
    type: "Cross-Modal Attention Transformer",
    purpose: "Fuse all modalities into a unified individual profile",
    inputs: "Face embedding, SNP embedding, biomarker embedding, questionnaire embedding",
    outputs: "Final personality vector + career compatibility scores",
    arch: "4-modality cross-attention → joint embedding (256-dim) → career head",
    dataset: "All above combined with career outcome labels",
    metrics: "Target: Spearman ρ > 0.45 vs. validated psychometric assessments",
    note: "Ablation study required to quantify each modality's contribution",
    color: "#00C49F",
  },
];

const genomicPipeline = [
  { step: "Sample Collection", detail: "Saliva or blood → DNA extraction (Chelex or column-based)", tech: "Lab-on-chip device" },
  { step: "Genotyping / Sequencing", detail: "SNP array (650k markers) or low-pass WGS (0.5×)", tech: "Illumina GSA or MGI DNBSEQ" },
  { step: "Quality Control", detail: "Call rate >98%, Hardy-Weinberg p>1e-6, MAF>1%, relatedness check", tech: "PLINK2, KING" },
  { step: "Imputation", detail: "Pre-phasing (SHAPEIT4) → imputation against TOPMed reference panel", tech: "Michigan Imputation Server" },
  { step: "Ancestry PCA", detail: "Project onto 1000 Genomes PCs, correct for stratification", tech: "FlashPCA2" },
  { step: "Polygenic Scoring", detail: "LDpred2-auto for Big Five, intelligence, educational attainment", tech: "bigsnpr R package" },
  { step: "Rare Variant Analysis", detail: "CADD scoring, OMIM lookup, autism gene list overlap", tech: "VEP, ANNOVAR" },
  { step: "Feature Export", detail: "Normalized PGS + binary rare variant flags → DL input vector", tech: "Pandas, NumPy" },
];

const bloodBiomarkers = [
  { category: "Hematology", markers: ["Hemoglobin", "WBC differential", "Platelets", "RBC indices"] },
  { category: "Hormones", markers: ["Cortisol (stress)", "Testosterone", "Estradiol", "DHEA-S", "Thyroid panel (TSH, T3, T4)"] },
  { category: "Metabolic", markers: ["Glucose", "HbA1c", "Insulin", "Lipid panel", "Uric acid"] },
  { category: "Neurotransmitter Precursors", markers: ["Tryptophan/Serotonin pathway", "Tyrosine/Dopamine pathway", "GABA precursors"] },
  { category: "Inflammatory", markers: ["hsCRP", "IL-6", "TNF-α", "Homocysteine"] },
  { category: "Nutritional", markers: ["Vitamin D", "B12", "Folate", "Iron/Ferritin", "Omega-3 index"] },
  { category: "Autism-Specific", markers: ["Oxytocin", "Arginine vasopressin", "BDNF", "Glutamate/GABA ratio"] },
];

const careerMapping = [
  { trait: "High Openness", careers: ["Research Scientist", "Software Architect", "Creative Director", "Data Scientist", "Philosopher"], color: "#6C63FF" },
  { trait: "High Conscientiousness", careers: ["Surgeon", "Accountant", "Project Manager", "Air Traffic Controller", "Pharmacist"], color: "#00C49F" },
  { trait: "High Attention to Detail (Autism)", careers: ["Cybersecurity Analyst", "Quality Assurance", "Actuary", "Taxonomist", "Database Administrator"], color: "#FFD166" },
  { trait: "High Fluid Intelligence", careers: ["Mathematician", "AI/ML Engineer", "Theoretical Physicist", "Chess Grandmaster", "Epidemiologist"], color: "#4ECDC4" },
  { trait: "Low Extraversion", careers: ["Writer", "Librarian", "Translator", "Archivist", "Remote Software Engineer"], color: "#F8961E" },
];

const ethicsIssues = [
  { severity: "HIGH", issue: "Genetic determinism fallacy", detail: "Personality PGS explain only 5–15% of variance. Environment, culture, and experience dominate. The system must never present genetic results as fate, only as probabilistic tendencies." },
  { severity: "HIGH", issue: "Bias & fairness in facial recognition", detail: "Current FER models show 10–34% higher error rates for darker skin tones and non-Western facial features. Mandatory fairness auditing across all demographic groups before deployment." },
  { severity: "HIGH", issue: "Informed consent & data sovereignty", detail: "Genomic data is uniquely re-identifiable. GDPR Article 9 classifies it as special category data. Require explicit consent, right to erasure, and local processing where possible." },
  { severity: "MEDIUM", issue: "Autism stigmatization risk", detail: "Framing autism traits as career constraints could reinforce discrimination. The system must emphasize cognitive strengths and neurodiversity as assets." },
  { severity: "MEDIUM", issue: "Population generalizability", detail: "Most GWAS data is European-ancestry (>80%). Polygenic scores trained on European populations may perform poorly for individuals of Middle Eastern, African, or Asian ancestry. Requires diverse training data." },
  { severity: "MEDIUM", issue: "Self-fulfilling prophecy", detail: "Career recommendations could limit aspiration if users over-trust AI outputs. Include disclaimers and emphasize the system as one input among many." },
  { severity: "LOW", issue: "Blood sample device security", detail: "A point-of-care device handling genetic samples requires certified lab protocols, chain of custody, and anti-contamination measures." },
];

const roadmap = [
  { phase: "Phase 1", duration: "0–6 months", title: "Foundation & Data Collection", tasks: ["Literature review completion", "IRB/ethics approval", "Partner with autism clinics (Egypt/MENA region)", "Dataset curation: SPARK, UK Biobank, CelebA", "Develop consent framework", "Baseline model prototyping in Jupyter"] },
  { phase: "Phase 2", duration: "6–18 months", title: "Model Development", tasks: ["Train FaceGenome-CNN on clinical data", "Implement SNP-Transformer with LDpred2 integration", "Develop BiomarkerNet with blood panel data", "Build preprocessing pipeline (GATK, PLINK2)", "Fairness auditing across demographics", "Master's thesis research contribution"] },
  { phase: "Phase 3", duration: "18–30 months", title: "Integration & Validation", tasks: ["Build FusionNet multi-modal architecture", "Validate against validated psychometric tests (NEO-PI-R)", "Pilot study with 100–200 participants", "Develop career recommendation engine (O*NET integration)", "Point-of-care blood device integration", "Publish research papers"] },
  { phase: "Phase 4", duration: "30–42 months", title: "Clinical Deployment", tasks: ["Build web + mobile application", "Deploy in educational institutions", "Longitudinal outcome tracking", "Feedback loop for model refinement", "Regulatory pathway (CE Mark / FDA if applicable)", "Open-source model release"] },
];

const techStack = [
  { layer: "Deep Learning", tools: ["PyTorch 2.x", "TensorFlow/Keras", "Hugging Face Transformers", "timm (vision models)"] },
  { layer: "Genomics", tools: ["PLINK2", "GATK 4", "LDpred2 (R)", "bigsnpr", "ANNOVAR", "VEP"] },
  { layer: "Data Processing", tools: ["NumPy", "Pandas", "SciPy", "scikit-learn", "Biopython"] },
  { layer: "Visualization", tools: ["Matplotlib", "Seaborn", "Power BI", "Plotly Dash"] },
  { layer: "Infrastructure", tools: ["Docker", "Jupyter Lab", "MongoDB", "PostgreSQL", "FastAPI"] },
  { layer: "Device (IoT)", tools: ["Raspberry Pi / NVIDIA Jetson", "OpenCV", "ONNX Runtime", "TFLite"] },
];

export default function App() {
  const [active, setActive] = useState("overview");

  return (
    <div style={{ fontFamily: "'Georgia', serif", background: "#0a0e1a", color: "#e8eaf0", minHeight: "100vh", display: "flex" }}>
      {/* Sidebar */}
      <nav style={{ width: 220, background: "#0d1220", borderRight: "1px solid #1e2a45", padding: "24px 0", position: "sticky", top: 0, height: "100vh", overflowY: "auto", flexShrink: 0 }}>
        <div style={{ padding: "0 20px 24px", borderBottom: "1px solid #1e2a45" }}>
          <div style={{ fontSize: 11, letterSpacing: 3, color: "#4a90d9", textTransform: "uppercase", marginBottom: 6 }}>Research System</div>
          <div style={{ fontSize: 16, fontWeight: "bold", color: "#fff", lineHeight: 1.3 }}>CareerMapping<br/>Genomics</div>
        </div>
        <div style={{ padding: "16px 0" }}>
          {sections.map(s => (
            <button key={s.id} onClick={() => setActive(s.id)}
              style={{ display: "block", width: "100%", textAlign: "left", padding: "10px 20px", background: active === s.id ? "#1e2a45" : "transparent", color: active === s.id ? "#4a90d9" : "#8899aa", border: "none", cursor: "pointer", fontSize: 13, borderLeft: active === s.id ? "3px solid #4a90d9" : "3px solid transparent", transition: "all 0.2s" }}>
              {s.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Main */}
      <main style={{ flex: 1, padding: "40px 48px", overflowY: "auto", maxWidth: 960 }}>

        {active === "overview" && (
          <section>
            <div style={{ fontSize: 11, letterSpacing: 3, color: "#4a90d9", textTransform: "uppercase", marginBottom: 8 }}>Deep Research & System Analysis</div>
            <h1 style={{ fontSize: 36, fontWeight: "bold", marginBottom: 8, color: "#fff" }}>CareerMapping<span style={{ color: "#4a90d9" }}>Genomics</span></h1>
            <p style={{ color: "#8899aa", fontSize: 15, lineHeight: 1.8, marginBottom: 32, maxWidth: 720 }}>
              An interdisciplinary deep learning system integrating <strong style={{ color: "#e8eaf0" }}>genomics, facial recognition, blood biomarker analysis, and psychology</strong> to predict personality traits and recommend optimal career paths — with a particular focus on neurodiversity and autism spectrum traits.
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, marginBottom: 40 }}>
              {pillars.map((p, i) => (
                <div key={i} style={{ background: "#0d1220", border: `1px solid ${p.color}33`, borderRadius: 12, padding: 20, borderTop: `3px solid ${p.color}` }}>
                  <div style={{ fontSize: 28, marginBottom: 8 }}>{p.icon}</div>
                  <div style={{ fontSize: 14, fontWeight: "bold", color: p.color, marginBottom: 6 }}>{p.title}</div>
                  <div style={{ fontSize: 12, color: "#8899aa", lineHeight: 1.6 }}>{p.desc}</div>
                </div>
              ))}
            </div>
            <div style={{ background: "#0d1220", border: "1px solid #1e2a45", borderRadius: 12, padding: 24, borderLeft: "4px solid #4a90d9" }}>
              <div style={{ fontSize: 13, fontWeight: "bold", color: "#4a90d9", marginBottom: 12 }}>PROJECT VISION</div>
              <p style={{ color: "#c8d4e0", lineHeight: 1.8, margin: 0 }}>
                When a person presents to a clinic or school, they receive: (1) a facial scan, (2) a small blood draw processed by a portable device, and (3) a brief questionnaire. Within minutes, the system outputs a detailed personality profile, a ranked list of career matches aligned with their genetic predispositions and cognitive strengths, and inclusive recommendations specifically calibrated for autism spectrum traits — transforming how societies support neurodivergent individuals in education and employment.
              </p>
            </div>
          </section>
        )}

        {active === "architecture" && (
          <section>
            <h2 style={{ fontSize: 28, color: "#fff", marginBottom: 8 }}>System Architecture</h2>
            <p style={{ color: "#8899aa", marginBottom: 32 }}>End-to-end pipeline from raw inputs to career recommendations</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {architectureLayers.map((layer, i) => (
                <div key={i} style={{ background: layer.color, border: `1px solid ${layer.border}55`, borderRadius: 10, padding: 20, borderLeft: `4px solid ${layer.border}` }}>
                  <div style={{ fontSize: 11, letterSpacing: 2, color: layer.border, marginBottom: 10, fontWeight: "bold" }}>{layer.name}</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    {layer.items.map((item, j) => (
                      <div key={j} style={{ fontSize: 13, color: "#c8d4e0", padding: "6px 12px", background: "rgba(0,0,0,0.3)", borderRadius: 6 }}>→ {item}</div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 24, background: "#0d1220", border: "1px solid #1e2a45", borderRadius: 10, padding: 20 }}>
              <div style={{ fontSize: 12, color: "#4a90d9", fontWeight: "bold", marginBottom: 8 }}>DEVICE INTEGRATION</div>
              <p style={{ color: "#8899aa", fontSize: 13, lineHeight: 1.7, margin: 0 }}>
                A portable edge device (NVIDIA Jetson Orin or equivalent) runs the camera for facial capture and connects to a lab-on-chip unit for rapid blood DNA extraction and SNP genotyping. The device processes data locally (privacy-preserving) then uploads only anonymized feature embeddings — never raw genetic data — to the cloud inference server.
              </p>
            </div>
          </section>
        )}

        {active === "modules" && (
          <section>
            <h2 style={{ fontSize: 28, color: "#fff", marginBottom: 8 }}>Deep Learning Modules</h2>
            <p style={{ color: "#8899aa", marginBottom: 32 }}>Four specialized neural networks fused into one unified prediction system</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
              {dlModels.map((m, i) => (
                <div key={i} style={{ background: "#0d1220", border: `1px solid ${m.color}44`, borderRadius: 12, padding: 24, borderTop: `3px solid ${m.color}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
                    <div>
                      <h3 style={{ margin: 0, color: m.color, fontSize: 18 }}>{m.name}</h3>
                      <div style={{ fontSize: 12, color: "#8899aa", marginTop: 2 }}>{m.type}</div>
                    </div>
                  </div>
                  <p style={{ color: "#c8d4e0", fontSize: 13, lineHeight: 1.6, marginBottom: 16 }}>{m.purpose}</p>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                    {[["Inputs", m.inputs], ["Architecture", m.arch], ["Training Data", m.dataset], ["Performance Target", m.metrics]].map(([k, v]) => (
                      <div key={k} style={{ background: "#0a0e1a", borderRadius: 8, padding: 12 }}>
                        <div style={{ fontSize: 10, color: "#4a90d9", letterSpacing: 1, textTransform: "uppercase", marginBottom: 4 }}>{k}</div>
                        <div style={{ fontSize: 12, color: "#c8d4e0", lineHeight: 1.5 }}>{v}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ marginTop: 12, padding: "8px 12px", background: `${m.color}11`, borderRadius: 6, fontSize: 12, color: m.color }}>
                    ⚠ {m.note}
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {active === "data" && (
          <section>
            <h2 style={{ fontSize: 28, color: "#fff", marginBottom: 8 }}>Data Pipeline</h2>
            <p style={{ color: "#8899aa", marginBottom: 32 }}>From raw biological samples to model-ready feature vectors</p>
            <div style={{ marginBottom: 32 }}>
              <h3 style={{ color: "#4a90d9", fontSize: 16, marginBottom: 16 }}>Genomic Processing Pipeline</h3>
              {genomicPipeline.map((step, i) => (
                <div key={i} style={{ display: "flex", gap: 16, marginBottom: 8 }}>
                  <div style={{ width: 28, height: 28, background: "#1e2a45", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, color: "#4a90d9", flexShrink: 0, marginTop: 4 }}>{i + 1}</div>
                  <div style={{ background: "#0d1220", border: "1px solid #1e2a45", borderRadius: 8, padding: 14, flex: 1 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div style={{ fontSize: 14, color: "#e8eaf0", fontWeight: "bold" }}>{step.step}</div>
                      <div style={{ fontSize: 11, color: "#4a90d9", background: "#0a1628", padding: "2px 8px", borderRadius: 4 }}>{step.tech}</div>
                    </div>
                    <div style={{ fontSize: 12, color: "#8899aa", marginTop: 4 }}>{step.detail}</div>
                  </div>
                </div>
              ))}
            </div>
            <div>
              <h3 style={{ color: "#4a90d9", fontSize: 16, marginBottom: 16 }}>Key Datasets</h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                {[
                  { name: "UK Biobank", desc: "500,000 participants, genotype + phenotype + health data. Gold standard for PGS training.", size: "500k participants" },
                  { name: "SPARK Autism Cohort", desc: "Largest autism genetic study. De novo variants, family trios, behavioral phenotypes.", size: "~250k individuals" },
                  { name: "CelebA + AffectNet", desc: "2M+ facial images with attribute annotations for CNN pretraining.", size: "2M+ images" },
                  { name: "NHANES", desc: "National Health and Nutrition Examination Survey — blood biomarkers + behavioral data.", size: "~100k participants" },
                  { name: "O*NET Database", desc: "Occupational requirements, knowledge, skills, abilities for 900+ careers.", size: "900+ careers" },
                  { name: "Custom Clinical Study", desc: "IRB-approved local cohort in Egypt/MENA region for culturally-adapted validation.", size: "Target: 500+ participants" },
                ].map((d, i) => (
                  <div key={i} style={{ background: "#0d1220", border: "1px solid #1e2a45", borderRadius: 8, padding: 16 }}>
                    <div style={{ fontSize: 14, color: "#4a90d9", fontWeight: "bold", marginBottom: 4 }}>{d.name}</div>
                    <div style={{ fontSize: 12, color: "#8899aa", lineHeight: 1.5, marginBottom: 8 }}>{d.desc}</div>
                    <div style={{ fontSize: 11, color: "#00C49F" }}>{d.size}</div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {active === "genomics" && (
          <section>
            <h2 style={{ fontSize: 28, color: "#fff", marginBottom: 8 }}>Genomic Analysis</h2>
            <p style={{ color: "#8899aa", marginBottom: 24 }}>The science behind genotype-to-phenotype prediction</p>
            <div style={{ background: "#0d1220", border: "1px solid #1e2a45", borderRadius: 12, padding: 20, marginBottom: 24 }}>
              <div style={{ fontSize: 13, color: "#4a90d9", fontWeight: "bold", marginBottom: 8 }}>SCIENTIFIC FOUNDATION</div>
              <p style={{ color: "#c8d4e0", fontSize: 13, lineHeight: 1.8, margin: 0 }}>
                Personality traits show heritability of 40–60% (twin studies). Genome-Wide Association Studies (GWAS) have identified thousands of common SNPs associated with Big Five traits, educational attainment, and cognitive ability. However, each individual SNP has tiny effect size (OR ~1.01–1.05). Polygenic Scores (PGS) aggregate effects of hundreds of thousands of SNPs into a single predictive score, currently explaining 5–15% of personality variance — useful as one signal among many, not deterministic. For autism, there is substantial genetic overlap with intelligence and educational attainment (Dice coefficient = 0.91 with intelligence SNPs, per bidirectional GWAS studies).
              </p>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>
              {[
                { title: "Common Variants (GWAS/PGS)", points: ["500k–1M SNPs per individual", "LDpred2-auto for Bayesian polygenic scoring", "Big Five traits: r = 0.22–0.38 with PGS", "Educational attainment: R² ≈ 0.13–0.16", "Cross-ancestry portability remains limited"] },
                { title: "Rare Variants (WES/WGS)", points: ["1000s of functional variants per individual", "CADD score > 20 = likely deleterious", "Known autism genes: >100 (SHANK3, CHD8, etc.)", "De novo variants: higher impact, ~3 per exome", "Gene burden tests: SKAT-O, BURDEN"] },
              ].map((box, i) => (
                <div key={i} style={{ background: "#0d1220", border: "1px solid #1e2a45", borderRadius: 10, padding: 16 }}>
                  <div style={{ fontSize: 13, color: "#00C49F", fontWeight: "bold", marginBottom: 12 }}>{box.title}</div>
                  {box.points.map((p, j) => <div key={j} style={{ fontSize: 12, color: "#c8d4e0", padding: "4px 0", borderBottom: "1px solid #1e2a45" }}>• {p}</div>)}
                </div>
              ))}
            </div>
            <div style={{ background: "#0d1220", border: "1px solid #FFD16644", borderRadius: 12, padding: 20 }}>
              <div style={{ fontSize: 13, color: "#FFD166", fontWeight: "bold", marginBottom: 8 }}>AUTISM-SPECIFIC GENOMICS</div>
              <p style={{ color: "#c8d4e0", fontSize: 13, lineHeight: 1.8, margin: 0 }}>
                Research shows 12,000+ SNPs are shared between autism and intelligence, and 12,000 shared with educational attainment (MiXeR analysis). This genetic overlap helps explain why autism frequently co-occurs with exceptional cognitive abilities in specific domains. The system leverages this genetic architecture to identify cognitive strengths — pattern recognition, systematic thinking, deep focus — and match them to careers where these traits are assets rather than obstacles.
              </p>
            </div>
          </section>
        )}

        {active === "facial" && (
          <section>
            <h2 style={{ fontSize: 28, color: "#fff", marginBottom: 8 }}>Facial Recognition Module</h2>
            <p style={{ color: "#8899aa", marginBottom: 24 }}>Morphometric + dynamic emotion tracking for personality inference</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>
              {[
                { title: "Static Morphometry", color: "#FFD166", items: ["3D facial landmark detection (468 points via MediaPipe)", "Facial symmetry index", "Canthal tilt, jaw geometry, orbital depth", "Skin texture analysis (pore density, pigmentation)", "Pre-trained on FER+ and AffectNet"] },
                { title: "Dynamic Emotion Tracking", color: "#6C63FF", items: ["FER (Facial Emotion Recognition) while watching stimuli", "Micro-expression detection (Action Units, Ekman)", "Temporal attention patterns", "Emotional reactivity profile", "Validated by MIT CCI physiognomy study"] },
              ].map((box, i) => (
                <div key={i} style={{ background: "#0d1220", border: `1px solid ${box.color}44`, borderRadius: 10, padding: 16, borderTop: `3px solid ${box.color}` }}>
                  <div style={{ fontSize: 13, color: box.color, fontWeight: "bold", marginBottom: 12 }}>{box.title}</div>
                  {box.items.map((item, j) => <div key={j} style={{ fontSize: 12, color: "#c8d4e0", padding: "4px 0", borderBottom: "1px solid #1e2a45" }}>→ {item}</div>)}
                </div>
              ))}
            </div>
            <div style={{ background: "#1e2410", border: "1px solid #FF6B6B44", borderRadius: 12, padding: 20 }}>
              <div style={{ fontSize: 13, color: "#FF6B6B", fontWeight: "bold", marginBottom: 8 }}>⚠ CRITICAL LIMITATIONS</div>
              <p style={{ color: "#c8d4e0", fontSize: 13, lineHeight: 1.8, margin: 0 }}>
                Facial recognition for personality prediction achieves only modest accuracy (Pearson r ≈ 0.2–0.4 for individual Big Five traits). Current literature shows deep learning can predict ~23% of personal attributes better than chance from facial pixels, but effect sizes are small. Critically: accuracy varies significantly across demographic groups, with documented higher error rates for people of color. In this project, facial features serve as ONE signal supplementing genetics and biomarkers — NOT as a standalone predictor.
              </p>
            </div>
          </section>
        )}

        {active === "blood" && (
          <section>
            <h2 style={{ fontSize: 28, color: "#fff", marginBottom: 8 }}>Blood Biomarker Analysis</h2>
            <p style={{ color: "#8899aa", marginBottom: 24 }}>Point-of-care blood analysis linked to neurological and behavioral phenotypes</p>
            <div style={{ background: "#0d1220", border: "1px solid #FF6B6B44", borderRadius: 12, padding: 20, marginBottom: 24 }}>
              <div style={{ fontSize: 13, color: "#FF6B6B", fontWeight: "bold", marginBottom: 8 }}>DEVICE CONCEPT</div>
              <p style={{ color: "#c8d4e0", fontSize: 13, lineHeight: 1.8, margin: 0 }}>
                A fingerpick blood sample (~0.5mL) is processed by a portable lab-on-chip device. Within 15–20 minutes, it outputs a panel of ~80 biomarkers. DNA is extracted for SNP genotyping (low-pass sequencing) while metabolites, hormones, and cell counts are measured simultaneously via electrochemical sensors and microfluidic immunoassays. The device connects via USB/Bluetooth to the main inference computer.
              </p>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {bloodBiomarkers.map((cat, i) => (
                <div key={i} style={{ background: "#0d1220", border: "1px solid #1e2a45", borderRadius: 8, padding: 14 }}>
                  <div style={{ fontSize: 12, color: "#FF6B6B", fontWeight: "bold", marginBottom: 8 }}>{cat.category}</div>
                  {cat.markers.map((m, j) => (
                    <div key={j} style={{ fontSize: 12, color: "#c8d4e0", padding: "3px 0" }}>• {m}</div>
                  ))}
                </div>
              ))}
            </div>
            <div style={{ marginTop: 24, background: "#0d1220", border: "1px solid #4ECDC444", borderRadius: 12, padding: 20 }}>
              <div style={{ fontSize: 13, color: "#4ECDC4", fontWeight: "bold", marginBottom: 8 }}>AUTISM-SPECIFIC BIOMARKERS</div>
              <p style={{ color: "#c8d4e0", fontSize: 13, lineHeight: 1.8, margin: 0 }}>
                Research has identified several blood markers with elevated diagnostic value for ASD: Oxytocin (social bonding regulation), BDNF (neural plasticity), glutamate/GABA ratio (excitatory/inhibitory balance), and inflammatory cytokines (IL-6, TNF-α). Combined with genetic PGS for autism, these biomarkers can achieve AUC 0.72–0.85 for autism spectrum classification — providing a biological foundation for the career recommendations rather than relying purely on behavioral assessment.
              </p>
            </div>
          </section>
        )}

        {active === "career" && (
          <section>
            <h2 style={{ fontSize: 28, color: "#fff", marginBottom: 8 }}>Career Mapping Engine</h2>
            <p style={{ color: "#8899aa", marginBottom: 24 }}>From personality profiles to O*NET-grounded career recommendations</p>
            <div style={{ background: "#0d1220", border: "1px solid #1e2a45", borderRadius: 12, padding: 20, marginBottom: 24 }}>
              <div style={{ fontSize: 13, color: "#4a90d9", fontWeight: "bold", marginBottom: 8 }}>SCORING METHODOLOGY</div>
              <p style={{ color: "#c8d4e0", fontSize: 13, lineHeight: 1.8, margin: 0 }}>
                Each career in the O*NET database has validated requirements across Knowledge, Skills, Abilities, and Work Styles (KSAWS). The system maps the individual's inferred Big Five profile, cognitive scores, and autism-spectrum trait quantification onto the O*NET KSAWS space using cosine similarity. A ranked list of the top 20 career matches is returned, each with a compatibility score, required education pathway, and an explanation of why the match was made — highlighting the individual's strengths.
              </p>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 24 }}>
              {careerMapping.map((row, i) => (
                <div key={i} style={{ background: "#0d1220", border: `1px solid ${row.color}44`, borderRadius: 10, padding: 16, display: "flex", gap: 20, alignItems: "flex-start" }}>
                  <div style={{ background: `${row.color}22`, color: row.color, padding: "6px 12px", borderRadius: 6, fontSize: 12, fontWeight: "bold", flexShrink: 0, whiteSpace: "nowrap" }}>{row.trait}</div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                    {row.careers.map((c, j) => (
                      <span key={j} style={{ background: "#0a0e1a", border: "1px solid #2a3a50", borderRadius: 20, padding: "4px 12px", fontSize: 12, color: "#c8d4e0" }}>{c}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <div style={{ background: "#0d1220", border: "1px solid #00C49F44", borderRadius: 12, padding: 20 }}>
              <div style={{ fontSize: 13, color: "#00C49F", fontWeight: "bold", marginBottom: 8 }}>NEURODIVERSITY FOCUS</div>
              <p style={{ color: "#c8d4e0", fontSize: 13, lineHeight: 1.8, margin: 0 }}>
                The system explicitly identifies autism-associated cognitive strengths: hyper-systemizing, exceptional working memory for specific domains, pattern detection, attention to procedural detail, and reduced susceptibility to social conformity bias. These are mapped to careers where these traits create genuine competitive advantage — not just "suitable" careers, but careers where neurodivergent individuals can <em>excel and lead</em>.
              </p>
            </div>
          </section>
        )}

        {active === "ethics" && (
          <section>
            <h2 style={{ fontSize: 28, color: "#fff", marginBottom: 8 }}>Ethics & Limitations</h2>
            <p style={{ color: "#8899aa", marginBottom: 24 }}>Critical considerations before any clinical or educational deployment</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {ethicsIssues.map((item, i) => {
                const colors = { HIGH: "#FF6B6B", MEDIUM: "#FFD166", LOW: "#00C49F" };
                return (
                  <div key={i} style={{ background: "#0d1220", border: `1px solid ${colors[item.severity]}44`, borderRadius: 10, padding: 16 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
                      <span style={{ background: colors[item.severity], color: "#0a0e1a", padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: "bold" }}>{item.severity}</span>
                      <span style={{ fontSize: 14, color: colors[item.severity], fontWeight: "bold" }}>{item.issue}</span>
                    </div>
                    <p style={{ fontSize: 13, color: "#c8d4e0", lineHeight: 1.7, margin: 0 }}>{item.detail}</p>
                  </div>
                );
              })}
            </div>
            <div style={{ marginTop: 24, background: "#0d1220", border: "1px solid #4a90d9", borderRadius: 12, padding: 20 }}>
              <div style={{ fontSize: 13, color: "#4a90d9", fontWeight: "bold", marginBottom: 8 }}>REQUIRED SAFEGUARDS</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {["IRB approval from an accredited ethics committee", "Written informed consent with genomic data rights explanation", "Demographic bias auditing with stratified fairness metrics", "No career disqualification — recommendations only", "All outputs include confidence intervals & uncertainty", "Right to erasure of all biological and genetic data", "Diverse training data including MENA-ancestry cohorts", "Independent ethics advisory board review"].map((s, i) => (
                  <div key={i} style={{ fontSize: 12, color: "#c8d4e0", padding: "6px 10px", background: "#0a0e1a", borderRadius: 6 }}>✓ {s}</div>
                ))}
              </div>
            </div>
          </section>
        )}

        {active === "roadmap" && (
          <section>
            <h2 style={{ fontSize: 28, color: "#fff", marginBottom: 8 }}>Research Roadmap</h2>
            <p style={{ color: "#8899aa", marginBottom: 32 }}>42-month phased plan from foundation to clinical deployment</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 20, marginBottom: 40 }}>
              {roadmap.map((phase, i) => {
                const phaseColors = ["#4a90d9", "#00C49F", "#FFD166", "#FF6B6B"];
                return (
                  <div key={i} style={{ background: "#0d1220", border: `1px solid ${phaseColors[i]}44`, borderRadius: 12, padding: 20, borderLeft: `4px solid ${phaseColors[i]}` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
                      <div>
                        <span style={{ fontSize: 11, color: phaseColors[i], letterSpacing: 2, fontWeight: "bold" }}>{phase.phase} • {phase.duration}</span>
                        <h3 style={{ margin: "4px 0 0", color: "#e8eaf0", fontSize: 16 }}>{phase.title}</h3>
                      </div>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                      {phase.tasks.map((task, j) => (
                        <div key={j} style={{ fontSize: 12, color: "#c8d4e0", padding: "6px 10px", background: "#0a0e1a", borderRadius: 6 }}>▸ {task}</div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
            <h3 style={{ color: "#4a90d9", fontSize: 16, marginBottom: 16 }}>Technology Stack</h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
              {techStack.map((stack, i) => (
                <div key={i} style={{ background: "#0d1220", border: "1px solid #1e2a45", borderRadius: 8, padding: 14 }}>
                  <div style={{ fontSize: 11, color: "#4a90d9", fontWeight: "bold", marginBottom: 8, letterSpacing: 1 }}>{stack.layer.toUpperCase()}</div>
                  {stack.tools.map((t, j) => <div key={j} style={{ fontSize: 12, color: "#8899aa", padding: "2px 0" }}>• {t}</div>)}
                </div>
              ))}
            </div>
          </section>
        )}

      </main>
    </div>
  );
}
