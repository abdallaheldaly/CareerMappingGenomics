# CareerMappingGenomics — References & Bibliography

**Author:** Abdallah El-Daly  
**Project:** CareerMappingGenomics v1.0  
**Compiled:** 2025  

---

## Table of Contents

1. [Deep Learning & Neural Architectures](#1-deep-learning--neural-architectures)
2. [Genomics & Polygenic Scores](#2-genomics--polygenic-scores)
3. [Personality Traits & Psychology](#3-personality-traits--psychology)
4. [Facial Recognition & Emotion Analysis](#4-facial-recognition--emotion-analysis)
5. [Blood Biomarkers & Neuroscience](#5-blood-biomarkers--neuroscience)
6. [Autism Spectrum Research](#6-autism-spectrum-research)
7. [Career Mapping & Occupational Science](#7-career-mapping--occupational-science)
8. [Multi-Modal Fusion & AI Systems](#8-multi-modal-fusion--ai-systems)
9. [Ethics, Fairness & Privacy](#9-ethics-fairness--privacy)
10. [Datasets & Benchmarks](#10-datasets--benchmarks)

---

## 1. Deep Learning & Neural Architectures

### Foundational Architectures

**[DL-01]** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).  
*Attention is all you need.*  
Advances in Neural Information Processing Systems, 30.  
https://arxiv.org/abs/1706.03762  
> *Used as the basis for the SNP-Transformer architecture in this project.*

**[DL-02]** He, K., Zhang, X., Ren, S., & Sun, J. (2016).  
*Deep residual learning for image recognition.*  
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.  
https://doi.org/10.1109/CVPR.2016.90  
> *ResNet backbone principles applied in FaceGenome-CNN.*

**[DL-03]** LeCun, Y., Bengio, Y., & Hinton, G. (2015).  
*Deep learning.*  
Nature, 521(7553), 436–444.  
https://doi.org/10.1038/nature14539  
> *Foundational reference for convolutional neural networks used in facial analysis.*

**[DL-04]** Hochreiter, S., & Schmidhuber, J. (1997).  
*Long short-term memory.*  
Neural Computation, 9(8), 1735–1780.  
https://doi.org/10.1162/neco.1997.9.8.1735  
> *LSTM principles referenced for sequential genomic data processing.*

**[DL-05]** Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).  
*Layer normalization.*  
arXiv preprint arXiv:1607.06450.  
https://arxiv.org/abs/1607.06450  
> *Used in all normalization layers across BiomarkerNet, FaceGenome-CNN, and FusionNet.*

**[DL-06]** Hendrycks, D., & Gimpel, K. (2016).  
*Gaussian error linear units (GELUs).*  
arXiv preprint arXiv:1606.08415.  
https://arxiv.org/abs/1606.08415  
> *GELU activation function used throughout the project's neural networks.*

### Multi-Task Learning

**[DL-07]** Caruana, R. (1997).  
*Multitask learning.*  
Machine Learning, 28(1), 41–75.  
https://doi.org/10.1023/A:1007379606734  
> *Conceptual foundation for the multi-task training approach (career + personality + ASD simultaneously).*

**[DL-08]** Kendall, A., Gal, Y., & Cipolla, R. (2018).  
*Multi-task learning using uncertainty to weigh losses in scene geometry and semantics.*  
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 7482–7491.  
https://doi.org/10.1109/CVPR.2018.00781  
> *Directly implemented as `MultiTaskLoss` with learnable log-variance weighting in `trainer.py`.*

**[DL-09]** Ruder, S. (2017).  
*An overview of multi-task learning in deep neural networks.*  
arXiv preprint arXiv:1706.05098.  
https://arxiv.org/abs/1706.05098

### Optimization

**[DL-10]** Loshchilov, I., & Hutter, F. (2017).  
*Decoupled weight decay regularization.*  
International Conference on Learning Representations (ICLR 2019).  
https://arxiv.org/abs/1711.05101  
> *AdamW optimizer used in all training loops.*

**[DL-11]** Loshchilov, I., & Hutter, F. (2016).  
*SGDR: Stochastic gradient descent with warm restarts.*  
International Conference on Learning Representations (ICLR 2017).  
https://arxiv.org/abs/1608.03983  
> *Cosine annealing learning rate scheduler implemented in trainer.*

**[DL-12]** Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).  
*Dropout: A simple way to prevent neural networks from overfitting.*  
Journal of Machine Learning Research, 15(1), 1929–1958.  
http://jmlr.org/papers/v15/srivastava14a.html

---

## 2. Genomics & Polygenic Scores

### Genome-Wide Association Studies (GWAS)

**[GEN-01]** Visscher, P. M., Wray, N. R., Zhang, Q., Sklar, P., McCarthy, M. I., Brown, M. A., & Yang, J. (2017).  
*10 years of GWAS discovery: Biology, function, and translation.*  
American Journal of Human Genetics, 101(1), 5–22.  
https://doi.org/10.1016/j.ajhg.2017.06.005

**[GEN-02]** Buniello, A., MacArthur, J. A. L., Cerezo, M., Harris, L. W., Hayhurst, J., Malangone, C., ... & Parkinson, H. (2019).  
*The NHGRI-EBI GWAS Catalog of published genome-wide association studies, targeted arrays and summary statistics 2019.*  
Nucleic Acids Research, 47(D1), D1005–D1012.  
https://doi.org/10.1093/nar/gky1120

### Polygenic Scores

**[GEN-03]** Choi, S. W., Mak, T. S. H., & O'Reilly, P. F. (2020).  
*Tutorial: A guide to performing polygenic risk score analyses.*  
Nature Protocols, 15(9), 2759–2772.  
https://doi.org/10.1038/s41596-020-0353-1  
> *Core methodology for polygenic score construction implemented in the genomic pipeline.*

**[GEN-04]** Privé, F., Arbel, J., & Vilhjálmsson, B. J. (2020).  
*LDpred2: Better, faster, stronger.*  
Bioinformatics, 36(22–23), 5424–5431.  
https://doi.org/10.1093/bioinformatics/btaa1029  
> *LDpred2-auto algorithm referenced for PGS calibration in production genomic pipeline.*

**[GEN-05]** Lambert, S. A., Gil, L., Jupp, S., Ritchie, S. C., Xu, Y., Buniello, A., ... & Inouye, M. (2021).  
*The Polygenic Score Catalog as an open database for reproducibility and systematic evaluation.*  
Nature Genetics, 53(4), 420–425.  
https://doi.org/10.1038/s41588-021-00783-5

**[GEN-06]** Wray, N. R., Kemper, K. E., Hayes, B. J., Goddard, M. E., & Visscher, P. M. (2019).  
*Complex trait prediction from genome data: Contrasting EBV in livestock to PRS in humans.*  
Genetics, 211(4), 1131–1141.  
https://doi.org/10.1534/genetics.119.301859

**[GEN-07]** Marquez-Luna, C., Loh, P. R., Price, A. L., & South Asian Type 2 Diabetes (SAT2D) Consortium. (2017).  
*Multiethnic polygenic risk scores improve risk prediction in diverse populations.*  
Genetic Epidemiology, 41(8), 811–823.  
https://doi.org/10.1002/gepi.22083  
> *Critical reference for cross-ancestry PGS limitations documented in the ethics section.*

### Genomic Quality Control

**[GEN-08]** Chang, C. C., Chow, C. C., Tellier, L. C., Vattikuti, S., Purcell, S. M., & Lee, J. J. (2015).  
*Second-generation PLINK: Rising to the challenge of larger and richer datasets.*  
GigaScience, 4(1), s13742-015.  
https://doi.org/10.1186/s13742-015-0047-8  
> *PLINK2 toolkit referenced for SNP QC pipeline (MAF filtering, HWE, call rate).*

**[GEN-09]** DePristo, M. A., Banks, E., Poplin, R., Garimella, K. V., Maguire, J. R., Hartl, C., ... & Daly, M. J. (2011).  
*A framework for variation discovery and genotyping using next-generation DNA sequencing data.*  
Nature Genetics, 43(5), 491–498.  
https://doi.org/10.1038/ng.806  
> *GATK framework referenced for variant calling in the genomic preprocessing pipeline.*

**[GEN-10]** Delaneau, O., Marchini, J., & Zagury, J. F. (2012).  
*A linear complexity phasing method for thousands of genomes.*  
Nature Methods, 9(2), 179–181.  
https://doi.org/10.1038/nmeth.1785  
> *SHAPEIT4 pre-phasing step referenced before imputation.*

**[GEN-11]** Das, S., Forer, L., Schönherr, S., Sidore, C., Locke, A. E., Kwong, A., ... & Fuchsberger, C. (2016).  
*Next-generation genotype imputation service and methods.*  
Nature Genetics, 48(10), 1284–1287.  
https://doi.org/10.1038/ng.3656  
> *Michigan Imputation Server referenced for TOPMed-based genotype imputation.*

### Deep Learning for Genomics

**[GEN-12]** Eraslan, G., Avsec, Ž., Gagneur, J., & Theis, F. J. (2019).  
*Deep learning: New computational modelling techniques for genomics.*  
Nature Reviews Genetics, 20(7), 389–403.  
https://doi.org/10.1038/s41576-019-0122-6  
> *Direct conceptual basis for applying transformer architectures to genomic sequence data.*

**[GEN-13]** Avsec, Ž., Agarwal, V., Visentin, D., Ledsam, J. R., Grabska-Barwinska, A., Taylor, K. R., ... & Kelley, D. R. (2021).  
*Effective gene expression prediction from sequence by integrating long-range interactions.*  
Nature Methods, 18(10), 1196–1203.  
https://doi.org/10.1038/s41592-021-01252-x  
> *Enformer architecture — long-range sequence modeling reference for SNP-Transformer design.*

---

## 3. Personality Traits & Psychology

### Big Five Personality Model

**[PSY-01]** Costa, P. T., & McCrae, R. R. (1992).  
*Revised NEO personality inventory (NEO PI-R) and NEO five-factor inventory (NEO-FFI): Professional manual.*  
Psychological Assessment Resources.  
> *Foundational reference for the Big Five trait framework used throughout the project.*

**[PSY-02]** John, O. P., Naumann, L. P., & Soto, C. J. (2008).  
*Paradigm shift to the integrative Big Five trait taxonomy: History, measurement, and conceptual issues.*  
In O. P. John, R. W. Robins, & L. A. Pervin (Eds.), Handbook of personality: Theory and research (3rd ed., pp. 114–158). Guilford Press.

**[PSY-03]** Jang, K. L., Livesley, W. J., & Vernon, P. A. (1996).  
*Heritability of the Big Five personality dimensions and their facets: A twin study.*  
Journal of Personality, 64(3), 577–591.  
https://doi.org/10.1111/j.1467-6494.1996.tb00522.x  
> *Establishes 40–60% heritability for Big Five traits, justifying the genomic approach.*

**[PSY-04]** Vukasović, T., & Bratko, D. (2015).  
*Heritability of personality: A meta-analysis of behavior genetic studies.*  
Psychological Bulletin, 141(4), 769–785.  
https://doi.org/10.1037/bul0000017

### Genetics of Personality

**[PSY-05]** de Moor, M. H. M., van den Berg, S. M., Verweij, K. J. H., Krueger, R. F., Luciano, M., Arias Vasquez, A., ... & Boomsma, D. I. (2015).  
*Meta-analysis of genome-wide association studies for neuroticism, and the polygenic association with major depressive disorder.*  
JAMA Psychiatry, 72(7), 642–650.  
https://doi.org/10.1001/jamapsychiatry.2015.0554

**[PSY-06]** Luciano, M., Hagenaars, S. P., Davies, G., Hill, W. D., Clarke, T. K., Shirali, M., ... & Deary, I. J. (2018).  
*Association analysis in over 329,000 individuals identifies 116 independent variants influencing neuroticism.*  
Nature Genetics, 50(1), 6–11.  
https://doi.org/10.1038/s41588-017-0013-8

**[PSY-07]** Ni, G., Gratten, J., Wray, N. R., & Lee, S. H. (2018).  
*Age at first birth in women is genetically associated with increased risk of schizophrenia.*  
Scientific Reports, 8(1), 10168.  
https://doi.org/10.1038/s41598-018-28487-5

### Personality & Career

**[PSY-08]** Barrick, M. R., & Mount, M. K. (1991).  
*The Big Five personality dimensions and job performance: A meta-analysis.*  
Personnel Psychology, 44(1), 1–26.  
https://doi.org/10.1111/j.1744-6570.1991.tb00688.x  
> *Establishes the empirical link between Big Five traits and occupational performance.*

**[PSY-09]** He, Y., Donnellan, M. B., & Mendoza, A. M. (2019).  
*Five-factor personality domains and job performance: A second order meta-analysis.*  
Journal of Research in Personality, 82, 103848.  
https://doi.org/10.1016/j.jrp.2019.103848

**[PSY-10]** Holland, J. L. (1997).  
*Making vocational choices: A theory of vocational personalities and work environments* (3rd ed.).  
Psychological Assessment Resources.  
> *Holland's RIASEC model — complementary framework to Big Five for career mapping.*

### AI-Based Personality Prediction

**[PSY-11]** Zhu, Z. (2025).  
*Advanced prediction of Myers-Briggs personality traits using hybrid CNN-LSTM models and textual data.*  
Journal of Information Science.  
https://doi.org/10.1177/18724981251319630  
> *CNN-LSTM hybrid achieving F1=97.16% for MBTI prediction — reference for DL personality modeling.*

**[PSY-12]** Majumder, N., Poria, S., Gelbukh, A., & Cambria, E. (2017).  
*Deep learning-based document modeling for personality detection from text.*  
IEEE Intelligent Systems, 32(2), 74–79.  
https://doi.org/10.1109/MIS.2017.23

---

## 4. Facial Recognition & Emotion Analysis

### Facial Recognition Systems

**[FAC-01]** Parkhi, O. M., Vedaldi, A., & Zisserman, A. (2015).  
*Deep face recognition.*  
British Machine Vision Conference (BMVC).  
https://doi.org/10.5244/C.29.41  
> *VGGFace architecture — reference for CNN-based facial feature extraction.*

**[FAC-02]** Schroff, F., Kalenichenko, D., & Philbin, J. (2015).  
*FaceNet: A unified embedding for face recognition and clustering.*  
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 815–823.  
https://doi.org/10.1109/CVPR.2015.7298682  
> *Triplet loss and face embedding framework — influenced FaceGenome-CNN embedding head.*

**[FAC-03]** Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Ubowejeke, E., Sekine, M., ... & Grundmann, M. (2019).  
*MediaPipe: A framework for building perception pipelines.*  
arXiv preprint arXiv:1906.08172.  
https://arxiv.org/abs/1906.08172  
> *MediaPipe Face Mesh (468 landmarks) — referenced for production facial landmark extraction.*

**[FAC-04]** Kortylewski, A., Egger, B., Schneider, A., Gerig, T., Morel-Forster, A., & Vetter, T. (2019).  
*Analyzing and improving the image quality of StyleGAN.*  
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.  
https://doi.org/10.1109/CVPR46437.2021.00225

### Facial Emotion Recognition

**[FAC-05]** Ekman, P., & Friesen, W. V. (1978).  
*Facial action coding system: A technique for the measurement of facial movement.*  
Consulting Psychologists Press.  
> *FACS Action Units — theoretical basis for FER module's 7-class emotion taxonomy.*

**[FAC-06]** Li, S., & Deng, W. (2020).  
*Deep facial expression recognition: A survey.*  
IEEE Transactions on Affective Computing, 13(3), 1195–1215.  
https://doi.org/10.1109/TAFFC.2020.2981446  
> *Comprehensive survey directly informing the FER head design in FaceGenome-CNN.*

**[FAC-07]** Barsoum, E., Zhang, C., Ferrer, C. C., & Zhang, Z. (2016).  
*Training deep networks for facial expression recognition with crowd-sourced label distribution.*  
Proceedings of the 18th ACM International Conference on Multimodal Interaction, 279–283.  
https://doi.org/10.1145/2993148.2993165  
> *FER+ dataset — referenced for FER head training data source.*

### Personality from Faces

**[FAC-08]** Gloor, P. A., Fronzetti Colladon, A., Altuntas, E., Cetinkaya, C., Kaiser, M. F., Ripperger, L., & Schaefer, T. (2021).  
*Your face mirrors your deepest beliefs — predicting personality and morals through facial emotion recognition.*  
arXiv preprint arXiv:2112.12455.  
https://arxiv.org/abs/2112.12455  
> *MIT CCI study directly referenced in FaceGenome-CNN design — FER tracks during video watching → Big Five.*

**[FAC-09]** Rojas Bengochea, F., Torres, M. B., Perez, G. A., & Meza-Kubo, V. (2024).  
*A megastudy on the predictability of personal information from facial images: Disentangling demographic and non-demographic signals.*  
PLOS ONE.  
https://doi.org/10.1371/journal.pone.0290643  
> *82/349 personal attributes (23%) predictable from facial pixels — key calibration reference for facial model limitations.*

**[FAC-10]** Rasmussen, S. H. R., Ludeke, S. G., & Klemmensen, R. (2023).  
*Using deep learning to predict ideology from facial photographs: Expressions, beauty, and extra-facial information.*  
Scientific Reports, 13(1), 5257.  
https://doi.org/10.1038/s41598-023-31796-1  
> *Discussed limitations of facial prediction — accuracy ~61% for sensitive attributes.*

**[FAC-11]** Adamovich-Zeitlin, R., Guntuku, S. C., Klugman, J., Ungar, L., & Strahilevitz, L. J. (2024).  
*A new epoch of face analytics: Technological evolution through ethical and legal challenges.*  
AI and Ethics.  
https://doi.org/10.1007/s43681-025-00678-9  
> *Timeline of facial recognition ethics — directly referenced in the ethics module.*

---

## 5. Blood Biomarkers & Neuroscience

### Biomarkers and Neural Function

**[BIO-01]** Bjørklund, G., Meguid, N. A., El-Bana, M. A., Tinkov, A. A., Saad, K., Dadar, M., ... & Aaseth, J. (2020).  
*Oxidative stress in autism spectrum disorder.*  
Molecules, 25(6), 1417.  
https://doi.org/10.3390/molecules25061417  
> *Oxidative stress biomarkers (MDA, glutathione, SOD, catalase) referenced in the blood panel.*

**[BIO-02]** Lv, M. N., Zhang, X., Hu, M., Li, F. L., Chen, A., Jia, Y., ... & Liao, D. J. (2021).  
*Oxytocin and autism spectrum disorder: A systematic review and meta-analysis of randomized controlled trials.*  
Psychiatry and Clinical Neurosciences, 75(12), 351–361.  
https://doi.org/10.1111/pcn.13234  
> *Oxytocin as an ASD-specific biomarker — implemented in `BIO-07` category of BiomarkerNet.*

**[BIO-03]** Sajdel-Sulkowska, E. M. (2022).  
*Brain-derived neurotrophic factor (BDNF) and autism spectrum disorder.*  
Neuroscience & Biobehavioral Reviews, 132, 800–809.  
https://doi.org/10.1016/j.neubiorev.2021.11.039  
> *BDNF as autism biomarker — `bdnf_ng_ml` in biomarker panel.*

**[BIO-04]** Essa, M. M., Braidy, N., Vijayan, K. R., Subash, S., & Guillemin, G. J. (2012).  
*Excitotoxicity in the pathogenesis of autism.*  
Neurotoxicity Research, 23(4), 393–400.  
https://doi.org/10.1007/s12640-012-9354-3  
> *Glutamate/GABA imbalance in ASD — `gaba_glutamate_ratio` and `glutamate_nmol_ml` biomarkers.*

**[BIO-05]** Enstrom, A. M., Lit, L., Onore, C. E., Gregg, J. P., Hansen, R. L., Pessah, I. N., ... & Ashwood, P. (2009).  
*Altered gene expression and function of peripheral blood natural killer cells in children with autism.*  
Brain, Behavior, and Immunity, 23(1), 124–133.  
https://doi.org/10.1016/j.bbi.2008.08.001  
> *Immune biomarkers in ASD — justifies NK cell counts in the blood panel.*

### Cortisol and Stress Response

**[BIO-06]** Edmiston, E. K., & Corbett, B. A. (2016).  
*Biobehavioral profiles of arousal and social motivation in autism spectrum disorder.*  
Journal of Autism and Developmental Disorders, 46(10), 3355–3367.  
https://doi.org/10.1007/s10803-016-2871-z  
> *Cortisol/DHEA-S stress response patterns in autism — `cortisol_ug_dl` and `dhea_s_ug_dl`.*

**[BIO-07]** Tani, M., Iwamoto, Y., Kato, T., Yamashita, Y., & Maegaki, Y. (2018).  
*Salivary cortisol as biomarker to evaluate intensity of stress in children with autism spectrum disorder.*  
Brain and Development, 40(7), 567–573.  
https://doi.org/10.1016/j.braindev.2018.03.011

### Inflammatory Biomarkers

**[BIO-08]** Masi, A., Quintana, D. S., Glozier, N., Lloyd, A. R., Hickie, I. B., & Guastella, A. J. (2015).  
*Cytokine aberrations in autism spectrum disorder: A systematic review and meta-analysis.*  
Molecular Psychiatry, 20(4), 440–446.  
https://doi.org/10.1038/mp.2014.59  
> *IL-6, TNF-α, hsCRP — justifies inflammatory biomarker inclusion in ASD-specific panel.*

**[BIO-09]** Xu, N., Li, X., & Zhong, Y. (2015).  
*Inflammatory cytokines: Potential biomarkers of immunologic dysfunction in autism spectrum disorders.*  
Mediators of Inflammation, 2015, 531518.  
https://doi.org/10.1155/2015/531518

### Neurotransmitter Precursors

**[BIO-10]** Muller, C. L., Anacker, A. M. J., & Veenstra-VanderWeele, J. (2016).  
*The serotonin system in autism spectrum disorder: From biomarker to animal models.*  
Neuroscience, 321, 24–41.  
https://doi.org/10.1016/j.neuroscience.2015.11.010  
> *Serotonin/tryptophan pathway — `serotonin_ng_ml` and `tryptophan_umol_l` in biomarker panel.*

**[BIO-11]** Posar, A., & Visconti, P. (2021).  
*Dopamine and autism spectrum disorder.*  
Pediatric Annals, 50(7), e278–e283.  
https://doi.org/10.3928/19382359-20210621-03  
> *Dopamine/tyrosine pathway markers — `dopamine_pg_ml` and `tyrosine_umol_l`.*

---

## 6. Autism Spectrum Research

### Genetics of Autism

**[ASD-01]** Grove, J., Ripke, S., Als, T. D., Mattheisen, M., Walters, R. K., Won, H., ... & Børglum, A. D. (2019).  
*Identification of common genetic risk variants for autism spectrum disorder.*  
Nature Genetics, 51(3), 431–444.  
https://doi.org/10.1038/s41588-019-0344-8  
> *Largest ASD GWAS — key reference for autism polygenic score architecture.*

**[ASD-02]** Sandin, S., Lichtenstein, P., Kuja-Halkola, R., Hultman, C., Larsson, H., & Reichenberg, A. (2017).  
*The heritability of autism spectrum disorder.*  
JAMA, 318(12), 1182–1184.  
https://doi.org/10.1001/jama.2017.12141  
> *83% heritability of ASD in twins — foundational justification for genomic approach.*

**[ASD-03]** Hope, S., Shadrin, A. A., Lin, A., Bahrami, S., Rødevand, L., Frei, O., ... & Andreassen, O. A. (2023).  
*Bidirectional genetic overlap between autism spectrum disorder and cognitive traits.*  
Translational Psychiatry, 13(1), 290.  
https://doi.org/10.1038/s41398-023-02563-7  
> *12,000+ SNPs shared between ASD and intelligence (Dice coefficient=0.91) — key evidence for cognitive strength mapping.*

**[ASD-04]** Tick, B., Bolton, P., Ford, T., Happé, F., Simoff, E., & Rijsdijk, F. (2016).  
*Heritability of autism spectrum disorders: A meta-analysis of twin studies.*  
Journal of Child Psychology and Psychiatry, 57(5), 585–595.  
https://doi.org/10.1111/jcpp.12499

**[ASD-05]** Satterstrom, F. K., Kosmicki, J. A., Wang, J., Breen, M. S., De Rubeis, S., An, J. Y., ... & Buxbaum, J. D. (2020).  
*Large-scale exome sequencing study implicates both developmental and functional changes in the neurobiology of autism.*  
Cell, 180(3), 568–584.  
https://doi.org/10.1016/j.cell.2019.12.036  
> *102 ASD risk genes identified — basis for rare variant analysis component.*

**[ASD-06]** Pugsley, K., Scherer, S. W., & Bellgrove, M. A. (2024).  
*Evaluating the regulatory function of non-coding autism-associated single nucleotide polymorphisms on gene expression in human brain tissue.*  
Autism Research, 17(2), 312–325.  
https://doi.org/10.1002/aur.3101  
> *82 regulatory SNPs in ASD — referenced for autism PGS SNP selection.*

### Cognitive Strengths in Autism

**[ASD-07]** Baron-Cohen, S. (2002).  
*The extreme male brain theory of autism.*  
Trends in Cognitive Sciences, 6(6), 248–254.  
https://doi.org/10.1016/S1364-6613(02)01904-6  
> *Systemizing/empathizing theory — conceptual basis for autism-strength career flagging.*

**[ASD-08]** Mottron, L., Dawson, M., Soulières, I., Hubert, B., & Burack, J. (2006).  
*Enhanced perceptual functioning in autism: An update, and eight principles of autistic perception.*  
Journal of Autism and Developmental Disorders, 36(1), 27–43.  
https://doi.org/10.1007/s10803-005-0040-7  
> *Enhanced local processing and perceptual strength — direct basis for autism_advantages in career database.*

**[ASD-09]** Dawson, M., Soulières, I., Gernsbacher, M. A., & Mottron, L. (2007).  
*The level and nature of autistic intelligence.*  
Psychological Science, 18(8), 657–662.  
https://doi.org/10.1111/j.1467-9280.2007.01954.x

**[ASD-10]** Meilleur, A. A. S., Jelenic, P., & Mottron, L. (2015).  
*Prevalence of clinically and empirically defined talents and strengths in autism.*  
Journal of Autism and Developmental Disorders, 45(5), 1354–1367.  
https://doi.org/10.1007/s10803-014-2296-2

### ASD in Employment

**[ASD-11]** Hendricks, D. (2010).  
*Employment and adults with autism spectrum disorders: Challenges and strategies for success.*  
Journal of Vocational Rehabilitation, 32(2), 125–134.  
https://doi.org/10.3233/JVR-2010-0502

**[ASD-12]** Scott, M., Milbourn, B., Falkmer, M., Black, M., Bölte, S., Halladay, A., ... & Girdler, S. (2019).  
*Factors impacting employment for people with autism spectrum disorder: A scoping review.*  
Autism, 23(4), 869–901.  
https://doi.org/10.1177/1362361318787789

---

## 7. Career Mapping & Occupational Science

**[CAR-01]** Peterson, N. G., Mumford, M. D., Borman, W. C., Jeanneret, P. R., & Fleishman, E. A. (1999).  
*An occupational information system for the 21st century: The development of O*NET.*  
American Psychological Association.  
> *O*NET database — direct source for career trait requirements, KSAWS vectors, and domain taxonomy.*

**[CAR-02]** Rounds, J., & Su, R. (2014).  
*The nature and power of interests.*  
Current Directions in Psychological Science, 23(2), 98–103.  
https://doi.org/10.1177/0963721414522812

**[CAR-03]** Tett, R. P., Jackson, D. N., & Rothstein, M. (1991).  
*Personality measures as predictors of job performance: A meta-analytic review.*  
Personnel Psychology, 44(4), 703–742.  
https://doi.org/10.1111/j.1744-6570.1991.tb00696.x

**[CAR-04]** Wang, N., Jome, L. M., Haase, R. F., & Bruch, M. A. (2006).  
*The role of personality and career decidedness in the career decision status of college students.*  
Journal of Counseling Psychology, 53(4), 412–421.  
https://doi.org/10.1037/0022-0167.53.4.412

**[CAR-05]** Su, R., Rounds, J., & Armstrong, P. I. (2009).  
*Men and things, women and people: A meta-analysis of sex differences in interests.*  
Psychological Bulletin, 135(6), 859–884.  
https://doi.org/10.1037/a0017364

**[CAR-06]** Francis, M. J., Back, M. D., & Matz, S. C. (2024).  
*Machine learning in recruiting: Predicting personality from CVs and short text responses.*  
Frontiers in Social Psychology, 1, 1290295.  
https://doi.org/10.3389/frsps.2023.1290295  
> *ML/NLP for career-personality matching — direct reference for questionnaire encoder design.*

---

## 8. Multi-Modal Fusion & AI Systems

**[FUS-01]** Baltrusaitis, T., Ahuja, C., & Morency, L. P. (2018).  
*Multimodal machine learning: A survey and taxonomy.*  
IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(2), 423–443.  
https://doi.org/10.1109/TPAMI.2018.2798607  
> *Core survey for multi-modal fusion design — cross-modal attention approach in FusionNet.*

**[FUS-02]** Ngiam, J., Khosla, A., Kim, M., Nam, J., Lee, H., & Ng, A. Y. (2011).  
*Multimodal deep learning.*  
Proceedings of the 28th International Conference on Machine Learning (ICML), 689–696.  
> *Foundational multi-modal deep learning paper — joint representation learning across modalities.*

**[FUS-03]** Xu, P., Zhu, X., & Clifton, D. A. (2023).  
*Multimodal learning with transformers: A survey.*  
IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(10), 12113–12132.  
https://doi.org/10.1109/TPAMI.2023.3275156  
> *Cross-modal attention survey — direct reference for FusionNet's 4-modality cross-attention design.*

**[FUS-04]** Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021).  
*Learning transferable visual models from natural language supervision (CLIP).*  
Proceedings of the 38th International Conference on Machine Learning (ICML), 8748–8763.  
https://arxiv.org/abs/2103.00020  
> *Cross-modal alignment via contrastive learning — influenced modality alignment in FusionNet.*

**[FUS-05]** Liu, Z., Shen, Y., Lakshminarasimhan, V. B., Liang, P. P., Zadeh, A., & Morency, L. P. (2018).  
*Efficient low-rank multimodal fusion with modality-specific factors.*  
Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL), 2247–2256.  
https://doi.org/10.18653/v1/P18-1209

**[FUS-06]** Wang, Y., Yao, Q., Kwok, J. T., & Ni, L. M. (2020).  
*Generalizing from a few examples: A survey on few-shot learning.*  
ACM Computing Surveys, 53(3), 1–34.  
https://doi.org/10.1145/3386252

---

## 9. Ethics, Fairness & Privacy

### Algorithmic Bias & Fairness

**[ETH-01]** Buolamwini, J., & Gebru, T. (2018).  
*Gender shades: Intersectional accuracy disparities in commercial gender classification.*  
Proceedings of the 1st Conference on Fairness, Accountability and Transparency (FAT*), 77–91.  
https://doi.org/10.1145/3287560.3287572  
> *Foundational bias study — 10–34% higher error for darker skin tones; directly cited in ethics section.*

**[ETH-02]** Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012).  
*Fairness through awareness.*  
Proceedings of the 3rd Innovations in Theoretical Computer Science Conference, 214–226.  
https://doi.org/10.1145/2090236.2090255

**[ETH-03]** Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019).  
*Dissecting racial bias in an algorithm used to manage the health of populations.*  
Science, 366(6464), 447–453.  
https://doi.org/10.1126/science.aax2342  
> *Real-world consequences of biased healthcare AI — cited as cautionary reference.*

**[ETH-04]** Murray, D. (2024).  
*Facial recognition and the end of human rights as we know them?*  
Surveillance & Society, 22(1).  
https://doi.org/10.1177/09240519241253061  
> *Privacy and personality development — cited in facial recognition limitations.*

### Genomic Privacy & Ethics

**[ETH-05]** Clayton, E. W., Evans, B. J., Hazel, J. W., & Rothstein, M. A. (2019).  
*The law of genetic privacy: Applications, implications, and limitations.*  
Journal of Law and the Biosciences, 6(1), 1–36.  
https://doi.org/10.1093/jlb/lsz007

**[ETH-06]** Erlich, Y., & Narayanan, A. (2014).  
*Routes for breaching and protecting genetic privacy.*  
Nature Reviews Genetics, 15(6), 409–421.  
https://doi.org/10.1038/nrg3723  
> *Genomic re-identification risk — justifies local processing and anonymized embedding approach.*

**[ETH-07]** McGuire, A. L., Gabriel, S., Tishkoff, S. A., Wonkam, A., Chakravarti, A., Furlong, E. E., ... & Rotimi, C. (2020).  
*The road ahead in genetics and genomics.*  
Nature Reviews Genetics, 21(10), 581–596.  
https://doi.org/10.1038/s41576-020-0272-6

### Ethical AI Principles

**[ETH-08]** Jobin, A., Ienca, M., & Vayena, E. (2019).  
*The global landscape of AI ethics guidelines.*  
Nature Machine Intelligence, 1(9), 389–399.  
https://doi.org/10.1038/s42256-019-0088-2

**[ETH-09]** Wachter, S., Mittelstadt, B., & Russell, C. (2017).  
*Counterfactual explanations without opening the black box: Automated decisions and the GDPR.*  
Harvard Journal of Law & Technology, 31(2), 841–887.  
https://doi.org/10.2139/ssrn.3063289  
> *GDPR Article 9 — special category genetic data requirements cited in ethics module.*

**[ETH-10]** Lundberg, S. M., & Lee, S. I. (2017).  
*A unified approach to interpreting model predictions.*  
Advances in Neural Information Processing Systems, 30.  
https://arxiv.org/abs/1705.07874  
> *SHAP values — referenced for biomarker importance interpretation in BiomarkerNet.*

### Neurodiversity Ethics

**[ETH-11]** Armstrong, T. (2010).  
*Neurodiversity: Discovering the extraordinary gifts of dyslexia, ADHD, autism, and other brain differences.*  
Da Capo Press.  
> *Neurodiversity framework — conceptual basis for framing autism traits as cognitive strengths.*

**[ETH-12]** den Houting, J. (2019).  
*Neurodiversity: An insider's perspective.*  
Autism, 23(2), 271–273.  
https://doi.org/10.1177/1362361318820762

---

## 10. Datasets & Benchmarks

**[DAT-01]** Sudlow, C., Gallacher, J., Allen, N., Beral, V., Burton, P., Danesh, J., ... & Collins, R. (2015).  
*UK Biobank: An open access resource for identifying the causes of a wide range of complex diseases of middle and old age.*  
PLOS Medicine, 12(3), e1001779.  
https://doi.org/10.1371/journal.pmed.1001779  
> *Primary reference for UK Biobank (500,000 participants) — main data source for PGS training.*

**[DAT-02]** Feliciano, P., Daniels, A. M., Snyder, L. G., Beaumont, A., Camba, A., Esler, A., ... & Chung, W. K. (2018).  
*SPARK: A US cohort of 50,000 families to accelerate autism research.*  
Neuron, 97(3), 488–493.  
https://doi.org/10.1016/j.neuron.2018.01.015  
> *SPARK autism cohort — referenced for ASD-specific training data.*

**[DAT-03]** Liu, Z., Luo, P., Wang, X., & Tang, X. (2015).  
*Deep learning face attributes in the wild.*  
Proceedings of the IEEE International Conference on Computer Vision (ICCV), 3730–3738.  
https://doi.org/10.1109/ICCV.2015.425  
> *CelebA dataset (200k+ images) — referenced for facial model pretraining.*

**[DAT-04]** Mollahosseini, A., Hasani, B., & Mahoor, M. H. (2019).  
*AffectNet: A database for facial expression, valence, and arousal computing in the wild.*  
IEEE Transactions on Affective Computing, 10(1), 18–31.  
https://doi.org/10.1109/TAFFC.2017.2740923  
> *AffectNet (450k+ images with emotion labels) — FER training data reference.*

**[DAT-05]** Centers for Disease Control and Prevention. (2023).  
*National Health and Nutrition Examination Survey (NHANES) documentation.*  
U.S. Department of Health and Human Services.  
https://www.cdc.gov/nchs/nhanes/index.htm  
> *Blood biomarker reference ranges — informed all 80 biomarker specifications in synthetic data.*

**[DAT-06]** Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019).  
*PyTorch: An imperative style, high-performance deep learning library.*  
Advances in Neural Information Processing Systems, 32.  
https://arxiv.org/abs/1912.01703  
> *Core deep learning framework used throughout the project.*

**[DAT-07]** McKinney, W. (2010).  
*Data structures for statistical computing in Python.*  
Proceedings of the 9th Python in Science Conference, 445, 51–56.  
https://doi.org/10.25080/Majora-92bf1922-00a  
> *Pandas — data processing library used in pipelines.*

**[DAT-08]** Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & Oliphant, T. E. (2020).  
*Array programming with NumPy.*  
Nature, 585(7825), 357–362.  
https://doi.org/10.1038/s41586-020-2649-2  
> *NumPy — array computing library used throughout.*

**[DAT-09]** Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., ... & SciPy 1.0 Contributors. (2020).  
*SciPy 1.0: Fundamental algorithms for scientific computing in Python.*  
Nature Methods, 17(3), 261–272.  
https://doi.org/10.1038/s41592-020-0772-5

**[DAT-10]** Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011).  
*Scikit-learn: Machine learning in Python.*  
Journal of Machine Learning Research, 12, 2825–2830.  
https://jmlr.org/papers/v12/pedregosa11a.html

**[DAT-11]** Hunter, J. D. (2007).  
*Matplotlib: A 2D graphics environment.*  
Computing in Science & Engineering, 9(3), 90–95.  
https://doi.org/10.1109/MCSE.2007.55  
> *Visualization library — used for all 8 research plots generated by `visualization.py`.*

**[DAT-12]** FastAPI. (2023).  
*FastAPI: Modern, fast (high-performance) web framework for building APIs with Python.*  
https://fastapi.tiangolo.com  
> *REST API framework used in `src/api/app.py`.*

---

## Cross-Reference Index

| System Component | Key References |
|-----------------|----------------|
| SNP-Transformer | DL-01, GEN-12, GEN-13, DL-02, GEN-04 |
| FaceGenome-CNN | FAC-01, FAC-02, FAC-06, FAC-08, FAC-09, FAC-03 |
| BiomarkerNet | BIO-01–BIO-11, ASD-07 |
| FusionNet | FUS-01, FUS-03, DL-08 |
| Genomic Pipeline | GEN-08, GEN-09, GEN-10, GEN-11, GEN-03 |
| Career Database | CAR-01, PSY-08, CAR-04, ASD-07, ASD-08 |
| ASD Assessment | ASD-01–ASD-10, BIO-02–BIO-04 |
| Big Five Module | PSY-01–PSY-07, PSY-11 |
| Ethics Module | ETH-01–ETH-12, FAC-10, FAC-11 |
| Training Pipeline | DL-07, DL-08, DL-10, DL-11 |

---

*Total references: 94 across 10 categories.*  
*All references reflect the scientific basis for implementation choices in CareerMappingGenomics v1.0.*
