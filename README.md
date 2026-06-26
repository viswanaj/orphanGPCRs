# Leveraging Agentic AI towards the study of "undruggable" receptors.

In this project, I leveraged the ability for AI to rapidly compile and deploy code to study the evolution of GPCRs, which are the **most clinically relevant targets to treat disease**. I focused on orphan GPCRs, which are receptors thought to be "undruggable". I started by searching SwissProt (protein database) for full sequences of human GPCRs. Then, I deployed multiple multiple sequence alignment (MSA) algorithms to construct a phylogenetic tree -- *this is the advantage of agentic AI* -- without it, I could not have implemented multiple MSAs in parallel to study evolutionary relationships. 


<img width="1000" height="750" alt="image" src="https://github.com/user-attachments/assets/9ead57be-6fab-46a4-8c3f-b6dbbe3656ef" />




<img width="1100" height="1100" alt="image" src="https://github.com/user-attachments/assets/0be6ee16-5535-4a86-829e-5fa981b4964b" />


**Observation 1** Orphan GPCRs seems to have evolved in bursts. This make sense for two reasons. I am still investigating these punctuated evolutionary bursts. 

**Observation 2** The so-called "unknown" GPCRs are largely olfactory GPCRs. They are rich and an expansive group of proteins. 

**Note 1** There is inherent error in any MSA algorithm. Therefore, the overlapping areas of agreement between different various algorithms is more likely to be correct. In v2 of this project, I only used MAFFT. There is an entire suite of MSA algorithms which I need to implement. Then, I would compare the constructed phylogenetic trees and would need to then find a "most parsimonious" 
This analysis reveals that orphan GPCRs evolved in "bursts" - there are period of time which are marked by an explosion of orphan GPCRs, followed by periods of time where so-called "druggable" GPCRs evolved. A natural next step for this project would be to analyze orphan GPCRs across species: for each genome, plot the number of orphan GPCRs (and optionally normalize by genome size) across evolutionary time, comparing trends across diverse organisms. This could highlight bursts of GPCR innovation, lineage-specific expansions, and relationships between genome size and the prevalence of orphan receptors.

**Note 2** I ultimately did employ multiple MSA algorithms to assess how feasible the "burst" phenomenon is, and it seems to be conserved across several different kinds of alignment algorithms. However, in informal terms, this phenomenon may merely be an artifact of the alignment algorithms themselves; Whenever two (or more) sequences are, to a similar degree, not readily related to a given clade of receptor, the algorithm may group these sequences into their own distinct clade...which would then give the appearance of a "burst" of orphan receptors. I will need to investigate this further. 

