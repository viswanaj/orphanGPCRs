# Leveraging Agentic AI towards the study of "undruggable" receptors.

In this project, I leveraged the ability for AI to rapidly compile and deploy code to study the evolution of GPCRs, which are the **most clinically relevant targets to treat disease**. I focused on orphan GPCRs, which are receptors thought to be "undruggable". I started by searching SwissProt (protein database) for full sequences of human GPCRs. Then, I deployed multiple multiple sequence alignment (MSA) algorithms to construct a phylogenetic tree -- *this is the advantage of agentic AI* -- without it, I could not have implemented multiple MSAs in parallel to study evolutionary relationships. 

In parallel, I attempted to use txGemma to predict ligands for each orphan receptor. To do this, though, I first wanted to validate the ability for txGemma to predict ligand affinities for GPCRs with a well-characterized pharmacology. From a preliminary screen, txGemma performed poorly in predicting the affinities for GPCR-ligand pairs. Next steps involve re-visiting my implementation of txGemma, and fine-tuning the model on a dataset which contains experimentally validated ligand-receptor affinities. 

<img width="1000" height="750" alt="image" src="https://github.com/user-attachments/assets/9ead57be-6fab-46a4-8c3f-b6dbbe3656ef" />




<img width="1100" height="1100" alt="image" src="https://github.com/user-attachments/assets/0be6ee16-5535-4a86-829e-5fa981b4964b" />


**Observation 1** Orphan GPCRs seems to have evolved in bursts. This make sense for two reasons. I am still investigating these punctuated evolutionary bursts. 

**Observation 2** The so-called "unknown" GPCRs are largely olfactory GPCRs. They are rich and an expansive group of proteins. 

**Note 1** There is inherent error in any MSA algorithm. Therefore, the overlapping areas of agreement between different various algorithms is more likely to be correct. In v2 of this project, I only used MAFFT. There is an entire suite of MSA algorithms which I need to implement. Then, I would compare the constructed phylogenetic trees and would need to then find a "most parsimonious" tree. 

**Note 2** At a first-pass, txGemma does not predict already known ligand-GPCR affinities correctly. This is likely due to my implementation. However, it is compelling to fine-tune txGemma for GPCR-centric drug discovery. This makes sense, intuitively. It is not unlike someone who has been trained in all of human literature, only to be asked to reproduce the works of Shakespeare -- knowledgeable as they might be, their generalist training has made them inept at the very thing they are being asked to do. However, with some reminders of Sonnet 5 and King Lear, they may more effectively recall and therefore, reproduce Shakespeare. 

This analysis reveals that orphan GPCRs evolved in "bursts" - there are period of time which are marked by an explosion of orphan GPCRs, followed by periods of time where so-called "druggable" GPCRs evolved. A natural next step for this project would be to analyze orphan GPCRs across species: for each genome, plot the number of orphan GPCRs (and optionally normalize by genome size) across evolutionary time, comparing trends across diverse organisms. This could highlight bursts of GPCR innovation, lineage-specific expansions, and relationships between genome size and the prevalence of orphan receptors.

