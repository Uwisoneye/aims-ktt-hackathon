# process_log.md

## Timeline
- Hour 1: Read the challenge brief, mapped required deliverables, and chose a Tier 1 baseline approach.
- Hour 2: Recreated the missing synthetic input files from the generator specification in the brief.
- Hour 3: Built the logistic regression scorer, threshold calibration, and explanation logic.
- Hour 4: Built dashboard scaffolding, generated printable PDFs, and documented product workflow trade-offs.

## Tools used
- ChatGPT (GPT-5.4 Thinking): planning, code drafting, debugging, README drafting.

## Three sample prompts used
1. "Help me convert this challenge brief into a minimal Tier 1 submission plan with the fastest possible path to working deliverables."
2. "Generate a synthetic households dataset with the exact columns from the brief and a stunting label driven by meals, water, sanitation, income, and number of under-5 children."
3. "Write a logistic regression baseline with explainable household-level drivers and printable PDF export."

## One discarded prompt
- "Build a fully polished production dashboard with sector polygons, multi-page PDF templates, and multilingual deployment."  
Discarded because it was too broad for the time budget and risked over-scoping.

## Hardest decision
The hardest decision was choosing a simple end-to-end baseline instead of chasing a more sophisticated geospatial model. The brief heavily rewards completeness, clarity, and local usability, so I prioritized a logistic regression scorer with printable district workflow support over deeper modeling. That kept the solution explainable in live defense and allowed me to ship all mandatory artifacts.
