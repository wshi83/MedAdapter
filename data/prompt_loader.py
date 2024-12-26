GSM8K_PROMPT = """Q: Ivan has a bird feeder in his yard that holds two cups of birdseed. Every week, he has to refill the emptied feeder. Each cup of birdseed can feed fourteen birds, but Ivan is constantly chasing away a hungry squirrel that steals half a cup of birdseed from the feeder every week. How many birds does Ivan’s bird feeder feed weekly?
A: Let's think step by step.
The squirrel steals 1/2 cup of birdseed every week, so the birds eat 2 - 1/2 = 1 1/2 cups of birdseed.
Each cup feeds 14 birds, so Ivan's bird feeder feeds 14 * 1 1/2 = 21 birds weekly.
#### The answer is 21

Q: Samuel took 30 minutes to finish his homework while Sarah took 1.3 hours to finish it. How many minutes faster did Samuel finish his homework than Sarah?
A: Let's think step by step.
Since there are 60 minutes in 1 hour, then 1.3 hours is equal to 1.3 x 60 = 78 minutes.
Thus, Samuel is 78 - 30 = 48 minutes faster than Sarah.
#### The answer is 48

Q: Julia bought 3 packs of red balls, 10 packs of yellow balls, and 8 packs of green balls. There were 19 balls in each package. How many balls did Julie buy in all?
A: Let's think step by step.
The total number of packages is 3 + 10 + 8 = 21.
Julia bought 21 × 19 = 399 balls.
#### The answer is 399

Q: Lexi wants to run a total of three and one-fourth miles. One lap on a particular outdoor track measures a quarter of a mile around. How many complete laps must she run?
A: Let's think step by step.
There are 3/ 1/4 = 12 one-fourth miles in 3 miles.
So, Lexi will have to run 12 (from 3 miles) + 1 (from 1/4 mile) = 13 complete laps.
#### The answer is 13
"""

PUBMEDQA_PROMPT = '''
Use the step-by-step method as shown in the example to answer the question. You should give the reasoning steps and final answer based on the provided context.

Example:
Q: Do familiar teammates request and accept more backup?
A: Transactive memory theory extends to high-stress environments in which members' expertise is highly overlapping.
Teammates' shared mental models about one another increase the likelihood that they will request and accept backup.
#### Yes.

Here is your question. Please respond to this question based on the context and by adhering to the given format: provide step-by-step reasoning (one sentence per line), then give the final answer (Yes/No/Maybe) after '####'.
'''.strip()

MEDMCQA_PROMPT = '''
Use the step-by-step method as shown in the example to answer the question. You should give the explanation steps and final answer based on the provided context.

Example:
Q: What is the most probable poal of entry of Aspergillus? (A) Puncture wound, (B) Blood, (C) Lungs, (D) Gastrointestinal tract 
A: Aspergillus species are widely distributed on decaying plants, producing chains of conidia. 
Aspergillus species unlike Candida species do not form the pa of normal flora of humans. 
They are ubiquitous in the environment; hence transmission of infection is mostly exogenous. 
Aspergillus transmission occurs by inhalation of airborne conidia. 
Risk Factors for invasive aspergillosis are: Glucocoicoid use (the most impoant risk factor) Profound neutropenia or Neutrophil dysfunction Underlying pneumonia or COPD, tuberculosis or sarcoidosis Antitumor necrosis factor therapy.
#### C.

Here is your question. Please respond to this question based on the context and by adhering to the given format: provide step-by-step reasoning (one sentence per line), then give the final answer (A/B/C/D) after '####'.
'''.strip()

MMLU_PROMPT = '''
Use the step-by-step method as shown in the example to answer the question. You should give the reasoning steps and final answer based on the provided context.

Example:
Q: What size of cannula would you use in a patient who needed a rapid blood transfusion (as of 2020 medical knowledge)? (A) 18 gauge, (B) 20 gauge, (C) 22 gauge, (D) 24 gauge. 
A: The gauge of a cannula indicates its diameter: the smaller the number, the larger the diameter of the cannula.
A larger diameter cannula allows for the rapid administration of fluids, including blood.
In emergency situations requiring rapid transfusion, a larger cannula is preferred to ensure quick delivery of blood to the patient.
An 18 gauge cannula is larger than the 20, 22, and 24 gauge options and is commonly used for rapid transfusions.
#### A.

Here is your question. Please respond to this question based on the context and by adhering to the given format: provide step-by-step reasoning (one sentence per line), then give the final answer (A/B/C/D) after '####'.
'''.strip()

MEDQA_PROMPT = '''
Use the step-by-step method as shown in the example to answer the question. You should give the reasoning steps and final answer based on the provided context.

Example:
Q: A 21-year-old sexually active male complains of fever, pain during urination, and inflammation and pain in the right knee. A culture of the joint fluid shows a bacteria that does not ferment maltose and has no polysaccharide capsule. The physician orders antibiotic therapy for the patient. The mechanism of action of action of the medication given blocks cell wall synthesis, which of the following was given? (A) Gentamicin, (B) Ciprofloxacin, (C) Ceftriaxone, (D) Trimethoprim.
A: The symptoms and culture results suggest a bacterial infection that affects both the urinary tract and joints, indicating a systemic infection.
Bacteria that do not ferment maltose and lack a polysaccharide capsule could indicate a variety of bacteria, but the treatment approach focuses on the mechanism of action of the antibiotic rather than the specific bacteria.
Antibiotics that block cell wall synthesis are typically beta-lactams, which include penicillins and cephalosporins.
Gentamicin is an aminoglycoside antibiotic, which works by inhibiting protein synthesis.
Ciprofloxacin is a fluoroquinolone, which works by inhibiting bacterial DNA gyrase and topoisomerase IV, affecting DNA replication.
Ceftriaxone is a third-generation cephalosporin, which works by inhibiting cell wall synthesis.
Trimethoprim is an antibiotic that inhibits bacterial dihydrofolate reductase, affecting folic acid synthesis.
#### C.

Here is your question. Please respond to this question based on the context and by adhering to the given format: provide step-by-step reasoning (one sentence per line), then give the final answer (A/B/C/D) after '####'.
'''.strip()

BIOASQ_PROMPT = '''
Use the step-by-step method as shown in the example to answer the question. You should give the reasoning steps and final answer based on the provided context.

Example:
Q: Can losartan reduce brain atrophy in Alzheimer's disease?
A: Losartan is primarily used for hypertension and may indirectly affect factors associated with Alzheimer's disease progression. 
Despite potential neuroprotective effects, such as reducing inflammation and oxidative stress, there is limited direct evidence linking losartan to reduced brain atrophy in Alzheimer's disease. 
Clinical trials specifically targeting this outcome are necessary to establish a definitive effect.
#### no

Here is your question. Please respond to this question based on the context and by adhering to the given format: provide step-by-step reasoning (one sentence per line), then give the final answer (yes/no) after '####'.
'''.strip()

MEDNLI_PROMPT = '''
What is the relationship between the given two sentences? Please answer from [entailment, neutral, contradiction]. Please give the answer after '####'.

Example:
Sentence A: Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.
Sentence B: Patient has elevated Cr
Answer: #### entailment

Here are the given two sentences. Please then give the final answer (entailment/neutral/contradiction) after '####'.
'''

MEDIQA_RQE_PROMPT = '''
Does the provided solution correctly answer thq question? Please answer from [true, false].

Example:
Question: What is High Blood Pressure?
Solution: High Blood Pressure. I know you may not answer this but my blood pressure comes up at night when I am asleep. I take four medicines. I have asked doctors why this happens and no one knows. This morning at four A.M. It was 164 and I took a clonidine to help get it done. It worries me so.
Judge: #### false

Here is the question and answer. Please then give the final judge (true/false) after '####'.
'''

PUBHEALTH_PROMPT = '''
Use the step-by-step method as shown in the example to answer the question. You should give the thought steps and final answer based on the provided context. Please judge whether the claim is true or false. 

Example:
Claim: Annual Mammograms May Have More False-Positives	October 18, 2011
Judge: This article reports on the results of a study of nearly 170,000 women who had screening mammograms beginning between age 40-59. The study found that over ten years of screening mammograms, over half of the women will experience a false-positive recall for additional mammography. In addition, 7%-9% of the women will have a biopsy for a suspicious lump which is not cancerous. Both of those percentages decrease if the woman is screened every other year rather than every year. Even with biennial mammography, 41% of women will experience a recall over 10 years of mammography. The study’s Principal Investigator emphasized that “in most cases, a recall doesn’t mean you have cancer.”  She hoped this knowledge would reduce the anxiety of women who are recalled. The story never explained the size of the decrease in the number of false positives between annual (61.3%) and biennial screening (41.6%). Our first two reviewers were a researcher who specializes in health decisions and a breast cancer survivor trained in evidence by the Natiional Breast Cancer Coalition’s Project LEAD. This study is valuable because it helps to quantify and compare the harms of annual and biennial screening, specifically the number of false positives and the number of unnecessary biopsies. Prior to this study, estimates of false positive screening mammography rates varied widely. The critical question is whether you can do less frequent screening, subject women to fewer harms and get similar results in terms of detection of “early stage” cancer. This study’s data seems to suggest that answer is yes.
#### mixture

Here is the claim. Please then give the final judge (true/false/mixture/unproven) after '####'.
'''
CORD19_PROMPT = '''Use one sentence to summarize the given paragraph. 

Example: 
Paragraph: Cardiovascular disease is the leading cause of death globally. While pharmacological advancements have improved the morbidity and mortality associated with cardiovascular disease, non-adherence to prescribed treatment remains a significant barrier to improved patient outcomes. A variety of strategies to improve medication adherence have been tested in clinical trials, and include the following categories: improving patient education, implementing medication reminders, testing cognitive behavioral interventions, reducing medication costs, utilizing healthcare team members, and streamlining medication dosing regimens. In this review, we describe specific trials within each of these categories and highlight the impact of each on medication adherence. We also examine ongoing trials and future lines of inquiry for improving medication adherence in patients with cardiovascular diseases.
Summary: Medication adherence in cardiovascular medicine.

Here is the paragraph. Please then give the final one-sentence summary after 'Summary:'.
'''

def load_prompts(dataset_name):
    if 'gsm8k' in dataset_name.lower():
        return None #GSM8K_PROMPT
    elif 'pubmedqa' in dataset_name.lower():
        return PUBMEDQA_PROMPT
    elif 'medmcqa' in dataset_name.lower():
        return MEDMCQA_PROMPT
    elif 'mmlu' in dataset_name.lower():
        return MMLU_PROMPT
    elif 'medqa' in dataset_name.lower():
        return MEDQA_PROMPT
    elif 'bioasq' in dataset_name.lower():
        return BIOASQ_PROMPT
    elif 'mednli' in dataset_name.lower():
        return MEDNLI_PROMPT
    elif 'mediqa-rqe' in dataset_name.lower():
        return MEDIQA_RQE_PROMPT
    elif 'pubhealth' in dataset_name.lower():
        return PUBHEALTH_PROMPT
    else:
        return None
    