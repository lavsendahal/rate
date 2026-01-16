from __future__ import annotations

from typing import Dict, List


# Canonical Merlin 30 disease keys (match label CSV columns).
MERLIN_DISEASES: List[str] = [
    "submucosal_edema",
    "renal_hypodensities",
    "aortic_valve_calcification",
    "coronary_calcification",
    "thrombosis",
    "metastatic_disease",
    "pancreatic_atrophy",
    "renal_cyst",
    "osteopenia",
    "surgically_absent_gallbladder",
    "atelectasis",
    "abdominal_aortic_aneurysm",
    "anasarca",
    "hiatal_hernia",
    "lymphadenopathy",
    "prostatomegaly",
    "biliary_ductal_dilation",
    "cardiomegaly",
    "splenomegaly",
    "hepatomegaly",
    "atherosclerosis",
    "ascites",
    "pleural_effusion",
    "hepatic_steatosis",
    "appendicitis",
    "gallstones",
    "hydronephrosis",
    "bowel_obstruction",
    "free_air",
    "fracture",
]


# Disease -> question used in your modality YAML and earlier LLM runs.
DISEASE_TO_QUESTION: Dict[str, str] = {
    "submucosal_edema": "Is there submucosal edema?",
    "renal_hypodensities": "Are there renal hypodensities?",
    "aortic_valve_calcification": "Is there aortic valve calcification?",
    "coronary_calcification": "Is there coronary calcification?",
    "thrombosis": "Is there thrombosis?",
    "metastatic_disease": "Is there metastatic disease?",
    "pancreatic_atrophy": "Is there pancreatic atrophy?",
    "renal_cyst": "Is there a renal cyst?",
    "osteopenia": "Is there osteopenia?",
    "surgically_absent_gallbladder": "Is the gallbladder surgically absent?",
    "atelectasis": "Is there atelectasis?",
    "abdominal_aortic_aneurysm": "Is there abdominal aortic aneurysm?",
    "anasarca": "Is there anasarca?",
    "hiatal_hernia": "Is there hiatal hernia?",
    "lymphadenopathy": "Is there lymphadenopathy?",
    "prostatomegaly": "Is there prostatomegaly?",
    "biliary_ductal_dilation": "Is there biliary ductal dilation?",
    "cardiomegaly": "Is there cardiomegaly?",
    "splenomegaly": "Is there splenomegaly?",
    "hepatomegaly": "Is there hepatomegaly?",
    "atherosclerosis": "Is there atherosclerosis?",
    "ascites": "Is there ascites?",
    "pleural_effusion": "Is there pleural effusion?",
    "hepatic_steatosis": "Is there hepatic steatosis?",
    "appendicitis": "Is there appendicitis?",
    "gallstones": "Are there gallstones?",
    "hydronephrosis": "Is there hydronephrosis?",
    "bowel_obstruction": "Is there bowel obstruction?",
    "free_air": "Is there free air (pneumoperitoneum)?",
    "fracture": "Is there a fracture?",
}

