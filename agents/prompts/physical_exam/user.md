Diagnosis (hidden from resident): {{diagnosis}}

Patient:
- Age: {{age}}, Gender: {{gender}}
- Symptoms: {{symptoms}}
- Vitals: HR {{heart_rate}}, BP {{blood_pressure}}, RR {{respiratory_rate}}, SpO2 {{oxygen_saturation}}%
- History: {{history}}
- Risk factors: {{risk_factors}}

Systems to examine: {{systems}}

MANDATORY GROUND-TRUTH FINDINGS — copy these verbatim into your findings for the matching system:
{{ground_truth_block}}

For systems listed above that have no ground-truth findings, generate findings consistent with the diagnosis, vitals, and symptoms.
For systems that do have ground-truth findings, include them exactly as stated and you may add clinically consistent supporting detail.
