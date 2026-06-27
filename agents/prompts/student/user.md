Difficulty: {{difficulty}}

Step number: {{step_number}}

Visible context:
{{visible_context_json}}

Decision guidance:
- Check simulation_control.fast_mode, steps_remaining, and near_end.
- If near_end is true and no diagnosis_guess exists, choose submit_diagnosis unless an uninterpreted important test result should be interpreted first.
- If steps_remaining is 1, submit_diagnosis with your best visible-evidence guess.
- If ECG, troponin, glucose, CT, X-ray, ABG, CBC, electrolytes, or other relevant results are visible and student_interpretations is empty, submit_interpretation if there is enough time before diagnosis.
- Do not ask low-yield repeat questions when near_end is true.

Return this JSON structure:
{
  "action": "ask_patient_question | order_investigation | submit_interpretation | submit_diagnosis | call_doctor | finish",
  "patient_question": "string or null",
  "investigation": "string or null",
  "interpretation": "string or null",
  "diagnosis_guess": "string or null",
  "consult_question": "string or null"
}

Return JSON only.
