Clinical reasoning for this investigation:
{{reasoning}}

Requested investigation: {{requested_investigation}}

Case context (for calibrating values):
{{case_json}}

Return this JSON structure:
{
  "investigation": "string",
  "status": "available",
  "structured_result": {
    "test": "string",
    "findings": ["string"],
    "values": [
      {
        "name": "string",
        "value": "string or number",
        "unit": "string",
        "reference_range": "string",
        "flag": "normal, high, low, mildly high, mildly low, or non-contributory"
      }
    ],
    "interpretation_hint": "string"
  },
  "summary": "string",
  "visual_payload": null
}

{{previous_issues_block}}Return JSON only. Do not include the diagnosis name in any output field.
