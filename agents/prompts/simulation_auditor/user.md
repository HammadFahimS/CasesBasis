Full internal simulation context:
{{audit_context_json}}

Return this JSON structure:
{
  "overall_quality": "pass | warning | fail",
  "findings": [
    {
      "category": "confirmed_bug | possible_model_quality_issue | expected_behavior",
      "severity": "low | medium | high",
      "issue": "string",
      "evidence": "string",
      "suggested_fix": "string"
    }
  ],
  "should_fix": true or false
}

Return JSON only.
