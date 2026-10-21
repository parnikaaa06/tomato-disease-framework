"""Provider-neutral instructions and limits for structured explanations."""

EXPLANATION_PROMPT = """\
Produce a cautious, evidence-grounded explanation for a tomato leaf image.
The classifier prediction and uncalibrated softmax score are fixed inputs;
never change, correct, or reinterpret them as calibrated probabilities.
Distinguish statements directly visible in the input image from the
classifier's predicted class. A Grad-CAM heatmap is a coarse model-attribution
visualization, not a lesion mask and not proof of disease-specific attention.
Do not invent symptoms, disease severity, progression, or treatment claims.
Describe only visible attributes supported by the input image. Use
"uncertain" when an interpretation is not directly supported. Do not provide
recommendations; no verified recommendation source is available.

Return one JSON object with exactly these fields:
- observations: a non-empty array of objects with text, evidence_sources,
  certainty.
- summary: a non-empty string.
- claims: a non-empty array of objects with text, claim_type, evidence_sources,
  certainty.

Each evidence_sources value must be "input_image" or "gradcam_heatmap".
Observation certainty must be "direct_visual_observation" or "uncertain".
Claim claim_type must be "visual_observation" or "interpretation"; claim
certainty must be "supported" or "uncertain". Return JSON only. Do not include
classifier fields, limitations, recommendations, markdown fences, or other
keys.
"""
