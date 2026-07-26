# SUA-Net model card

## Intended use

Retrospective research on binary benign–malignant breast-ultrasound lesion
classification across BUSI, BUSBRA, and BUS-UCLM.

## Inputs and outputs

Input: one grayscale B-mode ultrasound converted to three channels and resized
to 224×224. Output: benign/malignant logits and probabilities; optional
MC-Dropout predictive dispersion and qualitative Grad-CAM/SVA views.

## Validation scope

- BUSI: image-wise stratified five-fold validation; patient independence is
  not established and exact duplicates are reported.
- BUSBRA: patient-wise five-fold validation using the metadata `Case` field.
- BUS-UCLM: patient-wise five-fold validation using the filename patient code.

Cross-dataset performance degrades under representation and acquisition shift.
MC-Dropout is optional and does not establish improved discrimination or
calibrated clinical confidence.

## Out-of-scope use

- autonomous diagnosis or screening;
- replacement of radiologist assessment;
- deployment without prospective multicenter patient-level validation;
- use on unsupported classes, normal/no-lesion images, or unvalidated devices;
- treating Grad-CAM/SVA maps as segmentations;
- treating the model-only T4 timing as end-to-end clinical latency.


