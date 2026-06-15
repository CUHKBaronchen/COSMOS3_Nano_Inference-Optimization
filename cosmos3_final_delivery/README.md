# Cosmos3 T2V Final Delivery

## Primary deliverables

- `COSMOS3_T2V_FINAL_REPORT.pdf`: 11-page formal technical report with embedded Chinese fonts and same-noise visual comparisons.
- `COSMOS3_T2V_PRESENTATION.pptx`: 16:9, 16-slide defense deck. All visible text is rendered into 1920x1080 slide images to prevent missing-font/garbled-text issues.
- `COSMOS3_T2V_FINAL_REPORT.md`: searchable source summary.
- `EVIDENCE_INDEX.md`: absolute-path evidence index.
- `SHA256SUMS.txt`: integrity checksums.

## Video handling

PDF cannot play MP4 reliably, so it embeds multi-timepoint same-noise contact sheets and lists clickable absolute MP4 paths. The PPT embeds the same visual evidence and slide 11 contains a clickable link to the complete P3-vs-P4 anchor MP4. Full multi-prompt video paths are listed in the PDF and evidence index.

## Rebuild

`/root/autodl-tmp/cosmos3_final_delivery/.venv/bin/python /root/autodl-tmp/cosmos3_final_delivery/build_final_delivery.py`
