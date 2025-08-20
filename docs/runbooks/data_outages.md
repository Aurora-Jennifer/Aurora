# Data Outages

Purpose: Steps to handle vendor/API outages.

Entrypoints: , 

Do-not-touch:  logic during incident.

1) Switch quotes provider to  via flag; stop live.
2) Touch  to halt.
3) Validate last good snapshot; log anomaly.
