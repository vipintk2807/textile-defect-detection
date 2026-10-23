# Manual UI checklist

The Streamlit app could not be launched in the build environment, so these checks are **for you to run** (record results in a copy of this file). Prerequisite: an exported model in `models\` (the synthetic demo model is fine for UI checks).

Start: `python -m streamlit run app.py` and open the printed address.

| # | State / action | Expected | Result |
|---|---|---|---|
| 1 | Open with **no model** in `models\` | "Model unavailable" (✖ + text), error explains how to get a model, uploader disabled, no prediction possible | |
| 2 | Open with valid model | "Model ready: <id>" (✔ + text); synthetic-data warning if the model is synthetic | |
| 3 | No image selected | "No image selected" hint, Analyze not shown | |
| 4 | Upload valid JPG/PNG/WebP | Preview with correct aspect ratio and size caption, "Image ready for analysis" | |
| 5 | Upload a `.txt` renamed to `.png`, and a truncated image | "Invalid image" with a clear message, no preview | |
| 6 | Upload a file larger than the limit | Rejected with the MB limit stated | |
| 7 | Press **Analyze image** | Spinner "Processing image…", then predicted class, ranked softmax scores with the "not a calibrated probability" label, notice, model id and time | |
| 8 | Result with a high threshold (set `confidence_threshold: 0.99`, reload) | "below confidence threshold, review required" (⚠ + text), not "defect-free" | |
| 9 | Replace the image | Old result disappears; new image has no result until analysed | |
| 10 | **Clear image** | Image and result cleared | |
| 11 | Analyze three different images in a row | Each result belongs to the current image | |
| 12 | Tick **Save results to local history**, analyze, open the history expander | "Saved to local history"; row appears; CSV/JSON download works and matches | |
| 13 | Break the DB (make `outputs\inspections.db` a folder) | "History unavailable"/"Not saved" message; prediction still shown | |
| 14 | Very long class name (edit metadata in a copy) | Wraps, nothing clipped | |
| 15 | Narrow the browser window | Columns stack, no horizontal scrolling | |
| 16 | Keyboard: Tab through uploader, Analyze, Clear | Visible focus, all operable | |
| 17 | Browser dev-tools Network tab during analysis | No requests to external hosts | |
