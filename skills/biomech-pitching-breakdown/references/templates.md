# Which Template to Use

The skill produces three distinct report types. Use the right one for the job.

## 1. Standard Report — `scripts/build_standard_report.js`

**When to use:**
- Default for every pitcher analyzed
- User asks for "a report," "the analysis," or simply uploads a new pitcher
- Single-pitcher individualized output

**Format:** Portrait, 6–8 pages, full 8-cylinder structure
**Output:** `<LastName>_<FirstName>_Biomechanics_Report.docx`
**See:** `assets/standard_input_example.json` for input shape

## 2. Chain Breakdown Analysis — `scripts/build_chain_analysis.js`

**When to use:**
- The pitcher's amplification story is interesting (unusually high or unusually low pelvis-to-trunk amplification)
- User says something like "he moves slow," "the chain seems off," "where is the velocity going?"
- Want to demonstrate a model arm (Boyatt) or a chain-leak pattern (Stacks)
- Pairs well with the standard report — different angle, focused on one specific dimension

**Format:** Landscape, 4–5 pages
**Output:** `<LastName>_<FirstName>_Chain_Breakdown_Analysis.docx`
**See:** `assets/chain_input_example.json` for input shape (if you want one created, ask)

## 3. Cohort / Group Comparison

**When to use:**
- User wants to compare multiple pitchers
- Roster reviews
- Demonstrate archetypes side-by-side

**Format:** Landscape, multi-pitcher side-by-side tables. Variable structure depending on the goal.

**Build approach:** Use the same docx-js patterns as the other two scripts. There's no dedicated script for this because the format varies — write a tailored builder for the specific comparison.

## How to compose a report (process)

For a single pitcher analysis:

1. **Get the data.** Run `python scripts/extract_metrics.py <input>` to get a clean JSON of extracted metrics. Save to a working directory.

2. **Read the metrics.** Inspect the extracted values. Check the headline numbers — velocity, Cyl 3 product, HSS at FS, elbow torque, pelvis-to-trunk amplification, HzAbd dwell, MER.

3. **Identify the archetype.** Cross-reference with `references/archetypes.md`. Most pitchers fit one or sometimes two archetypes.

4. **Pick the dominant story.** Every pitcher's data has 2–4 things going on. Identify ONE dominant story (e.g., "his trunk doesn't dump," or "the chain breaks at the trunk," or "he's the cohort model arm") and frame the whole report around it. The session summary and bottom line should both lead with this story.

5. **Build the input JSON.** Fill in the structure from `assets/standard_input_example.json`. The interpretive sections — summary, whats_working, where_leaking, recommendations, bottom_line — come from YOU based on the data and the framework. The numeric sections come from the extraction script.

6. **Run the build script.** `node scripts/build_standard_report.js input.json output.docx`

7. **Share the docx with the user.** Use a computer:// link.

## Status decisions

When assigning a status (best/good/warn/bad) to a cylinder or detail metric, use `references/benchmarks.md` as the calibration source. Don't make up your own bands — the cohort context matters and the framework is calibrated.

If a metric is in an "ambiguous" zone (e.g., right on the boundary), prefer to highlight as `warn` and explain in the narrative rather than over-grading.

## When to also build the chain analysis

Build a chain analysis IN ADDITION to the standard report when:

- Pelvis-to-trunk amplification is below 1.2× or above 1.7× (interesting outlier)
- Pitcher ranks very high (#1–2) or very low (last 1–2) in any single segment velocity
- The data shows a "broken chain" pattern (e.g., humerus-to-hand >1.0× = arm rescue)
- The pitcher is a model arm (offer it as a teaching example)

Don't build it as the default — it's a secondary deep-dive. Offer it to the user as an option.