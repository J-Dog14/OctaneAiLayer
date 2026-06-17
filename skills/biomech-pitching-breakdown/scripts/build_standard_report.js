/*
 * Build a standard 8-cylinder biomechanics report (.docx, portrait).
 *
 * Usage:
 *   node build_standard_report.js <input.json> <output.docx>
 *
 * <input.json> is a JSON file containing the EXTRACTED METRICS (from extract_metrics.py)
 * PLUS the INTERPRETIVE SECTIONS (summary, strengths, leaks, recommendations,
 * bottom_line). The Python script populates the metrics; the calling Claude
 * populates the interpretive content based on cylinders.md + archetypes.md +
 * benchmarks.md.
 *
 * Input structure (see assets/standard_input_example.json):
 *   {
 *     "subject": { name, handed, age_str, height_str, weight_str, test_date, ... },
 *     "velocity": { mean, sd, throws: [{id, velocity}, ...] },
 *     "cylinders": [
 *       { num: "1", name: "Stride", value_text: "80.8 % of height",
 *         benchmark_text: "Elite 85-90 %", status: "good" }, ...
 *     ],
 *     "detail_metrics": [
 *       { label, value_text, ref_text, status }, ...
 *     ],
 *     "summary": "Free-text paragraph(s)...",
 *     "whats_working": [{ heading, body }, ...],
 *     "where_leaking": [{ heading, body }, ...],
 *     "recommendations": [{ heading, body }, ...],
 *     "bottom_line": "Closing paragraph...",
 *     "title": "Athlete Name — Right-Handed Pitcher"
 *   }
 *
 * The script requires the `docx` npm package — install with `npm install docx` from a directory
 * where you have write access.
 */

const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, WidthType, BorderStyle, ShadingType,
  LevelFormat
} = require('docx');

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 100, bottom: 100, left: 140, right: 140 };

const TABLE_CELL_SIZE = 24;   // 12pt body
const TABLE_HEADER_SIZE = 26; // 13pt headers

const STATUS_COLORS = {
  best: '548235',
  good: '70AD47',
  warn: 'ED7D31',
  bad: 'C00000',
  neutral: '7F7F7F',
};
const STATUS_LABELS = {
  best: 'Elite',
  good: 'Strength',
  warn: 'Watch',
  bad: 'Flag',
  neutral: 'Neutral',
};

const P = (text) => new Paragraph({ spacing: { after: 120 }, children: Array.isArray(text) ? text : [new TextRun(text)] });
const bullet = (text) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 }, children: Array.isArray(text) ? text : [new TextRun(text)] });
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 140 }, children: [new TextRun(text)] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 120 }, children: [new TextRun(text)] });
const h3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 160, after: 80 }, children: [new TextRun(text)] });

function cell(text, opts = {}) {
  const { bold = false, fill = null, width = 1872, align = AlignmentType.LEFT, color = null, size = TABLE_CELL_SIZE } = opts;
  return new TableCell({ borders, margins: cellMargins,
    width: { size: width, type: WidthType.DXA },
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text: String(text), bold, color, size })] })] });
}
function headerCell(text, width) {
  return new TableCell({ borders, margins: cellMargins,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "1F4E78", type: ShadingType.CLEAR },
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, color: "FFFFFF", size: TABLE_HEADER_SIZE })] })] });
}
function statusCell(status, width) {
  const fill = STATUS_COLORS[status] || STATUS_COLORS.neutral;
  const label = STATUS_LABELS[status] || 'Neutral';
  return new TableCell({ borders, margins: cellMargins,
    width: { size: width, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: label, bold: true, color: 'FFFFFF', size: TABLE_CELL_SIZE })] })] });
}

function buildSubjectTable(s) {
  return [
    new TableRow({ children: [
      cell("Athlete", { bold: true, fill: "E7E6E6", width: 2200 }), cell(s.name, { width: 3000 }),
      cell("Test Date", { bold: true, fill: "E7E6E6", width: 1700 }), cell(s.test_date || '', { width: 2460 }),
    ]}),
    new TableRow({ children: [
      cell("DOB / Age", { bold: true, fill: "E7E6E6", width: 2200 }), cell(s.age_str || '', { width: 3000 }),
      cell("Handedness", { bold: true, fill: "E7E6E6", width: 1700 }), cell(s.handed === 'LHP' ? 'Left' : 'Right', { width: 2460 }),
    ]}),
    new TableRow({ children: [
      cell("Height / Weight", { bold: true, fill: "E7E6E6", width: 2200 }), cell(s.size_str || '', { width: 3000 }),
      cell("Pitches", { bold: true, fill: "E7E6E6", width: 1700 }), cell(s.pitches_str || '5 fastballs', { width: 2460 }),
    ]}),
    new TableRow({ children: [
      cell("Lab / Operator", { bold: true, fill: "E7E6E6", width: 2200 }), cell(`${s.lab || 'B_Young Biomechanics'} — ${s.operator || 'Brandon Young'}`, { width: 3000 }),
      cell("System", { bold: true, fill: "E7E6E6", width: 1700 }), cell("Qualisys / Visual3D 4.6", { width: 2460 }),
    ]}),
  ];
}

function buildVelocityTable(velos) {
  const headerRow = [cell("Pitch", { bold: true, fill: "DDEBF7", width: 2340 })];
  const dataRow = [cell("Velocity (mph)", { bold: true, width: 2340 })];
  const colW = Math.floor(7020 / velos.length);
  velos.forEach(t => {
    headerRow.push(cell(t.id.replace(/Fastball\s+\w+\s+/, 'FB #'), { bold: true, fill: "DDEBF7", width: colW, align: AlignmentType.CENTER }));
    dataRow.push(cell(t.velocity != null ? t.velocity.toFixed(1) : '—', { width: colW, align: AlignmentType.CENTER }));
  });
  return [new TableRow({ children: headerRow }), new TableRow({ children: dataRow })];
}

function buildCylindersTable(cylinders) {
  const header = new TableRow({ children: [
    headerCell("Cylinder", 2400),
    headerCell("Value", 2900),
    headerCell("Reference / Benchmark", 2400),
    headerCell("Status", 1660),
  ]});
  const rows = [header];
  cylinders.forEach(c => {
    rows.push(new TableRow({ children: [
      cell(`${c.num}. ${c.name}`, { bold: true, width: 2400 }),
      cell(c.value_text, { width: 2900, align: AlignmentType.CENTER, bold: true }),
      cell(c.benchmark_text, { width: 2400, align: AlignmentType.CENTER }),
      statusCell(c.status, 1660),
    ]}));
  });
  return rows;
}

function buildDetailTable(detailMetrics) {
  const header = new TableRow({ children: [
    headerCell("Detailed Metric", 3300),
    headerCell("Pitcher (mean)", 1900),
    headerCell("Reference", 2700),
    headerCell("Status", 1460),
  ]});
  const rows = [header];
  detailMetrics.forEach(m => {
    rows.push(new TableRow({ children: [
      cell(m.label, { bold: true, width: 3300 }),
      cell(m.value_text, { width: 1900, align: AlignmentType.CENTER, bold: true }),
      cell(m.ref_text, { width: 2700, align: AlignmentType.CENTER }),
      statusCell(m.status, 1460),
    ]}));
  });
  return rows;
}

function buildDoc(input) {
  const s = input.subject;
  const children = [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
      children: [new TextRun({ text: "Pitching Biomechanics Report", bold: true, size: 40 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
      children: [new TextRun({ text: input.title || `${s.name} — ${s.handed === 'LHP' ? 'Left' : 'Right'}-Handed Pitcher`, size: 26, color: "555555" })] }),

    new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [2200, 3000, 1700, 2460], rows: buildSubjectTable(s) }),
    P(" "),

    h1("Session Summary"),
    P(input.summary || ''),
  ];

  if (input.velocity && input.velocity.throws && input.velocity.throws.length) {
    children.push(h2("Velocity Across Throws"));
    if (input.velocity.mean != null) {
      children.push(P(`Mean ${input.velocity.mean.toFixed(1)} mph, SD ${input.velocity.sd != null ? input.velocity.sd.toFixed(2) : '—'} (${input.velocity.throws.length} throws).`));
    }
    const colW = Math.floor(7020 / input.velocity.throws.length);
    children.push(new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [2340, ...input.velocity.throws.map(() => colW)],
      rows: buildVelocityTable(input.velocity.throws),
    }));
    children.push(P(" "));
  }

  if (input.cylinders && input.cylinders.length) {
    children.push(h1("The 8 Cylinders"));
    children.push(P("Color-coded against benchmark ranges and against the cohort."));
    children.push(new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [2400, 2900, 2400, 1660],
      rows: buildCylindersTable(input.cylinders),
    }));
    children.push(P(" "));
  }

  if (input.detail_metrics && input.detail_metrics.length) {
    children.push(h2("Underlying Metrics"));
    children.push(new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [3300, 1900, 2700, 1460],
      rows: buildDetailTable(input.detail_metrics),
    }));
    children.push(P(" "));
  }

  if (input.whats_working && input.whats_working.length) {
    children.push(h1("What's Working"));
    input.whats_working.forEach(s => {
      children.push(h3(s.heading));
      children.push(P(s.body));
    });
  }

  if (input.where_leaking && input.where_leaking.length) {
    children.push(h1("Where Velocity Is Leaking"));
    input.where_leaking.forEach(s => {
      children.push(h3(s.heading));
      children.push(P(s.body));
    });
  }

  if (input.recommendations && input.recommendations.length) {
    children.push(h1("Recommendations"));
    input.recommendations.forEach(r => {
      if (r.heading) {
        children.push(bullet([new TextRun({ text: r.heading + ' — ', bold: true }), new TextRun(r.body || '')]));
      } else {
        children.push(bullet(r.body || ''));
      }
    });
  }

  if (input.bottom_line) {
    children.push(h1("Bottom Line"));
    children.push(P(input.bottom_line));
  }

  children.push(P(" "));
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "— End of report —", italics: true, color: "888888" })] }));

  return new Document({
    creator: "8ctane Baseball",
    title: `${s.name} — Biomechanics Report`,
    styles: {
      default: { document: { run: { font: "Calibri", size: 22 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 30, bold: true, color: "1F4E78", font: "Calibri" },
          paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 26, bold: true, color: "2E75B6", font: "Calibri" },
          paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, color: "333333", font: "Calibri" },
          paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
      ]
    },
    numbering: {
      config: [
        { reference: "bullets",
          levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }
      ]
    },
    sections: [{
      properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } } },
      children
    }]
  });
}

async function main() {
  const [, , inputPath, outputPath] = process.argv;
  if (!inputPath || !outputPath) {
    console.error("Usage: node build_standard_report.js <input.json> <output.docx>");
    process.exit(1);
  }
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const doc = buildDoc(input);
  const buf = await Packer.toBuffer(doc);
  fs.writeFileSync(outputPath, buf);
  console.log(`Wrote ${outputPath} (${buf.length} bytes)`);
}

main().catch(e => { console.error(e); process.exit(1); });