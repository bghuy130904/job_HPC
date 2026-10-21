// Optional Work Mode review/export adapter. HPC uses report_dipole.py.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const [input, output, previews] = process.argv.slice(2);
if (!input || !output) throw new Error('Usage: export_artifact_xlsx.mjs workbook.json output.xlsx [preview-directory]');
const plan = JSON.parse(await fs.readFile(input, 'utf8'));
const workbook = Workbook.create();
const col = n => { let s = ''; for (; n; n = Math.floor((n-1)/26)) s = String.fromCharCode(65+(n-1)%26)+s; return s; };
for (const spec of plan.sheets) workbook.worksheets.add(spec.name);
for (const spec of plan.sheets) {
  const sheet = workbook.worksheets.getItem(spec.name);
  const end = col(spec.headers.length), last = spec.rows.length+4;
  sheet.showGridLines = false;
  sheet.getRange(`A1:${end}${Math.max(last,5)}`).format.font = {name:'Arial',size:11,color:'#222222'};
  sheet.getRange(`A1:${end}1`).merge();
  sheet.getRange('A1').values = [[`${plan.title} — ${spec.name}`]];
  sheet.getRange('A1').format.font = {name:'Arial',size:14,bold:true};
  sheet.getRange(`A2:${end}2`).merge();
  sheet.getRange('A2').values = [[spec.note]];
  sheet.getRange(`A2:${end}2`).format.wrapText = true;
  sheet.getRange(`A2:${end}2`).format.rowHeight = spec.headers.length <= 3 ? 65 : 42;
  sheet.getRange(`A4:${end}4`).values = [spec.headers];
  sheet.getRange(`A4:${end}4`).format = {font:{name:'Arial',size:11,bold:true},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:38,
    borders:{bottom:{style:'thin',color:'#777777'}}};
  if (spec.rows.length) sheet.getRange(`A5:${end}${last}`).values = spec.rows;
  for (let i=0;i<spec.headers.length;i++) {
    const h=spec.headers[i], c=col(i+1), r=sheet.getRange(`${c}4:${c}${Math.max(last,5)}`);
    r.format.columnWidth=h.endsWith('_Eh')?24:['error','value'].includes(h)?44:['estimator','status','source','parameter','ob_stability'].includes(h)?24:18;
    if (last>=5) {
      const d=sheet.getRange(`${c}5:${c}${last}`);
      d.setNumberFormat(h==='value'?'General':h.endsWith('_Eh')?'0.000000000000':h.endsWith('_D')?'0.000000':h.includes('pct')?'0.000':h.startsWith('n_')||h.includes('cycles')||['charge','spin','atoms','nao','threads','max_memory_MB'].includes(h)?'0':/gradient|fia|field_au|hermiticity|delta_energy/.test(h)?'0.00E+00':'0.0000');
      d.format.verticalAlignment='center';
      if (['error','value','parameter'].includes(h)) d.format.wrapText=true;
    }
  }
  for (const [address,formula] of Object.entries(spec.formulas)) sheet.getRange(address).formulas=[[formula]];
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(spec.headers.length>2?2:1);
  if (spec.rows.length) {
    const table=sheet.tables.add(`A4:${end}${last}`,true,`${spec.name.replace(/_/g,'')}Data`);
    table.style='TableStyleLight1';
  }
  if (spec.name==='Config' && last>=5) sheet.getRange(`A5:B${last}`).format.autofitRows();
}
workbook.recalculate();
const errors=await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:2000});
console.log(errors.ndjson);
console.log((await workbook.inspect({kind:'table',range:'Results!A4:J10',include:'values,formulas',tableMaxRows:7,tableMaxCols:10,maxChars:2500})).ndjson);
await fs.mkdir(path.dirname(output),{recursive:true});
await (await SpreadsheetFile.exportXlsx(workbook)).save(output);
if (previews) {
  await fs.mkdir(previews,{recursive:true});
  for (const spec of plan.sheets) {
    const sheet=workbook.worksheets.getItem(spec.name);
    const r=`A1:${col(Math.min(spec.headers.length,10))}${Math.min(12,spec.rows.length+4)}`;
    const blob=await workbook.render({sheetName:sheet.name,range:r,scale:1,format:'png'});
    await fs.writeFile(path.join(previews,`${sheet.name}.png`),new Uint8Array(await blob.arrayBuffer()));
  }
}
