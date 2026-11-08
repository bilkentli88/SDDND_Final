"""Create one durable, self-contained checkpoint/delivery ZIP; no external repos."""
import argparse,json,zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.results.resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
 cal=json.loads((root/'calibration_records.json').read_text()) if (root/'calibration_records.json').exists() else []
 final=json.loads((root/'final_records.json').read_text()) if (root/'final_records.json').exists() else []
 audit=json.loads((root/'audit_report.json').read_text()) if (root/'audit_report.json').exists() else None
 status=dict(calibration_records=len(cal),final_records=len(final),calibration_failures=sum(r['status']!='ok' for r in cal),final_failures=sum(r['status']!='ok' for r in final),audit=audit)
 report=HERE/'REPORT.md'
 if not report.exists() or audit is None:
  report.write_text('# Experiment4 checkpoint\n\nStatus: '+json.dumps(status,indent=2)+'\n\nThis is a recovery checkpoint, not completed comparative evidence. See PROTOCOL.md for the original interruption and rerun disclosure. No superiority result is asserted.\n')
 temp=a.output.with_suffix('.tmp')
 with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
  z.writestr('checkpoint_status.json',json.dumps(status,indent=2)+'\n')
  z.writestr('REPORT.md',report.read_text())
  z.writestr('README.md',(HERE/'README.md').read_text())
  for file in sorted(HERE.iterdir()):
   if file.is_file() and file.suffix in ['.py','.json','.md','.txt','.log','.ipynb','.png','.pdf']:z.write(file,'source/'+file.name)
  if root.exists():
   for file in sorted(root.rglob('*')):
    if file.is_file() and file.suffix!='.tmp' and '__pycache__' not in file.parts:z.write(file,'results/'+str(file.relative_to(root)))
  smoke=HERE.parent/'e4_smoke'
  if smoke.exists():
   for file in sorted(smoke.rglob('*')):
    if file.is_file() and '__pycache__' not in file.parts:z.write(file,'reserved_smoke/'+str(file.relative_to(smoke)))
 temp.replace(a.output);print(json.dumps(dict(path=str(a.output.resolve()),bytes=a.output.stat().st_size,status=status),indent=2))

if __name__=='__main__':main()
