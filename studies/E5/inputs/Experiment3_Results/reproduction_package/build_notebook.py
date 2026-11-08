from pathlib import Path
import json
import textwrap

ROOT=Path(__file__).resolve().parent
cells=[]
def md(text):
    cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(text).strip().splitlines(keepends=True)})
def code(text):
    cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                  'source':textwrap.dedent(text).strip().splitlines(keepends=True)})

md('''
# ASOC Experiment 3: observation and training-solver resolution

Run on CPU. This self-contained package crosses observation spacing 0.05/0.025 with Euler training step 0.025/0.0125, under informative histories and noise SD 0.0005. Anchors are re-estimated over a fixed physical window. The particular delay law is synthetic. The fixed study has 180 actual neural fits; smoke outputs are separate and are not scientific evidence. Final outputs include all checkpoints, paired observations, anchors, curves, numerical contrasts, the protocol and a replay audit.
''')
md('''
## 1. Upload the run package

Select `ASOC_Experiment3_Run_Package.zip`, not an Overleaf ZIP. Read the displayed model check and protocol before execution.
''')
code('''
from google.colab import files
from pathlib import Path
import zipfile
import subprocess
import sys
import json

uploaded=files.upload()
matches=[name for name in uploaded if name.startswith('ASOC_Experiment3_Run_Package') and name.lower().endswith('.zip')]
if len(matches)!=1:
    raise ValueError('Select one ASOC_Experiment3_Run_Package ZIP.')
PACKAGE=Path('/content/asoc_experiment3')
PACKAGE.mkdir(exist_ok=True)
with zipfile.ZipFile(matches[0]) as archive:
    for item in archive.infolist():
        if not (PACKAGE/item.filename).resolve().is_relative_to(PACKAGE.resolve()):
            raise ValueError('Unexpected archive path')
    archive.extractall(PACKAGE)
print((PACKAGE/'ASOC_Experiment2_Model_Check.md').read_text())
print((PACKAGE/'ASOC_Experiment3_Protocol.md').read_text())
''')
md('''
## 2. Install CPU dependencies and run validation

The runner records actual versions. The checks use algebraic identities, a separate ODE solver on an initial-segment fixture, a numerical gradient check, and labelled accounting fixtures. They do not tune the experiment against noisy delay errors.
''')
code('''
subprocess.run([sys.executable,'-m','pip','install','-q','torch==2.10.0',
                '--index-url','https://download.pytorch.org/whl/cpu'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(PACKAGE/'requirements.txt')],check=True)
subprocess.run([sys.executable,'-u',str(PACKAGE/'validate_experiment3.py')],cwd=PACKAGE,check=True)
''')
md('''
## 3. Smoke run on reserved seeds

All four resolution cells, one reserved dataset/initialization, and reduced updates. Expected accounting is 12 actual neural fits and four direct functions. Shared-time observations and initial parameters remain paired. Smoke results stay outside the full experiment.
''')
code('''
SMOKE=PACKAGE/'smoke_results'
subprocess.run([sys.executable,'-u',str(PACKAGE/'run_experiment3.py'),'--mode','smoke',
                '--output',str(SMOKE),'--resume'],cwd=PACKAGE,check=True)
smoke=json.loads((SMOKE/'summary.json').read_text())
assert smoke['complete'] and smoke['trained_neural_fits']==12
assert smoke['direct_functions']==4
subprocess.run([sys.executable,'-u',str(PACKAGE/'audit_experiment3.py'),
                '--results',str(SMOKE)],cwd=PACKAGE,check=True)
print('Smoke execution and replay checks passed.')
''')
md('''
## 4. Run the fixed full study

All seeds, losses, physical windows, budgets and contrasts are fixed. Coarse observations are downsampled from the fine noisy data. There are five independent base-noise datasets, not 20 independent resolution datasets. Every frozen full model is also evaluated at three solver steps on a common observation grid. Resume preserves recorded outcomes and does not silently retry failed seeds.
''')
code('''
assert json.loads((SMOKE/'summary.json').read_text())['complete']
RESULTS=PACKAGE/'experiment3_results'
subprocess.run([sys.executable,'-u',str(PACKAGE/'run_experiment3.py'),'--mode','full',
                '--output',str(RESULTS),'--resume'],cwd=PACKAGE,check=True)
summary=json.loads((RESULTS/'summary.json').read_text())
print(json.dumps(summary,indent=2))
assert summary['complete']
subprocess.run([sys.executable,'-u',str(PACKAGE/'audit_experiment3.py'),
                '--results',str(RESULTS)],cwd=PACKAGE,check=True)
''')
md('''
## 5. Download the complete results and audit

Run this cell after the study, including after a failure if records need to be preserved. The bundle includes scientific results separately from smoke results and all source inputs. Preserve all conditions for manuscript integration; do not replace failed or unfavorable outcomes.
''')
code('''
import tempfile
import shutil

RESULTS=PACKAGE/'experiment3_results'
assert RESULTS.exists(),'No full-study outputs exist yet.'
with tempfile.TemporaryDirectory(prefix='asoc_e3_return_') as folder:
    folder=Path(folder)
    for name in ['experiment3_results','smoke_results']:
        if (PACKAGE/name).exists():
            shutil.copytree(PACKAGE/name,folder/name)
    source=folder/'reproduction_package'
    source.mkdir()
    for path in PACKAGE.iterdir():
        if path.is_file() and path.suffix in ('.py','.md','.json','.ipynb','.txt','.log'):
            shutil.copy2(path,source/path.name)
    archive=shutil.make_archive('/content/ASOC_Experiment3_Results','zip',folder)
files.download(archive)
''')

for i,cell in enumerate(cells):
    cell['id']=f'asoc-e3-{i:02d}'
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
 'language_info':{'name':'python'},'colab':{'name':'ASOC_Experiment3_Colab.ipynb','provenance':[]}},
 'nbformat':4,'nbformat_minor':5}
(ROOT/'ASOC_Experiment3_Colab.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
print(f'Created unexecuted notebook with {len(cells)} cells.')
