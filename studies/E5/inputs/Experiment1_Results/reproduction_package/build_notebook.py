from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(text).strip().splitlines(keepends=True)})


def code(text):
    cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
                  'source':textwrap.dedent(text).strip().splitlines(keepends=True)})


md('''
# ASOC Experiment 1: joint fitting versus anchor-only delay learning

Use a **CPU runtime** and run cells in order. This notebook executes the fixed prospective protocol supplied in the package. It does not use the old exploratory pilot as new confirmation. The full run has five new observation-noise datasets, three paired optimization seeds, four neural methods (60 fits), and two deterministic baselines per dataset.

Upload `ASOC_Experiment1_Run_Package.zip` in the first cell. At the end, download and return `ASOC_Experiment1_Results.zip`. Keep the session connected during training. If interrupted, rerun using the same retained output directory; the manifest guards the resume. No full results are embedded in this notebook.
''')
md('''
## 1. Upload the run package

Upload the complete ZIP, not the Overleaf projects. Uploaded files are used only in this Colab session.
''')
code('''
from google.colab import files
from pathlib import Path
import zipfile
import sys
import subprocess
import json

uploaded = files.upload()
matches = [name for name in uploaded if name.startswith('ASOC_Experiment1_Run_Package') and name.lower().endswith('.zip')]
if len(matches) != 1:
    raise ValueError('Upload the file named ASOC_Experiment1_Run_Package.zip.')
PACKAGE = Path('/content/asoc_experiment1')
PACKAGE.mkdir(exist_ok=True)
with zipfile.ZipFile(matches[0]) as archive:
    for item in archive.infolist():
        if not (PACKAGE / item.filename).resolve().is_relative_to(PACKAGE.resolve()):
            raise ValueError('Unexpected path in archive')
    archive.extractall(PACKAGE)
assert (PACKAGE / 'run_experiment1.py').is_file()
print((PACKAGE / 'ASOC_Experiment1_Protocol.md').read_text())
''')
md('''
## 2. Install CPU training dependencies and select the pinned repository

The original confirmation record identifies PyTorch 2.10.0 and NumPy 2.4.2. Other package versions are recorded by the runner. All fits use one CPU thread. No archived neural checkpoint is loaded or used as a starting model.
''')
code('''
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'torch==2.10.0', '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'numpy==2.4.2', 'pandas>=2.2,<4', 'scipy>=1.11,<2', 'matplotlib>=3.9,<4'], check=True)
config = json.loads((PACKAGE / 'protocol.json').read_text())
REPO = Path('/content/NSDDNew_ASOC_E1')
if not REPO.exists():
    subprocess.run(['git', 'clone', '--quiet', config['repository'], str(REPO)], check=True)
subprocess.run(['git', '-C', str(REPO), 'checkout', '--quiet', config['commit']], check=True)
print('Repository commit:', subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip())
''')
md('''
## 3. Verify archived anchors and package accounting

These checks use archived seeds and explicitly labelled synthetic fixtures. They produce no new experiment outcomes.
''')
code('''
subprocess.run([sys.executable, '-u', str(PACKAGE / 'validate_package.py'),
                '--repo', str(REPO)], cwd=PACKAGE, check=True)
subprocess.run([sys.executable, '-u', str(PACKAGE / 'check_training_parity.py'),
                '--repo', str(REPO), '--output', str(PACKAGE / 'colab_training_parity.json')],
               cwd=PACKAGE, check=True)
''')
md('''
## 4. Run the neural smoke check

This uses old seeds and a tiny training budget. It must pass before the full experiment. Smoke results are separate and cannot be reported as scientific evidence.
''')
code('''
SMOKE = PACKAGE / 'smoke_results'
subprocess.run([sys.executable, '-u', str(PACKAGE / 'run_experiment1.py'),
                '--repo', str(REPO), '--mode', 'smoke', '--output', str(SMOKE), '--resume'],
               cwd=PACKAGE, check=True)
smoke_summary = json.loads((SMOKE / 'summary.json').read_text())
assert smoke_summary['complete'] and smoke_summary['recorded_neural_fits'] == 4
assert smoke_summary['recorded_direct_functions'] == 2
print('Smoke check passed. These values are excluded from the full experiment.')
''')
md('''
## 5. Run the complete fixed experiment

This is the main computation: P3, P4, A1 and A2 on five fresh noise datasets with three paired initializations. All hyperparameters and budgets are fixed in `protocol.json`. Progress is printed after every fit. Resume uses identical settings and skips completed records, including recorded failures; it does not silently replace failed seeds.
''')
code('''
assert json.loads((SMOKE / 'summary.json').read_text())['complete'], 'Run the smoke check first.'
RESULTS = PACKAGE / 'experiment1_results'
subprocess.run([sys.executable, '-u', str(PACKAGE / 'run_experiment1.py'),
                '--repo', str(REPO), '--mode', 'full', '--output', str(RESULTS), '--resume'],
               cwd=PACKAGE, check=True)
summary = json.loads((RESULTS / 'summary.json').read_text())
assert summary['complete']
print(json.dumps(summary, indent=2))
subprocess.run([sys.executable, '-u', str(PACKAGE / 'audit_completed_run.py'),
                '--repo', str(REPO), '--results', str(RESULTS)], cwd=PACKAGE, check=True)
''')
md('''
## 6. Download the complete audit bundle

Run this cell even after a training error to preserve available records and failure diagnostics. The ZIP includes checkpoints, observations, curves, all metrics, manifests, the protocol and code. Return it for auditing before any results are added to the article.
''')
code('''
import shutil
import tempfile

RESULTS = PACKAGE / 'experiment1_results'
assert RESULTS.exists(), 'There is no full-experiment output to download yet.'
with tempfile.TemporaryDirectory(prefix='asoc_e1_return_') as staging:
    staging = Path(staging)
    for name in ['experiment1_results', 'smoke_results']:
        if (PACKAGE / name).exists():
            shutil.copytree(PACKAGE / name, staging / name)
    for name in ['ASOC_Experiment1_Protocol.md', 'protocol.json', 'run_experiment1.py',
                 'validate_package.py', 'README.md', 'ASOC_Experiment1_Colab.ipynb',
                 'preparation_validation.log', 'check_training_parity.py',
                 'neural_smoke_validation.json', 'neural_training_parity.json',
                 'colab_training_parity.json', 'audit_completed_run.py',
                 'NSDDNew_LICENSE.txt']:
        if (PACKAGE / name).exists():
            shutil.copy2(PACKAGE / name, staging / name)
    archive = shutil.make_archive('/content/ASOC_Experiment1_Results', 'zip', staging)
files.download(archive)
''')

notebook = {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                    'language_info':{'name':'python'},
                                    'colab':{'name':'ASOC_Experiment1_Colab.ipynb','provenance':[]}},
            'nbformat':4,'nbformat_minor':5}
for index, cell in enumerate(cells):
    cell['id'] = f'asoc-e1-{index:02d}'
(ROOT/'ASOC_Experiment1_Colab.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
print(f'Wrote notebook with {len(cells)} cells, no execution outputs.')
