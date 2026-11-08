"""Make a self-contained Colab launcher without bundling third-party source."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
names=['protocol.json','PROTOCOL.md','e4_data.py','e4_neural.py','e4_analysis.py','run.py','audit.py','validate.py','README.md']
files={n:(HERE/n).read_text() for n in names}
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True))
def code(s):return dict(cell_type='code',metadata={},source=s.splitlines(keepends=True),execution_count=None,outputs=[])
cells=[md('# ASOC E4: independent constant-delay rerun\n\nCPU only. This notebook writes the frozen sources, obtains pinned external releases, runs reserved checks, then calibrates and evaluates new fits with the recorded design. The package report discloses the earlier interrupted attempt. It is an adapted architecture/solver comparison, not a reproduction of the source paper’s original experiment. Run all cells in order. A full run can take substantial time; save outputs before a runtime ends.\n'),
code('import sys,subprocess,json\nfrom pathlib import Path\nroot=Path("/content/experiment4")\nroot.mkdir(parents=True,exist_ok=True)\nsource_files='+repr(files)+'\nfor name,text in source_files.items():\n    (root/name).write_text(text)\n'),
code('subprocess.run([sys.executable,"-m","pip","install","torch==2.10.0","--index-url","https://download.pytorch.org/whl/cpu"],check=True)\nsubprocess.run([sys.executable,"-m","pip","install","numpy==2.3.5","jaxtyping==0.3.11","pandas==2.2.3"],check=True)\nfor name,url,commit in [("external_torchdde","https://github.com/onofriosmr/torchdde.git","4888c63dc84e966f6c6553277adc3851a2cca4fc"),("external_learnable_delays","https://github.com/onofriosmr/learnable_delays.git","9b573c4bed5efee9fe9c6b765044dbd506b5bf07")]:\n    folder=root/name\n    if not folder.exists():\n        subprocess.run(["git","clone","--branch","DCE_NDDE",url,str(folder)],check=True)\n        subprocess.run(["git","-C",str(folder),"checkout",commit],check=True)\n    assert subprocess.check_output(["git","-C",str(folder),"rev-parse","HEAD"],text=True).strip()==commit\nsubprocess.run([sys.executable,"-m","pip","install","--no-deps","-e",str(root/"external_torchdde")],check=True)\n'),
code('def run(script,*args):\n    subprocess.run([sys.executable,str(root/script),*map(str,args)],cwd=root,check=True)\ncommon=["--solver",root/"external_torchdde","--architecture",root/"external_learnable_delays"]\nrun("validate.py")\nrun("run.py","--results",root/"reserved_smoke",*common,"--mode","smoke","--stage","calibration")\nrun("run.py","--results",root/"reserved_smoke",*common,"--mode","smoke","--stage","final")\nrun("audit.py","--results",root/"reserved_smoke","--architecture",root/"external_learnable_delays")\n'),
code('results=root/"independent_results"\nrun("run.py","--results",results,*common,"--stage","calibration")\n'),
code('run("run.py","--results",results,*common,"--stage","final")\nrun("audit.py","--results",results,"--architecture",root/"external_learnable_delays")\n'),
code('import shutil\narchive=shutil.make_archive("/content/E4_Independent_Results","zip",root_dir=results)\nfrom google.colab import files\nfiles.download(archive)\n')]
for i,c in enumerate(cells):c['id']=f'e4_{i}'
n=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python3'}},cells=cells)
(HERE/'Experiment4_Colab.ipynb').write_text(json.dumps(n,indent=2,ensure_ascii=False)+'\n')
print('Notebook written.')
