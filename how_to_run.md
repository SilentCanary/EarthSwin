Updated todo list

I’ll trace the actual entry points, environment requirements, dataset paths, and train/test scripts across the repository, then give you a command sequence that matches this code rather than relying only on the README.







Search complete: Trace run commands

Searched codebase for "EarthSwin repository install dependencies prepare data run tests train model perform inference evaluation", no results

Searched for files matching `**/EarthSwin/**`, 50 matches

Searched for files matching `**/EarthSwin/README*`, 1 match

Searched for files matching `**/EarthSwin/environment.yml`, 1 match

Searched for files matching `**/EarthSwin/config.py`, 1 match

Searched for files matching `**/EarthSwin/**/*.py`, 40 matches

Searched for files matching `**/EarthSwin/**/*.ipynb`, no matches

Searched for files matching `**/EarthSwin/tests/**/*.py`, no matches

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Searched for text `EarthSwin`, 19 results

Searched for text `preprocessing`, 3 results

Searched for text `diagnose`, 1 result

Searched for files matching `**/EarthSwin/models/train_earthswin.py`, 1 match

Read 

Read 

Read 

Searched for text `train.py`, 1 result

Searched for files matching `**/EarthSwin/models/**/*test*`, 11 matches

Read , lines 1 to 260

Read 

Read , lines 1 to 260

Read , lines 1 to 260

Read 

Read 

The main issue is already visible: the README’s training command is incomplete, and several scripts still contain absolute paths from the original Windows machine. I’m checking every executable script and test entry point now so I can separate commands that work immediately from commands that require path edits or external data.

Ran terminal command: Get-ChildItem -Recurse -File | Select-Object FullName | Sort-Object FullName; Write-Output '--- ENTRYPOINTS ---'; rg -n "if __name__|argparse|click|pytest|DataLoader|torch\.save|torch\.load|def main|train|test" --glob "*.py" .; Write-Output '--- ABSOLUTE PATHS ---'; rg -n "[A-Za-z]:\\|[A-Za-z]:/|dataset/|DATA|PATH" --glob "*.py" --glob "*.yml" --glob "README.md" .
Completed with input: Get-ChildItem -Recurse -File | Select-Object FullName | Sort-Object FullName; Write-Output '--- ENTRYPOINTS ---'; rg -n "if __name__|argparse|click|pytest|DataLoader|torch\.save|torch\.load|def main|train|test" --glob "*.py" .; Write-Output '--- ABSOLUTE PATHS ---'; rg -n "[A-Za-z]:\\|[A-Za-z]:/|dataset/|DATA|PATH" --glob "*.py" --glob "*.yml" --glob "README.md" .

Ran terminal command: Get-ChildItem -Recurse -Filter *.py | Select-String -Pattern 'if __name__|argparse|DataLoader|torch\.save|torch\.load|def main|train|pytest' | Select-Object Path,LineNumber,Line | Format-Table -AutoSize | Out-String -Width 240

Ran terminal command: Get-ChildItem -Recurse -Include *.py,*.yml,README.md | Select-String -Pattern '[A-Za-z]:\\|[A-Za-z]:/|dataset/|DATA|PATH' | Select-Object Path,LineNumber,Line | Format-Table -AutoSize | Out-String -Width 240

Ran terminal command: Get-Content models\train_earthswin.py -Tail 220; Write-Output '--- graph builder ---'; Get-Content models\build_weekly_graph.py -First 260; Write-Output '--- preprocessing scripts ---'; Get-Content preprocessing\create_chunks.py -First 180; Get-Content preprocessing\create_patches.py -First 180
Completed with input: Get-Content models\train_earthswin.py -Tail 220; Write-Output '--- graph builder ---'; Get-Content models\build_weekly_graph.py -First 260; Write-Output '--- preprocessing scripts ---'; Get-Content preprocessing\create_chunks.py -First 180; Get-Content preprocessing\create_patches.py -First 180

The repository is one level below the terminal’s current directory, and the first scan also confirms `rg` is unavailable on this machine. I’m rerunning the inspection from `EarthSwin` using PowerShell-native search, then I’ll give you copy-paste commands for both the component tests and the data pipeline.

Ran terminal command: Set-Location E:\major\EarthSwin; Get-ChildItem -Recurse -Filter *.py | Select-String -Pattern 'if __name__|argparse|DataLoader|torch\.save|torch\.load|def main|pytest|train' | Select-Object Path,LineNumber,Line | Format-Table -AutoSize | Out-String -Width 240

Ran terminal command: Get-ChildItem -Recurse -Include *.py,*.yml,README.md | Select-String -Pattern '[A-Za-z]:\\|[A-Za-z]:/|dataset/|DATA|PATH' | Select-Object Path,LineNumber,Line | Format-Table -AutoSize | Out-String -Width 240

Ran terminal command: Write-Output '--- train tail ---'; Get-Content models\train_earthswin.py -Tail 220; Write-Output '--- graph builder ---'; Get-Content models\build_weekly_graph.py -First 260; Write-Output '--- preprocessing headers ---'; Get-Content preprocessing\create_chunks.py -First 180; Get-Content preprocessing\create_patches.py -First 180

Ran terminal command: Write-Output '--- executable markers ---'; Get-ChildItem -Recurse -Filter *.py | ForEach-Object { $p=$_.FullName; $n=0; Get-Content $p | ForEach-Object { $n++; if ($_ -match 'if __name__|argparse|def main|pytest|torch\.save|torch\.load') { '{0}:{1}: {2}' -f $p,$n,$_ } } }

Ran terminal command: Write-Output '--- hard-coded paths ---'; Get-ChildItem -Recurse -Include *.py,*.yml,README.md | Select-String -Pattern '[A-Za-z]:\\|[A-Za-z]:/' | ForEach-Object { '{0}:{1}: {2}' -f $_.Path,$_.LineNumber,$_.Line.Trim() }

Ran terminal command: Write-Output '--- train main ---'; $x=Get-Content models\train_earthswin.py; $x[250..($x.Length-1)]; Write-Output '--- graph main ---'; $x=Get-Content models\build_weekly_graph.py; $x[0..($x.Length-1)]

Ran terminal command: $files='preprocessing\create_chunks.py','preprocessing\create_patches.py','preprocessing\create_labels.py','preprocessing\calculate_normalization_stats.py','preprocessing\prepare_dataset.py','preprocessing\create_pytorch_dataset.py','models\build_weekly_graph.py','models\train_earthswin.py'; foreach($f in $files){ Write-Output "--- $f ---"; $x=Get-Content $f; $hits=Select-String -Path $f -Pattern 'if __name__|^def main|INPUT|OUTPUT|DIR|PATH|WEEK|week|split|csv' ; $hits | Select-Object -First 80 | ForEach-Object { "$($_.LineNumber): $($_.Line.Trim())" } }

Ran terminal command: Get-Content models\test\test.py -First 80; Write-Output '--- test tails ---'; Get-ChildItem models\test\*.py | ForEach-Object { $f=$_.FullName; Write-Output "--- $f ---"; Get-Content $f -Tail 12 }

Ran terminal command: Write-Output '--- preprocessing main blocks ---'; $files='preprocessing\create_chunks.py','preprocessing\create_patches.py','preprocessing\create_labels.py','preprocessing\calculate_normalization_stats.py','preprocessing\prepare_dataset.py','preprocessing\create_pytorch_dataset.py'; foreach($f in $files){ Write-Output "--- $f ---"; $x=Get-Content $f; $start=[Math]::Max(0,$x.Length-75); $x[$start..($x.Length-1)] }

Ran terminal command: foreach($f in 'preprocessing\create_chunks.py','preprocessing\create_patches.py','preprocessing\create_labels.py','preprocessing\calculate_normalization_stats.py','preprocessing\prepare_dataset.py','preprocessing\create_pytorch_dataset.py','models\build_weekly_graph.py'){ Write-Output "--- $f ---"; $x=Get-Content $f; $x[0..35]; $x[($x.Length-45)..($x.Length-1)] }

Ran terminal command: Write-Output '--- training config and main ---'; $x=Get-Content models\train_earthswin.py; $x[0..70]; $x[590..($x.Length-1)]

Ran terminal command: Write-Output '--- repository data files ---'; Get-ChildItem -Recurse -File | Where-Object { $_.Extension -in '.tif','.npz','.csv','.json','.pt' } | Select-Object FullName,Length | Format-Table -AutoSize | Out-String -Width 220

The local checkout contains no imagery, patch archives, CSV manifests, graphs, or model checkpoints, so only dependency setup and data-free component tests can run here. The full pipeline must use the separately supplied dataset, and it will not run unchanged until the embedded `C:\Users\advit\...` paths are redirected to your machine.



Ran terminal command: foreach($f in 'models\test\test_attention.py','models\test\test_fusion.py','models\test\test_cnn.py','models\test\test_gat.py','models\test\test_patch_pyramid.py','models\test\test_swin.py','models\test\test_temporal.py'){ Write-Output "--- $f ---"; Get-Content $f -First 35 }

I read the repository. The README is incomplete, and the current checkout has no dataset files. Also, many scripts still reference the original path:

`C:\Users\advit\Documents\major project`

## 1. Create the environment

From PowerShell:

```powershell
Set-Location E:\major\EarthSwin

conda env create -f environment.yml
conda activate landslide
```

Verify PyTorch:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Set the model import path for this terminal:

```powershell
$env:PYTHONPATH = "E:\major\EarthSwin\models"
```

## 2. Important path requirement

Before running preprocessing or training, replace every occurrence of:

```text
C:/Users/advit/Documents/major project
```

with:

```text
E:/major/EarthSwin
```

Affected files include:

- merge.py
- diagnose.py
- preprocessing/create_chunks.py
- preprocessing/create_patches.py
- preprocessing/create_labels.py
- preprocessing/calculate_normalization_stats.py
- preprocessing/prepare_dataset.py
- models/build_weekly_graph.py
- models/train_earthswin.py
- several files under `models/test`

The dataset should eventually have this structure:

```text
E:\major\EarthSwin\dataset\EarthSentinel_2016\
    HP_week_*.tif
    merged\
    patch_chunks\
    multiscale_patches\
    temporal_dataset\
    graphs\
```

The required raw imagery and processed data are not currently in the repository.

## 3. Preprocessing sequence

After placing the raw `.tif` files under `dataset\EarthSentinel_2016`, run:

```powershell
Set-Location E:\major\EarthSwin
conda activate landslide

python merge.py
python preprocessing\create_chunks.py
python preprocessing\create_patches.py
python preprocessing\create_labels.py
python preprocessing\calculate_normalization_stats.py
python preprocessing\prepare_dataset.py
python preprocessing\create_pytorch_dataset.py
```

Then inspect the generated data:

```powershell
python preprocessing\diagnose.py
```

Note: `merge.py` currently processes only weeks `1, 2, 3, 4, 5, 13, 14`, despite printing that all 14 weeks were merged. The remaining weeks may need to be added before running the complete pipeline.

## 4. Build spatial graphs

After the temporal CSV files and multiscale patches exist:

```powershell
python models\build_weekly_graph.py
```

This should create:

```text
dataset\EarthSentinel_2016\graphs\
    train\
    val\
    test\
```

## 5. Run training

```powershell
Set-Location E:\major\EarthSwin
conda activate landslide
$env:PYTHONPATH = "E:\major\EarthSwin\models"

python models\train_earthswin.py
```

Training should produce:

```text
models\earthswin_best.pt
models\training_history.json
```

## 6. Run available tests

The self-contained tests are:

```powershell
Set-Location E:\major\EarthSwin
conda activate landslide
$env:PYTHONPATH = "E:\major\EarthSwin\models"

python models\test\test_attention.py
python models\test\test_fusion.py
```

Most other tests require the external dataset and the original hard-coded paths:

```powershell
python models\test\test_cnn.py
python models\test\test_gat.py
python models\test\test_graph.py
python models\test\test_graph_loader.py
python models\test\test_patch_pyramid.py
python models\test\test_swin.py
python models\test\test_temporal.py
python models\test\test_temporal_model.py
python models\test\test.py
```

At present, those data-dependent tests and training cannot run because the dataset, CSV manifests, graph files, and patch files are absent from the checkout.

Created 3 todos