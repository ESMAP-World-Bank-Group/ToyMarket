

$if not set A $set A output/simulations_run_20250625_151552/TwoZone2030_NoTransmission/input/A.csv
$if not set B $set B output/simulations_run_20250625_151552/TwoZone2030_NoTransmission/input/B.csv
$if not set pContractVolume $set pContractVolume output/simulations_run_20250625_151552/TwoZone2030_NoTransmission/input/pContractVolume.csv

$log ### reading using Connect and CSV Input with Python

$onEmbeddedCode Connect:

    
- CSVReader:
    trace: 0
    file: %A%
    name: A
    indexColumns: [1,2,3,4,5]
    valueColumns: [6]
    type: par
    
- CSVReader:
    trace: 0
    file: %B%
    name: B
    indexColumns: [1,2,3,4,5]
    valueColumns: [6]
    type: par
    
- CSVReader:
    trace: 0
    file: %pContractVolume%
    name: pContractVolume
    indexColumns: [1,2,3,4,5,6]
    valueColumns: [7]
    type: par
    

- GDXWriter:
    file: %GDX_INPUT%_Cournot.gdx
    symbols: all
$offEmbeddedCode

