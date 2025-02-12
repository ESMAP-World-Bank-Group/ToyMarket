
$ifThenI.READER "%READER%" == CONNECT_CSV
$log ### reading using Connect and CSV Input

$onEmbeddedCode Connect:

    
- CSVReader:
    trace: 0
    file: input/A.csv
    name: A
    indexColumns: [1,2,3,4,5]
    valueColumns: [6]
    type: par
    
- CSVReader:
    trace: 0
    file: input/B.csv
    name: B
    indexColumns: [1,2,3,4,5]
    valueColumns: [6]
    type: par
    

- CSVReader:
    trace: 0
    file: input/pContractVolume.csv
    name: pContractVolume
    indexColumns: [1,2,3,4,5,6]
    valueColumns: [7]
    type: par
    

- GDXWriter:
    file: %GDX_INPUT%_Cournot.gdx
    symbols: all
$offEmbeddedCode

$elseIfI.READER %READER% == CONNECT_CSV_PYTHON
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

$else.READER
$abort 'No valid READER specified. Allowed are GDXXRW and CONNECT.'
$endif.READER