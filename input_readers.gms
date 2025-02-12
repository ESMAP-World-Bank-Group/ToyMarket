
$ifThenI.READER "%READER%" == CONNECT_CSV
$log ### reading using Connect and CSV Input

$onEmbeddedCode Connect:

- CSVReader:
    trace: 0
    file: input/y.csv
    name: y
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set

        
- CSVReader:
    trace: 0
    file: input/pDuration.csv
    name: pDuration
    header: [1]
    indexColumns: [1, 2]
    type: par
    

- CSVReader:
    trace: 0
    file: input/pGenData.csv
    name: gmap
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2,3,4,5]
    type: set

- CSVReader:
    trace: 0
    file: input/pGenData.csv
    name: pGenDatax
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2,3,4,5]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/pFirmData.csv
    name: pFirmData
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2,3]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pDemandProfile.csv
    name: pDemandProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2,3]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pDemandForecast.csv
    name: pDemandForecast
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pVREgenProfile.csv
    name: pVREgenProfileTech
    indexColumns: [1,2,3,4]
    header: [1]
    type: par
    
    
- CSVReader:
    trace: 0
    file: input/pAvailability.csv
    name: pAvailability
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pMinGen.csv
    name: pMinGen
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/pFuelPrice.csv
    name: pFuelPrice
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pScalars.csv
    name: pScalars
    indexColumns: [1]
    valueColumns: [2]
    type: par
    

- GDXWriter:
    file: %GDX_INPUT%.gdx
    symbols: all
$offEmbeddedCode

$elseIfI.READER %READER% == CONNECT_CSV_PYTHON
$log ### reading using Connect and CSV Input with Python

$onEmbeddedCode Connect:

- CSVReader:
    trace: 0
    file: %y%
    name: y
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set

        
- CSVReader:
    trace: 0
    file: %pDuration%
    name: pDuration
    header: [1]
    indexColumns: [1, 2]
    type: par
    

- CSVReader:
    trace: 0
    file: %pGenData%
    name: gmap
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2,3,4,5]
    type: set

- CSVReader:
    trace: 0
    file: %pGenData%
    name: pGenDatax
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2,3,4,5]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pFirmData%
    name: pFirmData
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2,3]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: %pDemandProfile%
    name: pDemandProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2,3]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: %pDemandForecast%
    name: pDemandForecast
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2]
    header: [1]
    type: par

    
- CSVReader:
    trace: 0
    file: %pVREgenProfile%
    name: pVREgenProfileTech
    indexColumns: [1,2,3,4]
    header: [1]
    type: par
    
    
- CSVReader:
    trace: 0
    file: %pAvailability%
    name: pAvailability
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: %pMinGen%
    name: pMinGen
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pFuelPrice%
    name: pFuelPrice
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: %pScalars%
    name: pScalars
    indexColumns: [1]
    valueColumns: [2]
    type: par
    

- GDXWriter:
    file: %GDX_INPUT%.gdx
    symbols: all
$offEmbeddedCode

$else.READER
$abort 'No valid READER specified. Allowed are GDXXRW and CONNECT.'
$endif.READER