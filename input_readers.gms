
$log ### reading using Connect and CSV Input


* SETTINGS
$if not set pSettings $set pSettings input/settings/pSettings.csv
$if not set y $set y input/settings/y.csv
$if not set pDuration $set pDuration input/settings/pDuration.csv
$if not set pMinGen $set pMinGen input/pMinGen.csv
$if not set pFuelPrice $set pFuelPrice input/pFuelPrice.csv

* LOAD DATA
$if not set pDemandForecast $set pDemandForecast input/demand/pDemandForecast.csv
$if not set pDemandProfile $set pDemandProfile input/demand/pDemandProfile.csv

* GENERATION DATA
$if not set pGenData $set pGenData input/gendata/pGenData.csv
$if not set pStorageData $set pStorageData input/gendata/pStorageData.csv

* AVAILABILITY
$if not set pVREgenProfile $set pVREgenProfile input/availability/pVREProfile.csv
$if not set pAvailability $set pAvailability input/availability/pAvailabilityCurrent.csv


* FIRM DATA
$if not set pFirmData $set pFirmData input/firm/pFirmData.csv


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
    file: %pStorageData%
    name: pStorageData
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pFirmData%
    name: pFirmData
    indexSubstitutions: {.nan: ""}
    indexColumns: [1]
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
    file: %pSettings%
    name: pSettings
    indexColumns: [1]
    valueColumns: [2]
    type: par
    

- GDXWriter:
    file: %GDX_INPUT%.gdx
    symbols: all
$offEmbeddedCode

