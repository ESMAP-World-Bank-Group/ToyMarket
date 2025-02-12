
$ifThen not set BASE_FILE
$set BASE_FILE "base.gms"
$endIf
$log BASE_FILE is "%BASE_FILE%"

$include %BASE_FILE%

$ifThen not set REPORT_FILE
$set REPORT_FILE "report.gms"
$endIf
$log REPORT_FILE is "%REPORT_FILE%"


Set
   gstatus  'generator status' / Existing, Candidate, Committed /
   gstatusmap(g,gstatus) 'Generator status map'

*----------------------------------------------------------------------------------------
*                          DATA IMPORT FILE
*----------------------------------------------------------------------------------------

$if not set READER $set READER CONNECT_CSV

$ifThen not set READER_FILE
$set READER_FILE "input_readers.gms"
$endIf

$ifThen not set XLS_INPUT
$  if     set GDX_INPUT $set XLS_INPUT "%GDX_INPUT%.%ext%"
$  if not set GDX_INPUT $set XLS_INPUT data%system.dirsep%input%system.dirsep%MarketModelInput.xlsx
$endIf
$setNames "%XLS_INPUT%" fp GDX_INPUT fe

$log ### Debugging XLS_INPUT: %XLS_INPUT%
$log ### Debugging GDX_INPUT: %GDX_INPUT%
$log ### Debugging READER: %READER%
$log ### Debugging READER FILE: %READER_FILE%
$log ### Debugging READER COURNOT FILE: %READER_FILE_COURNOT%


* TODO: add include_readers_cournot, only if scen = Cournot

$include %READER_FILE%

$ifThen not set READER_FILE_COURNOT
$set READER_FILE_COURNOT "input_readers_cournot.gms"
$endIf

$if %SCENARIO% == Cournot $include "%READER_FILE_COURNOT%"


*$ifThen %SCENARIO% == Cournot
*$include %READER_FILE_COURNOT%
*$endIf


*----------------------------------------------------------------------------------------
*                          DATA IMPORT COMMANDS
*----------------------------------------------------------------------------------------


$gdxIn %GDX_INPUT%

* Loading parameters from the GDX file 
$load    y, pDuration, gmap, pGenDatax, pFirmData, pDemandProfile, pDemandForecast, pVREgenProfileTech, pAvailability, pMinGen, pFuelPrice, pScalars

$gdxIn



$ifThen %SCENARIO% == Cournot
$gdxIn %GDX_INPUT%_Cournot.gdx
$load A, B, pContractVolume
$gdxIn
$endIf



*----------------------------------------------------------------------------------------
*                          PREPROCESSING OF DATA
*----------------------------------------------------------------------------------------


Set

gstatusmap(g,gstatus) 'Generator status map'
;



Parameters

gstatIndex(gstatus) / Existing 0, Candidate 2, Committed 1 /
pCapacity(g,y)         'Capacity of plant per year';


* sStartYear(y) = y.first;

pGenData(g,sGenMap) = sum((i,z,tech,f), pGenDatax(g,i,z,tech,f,sGenMap));
gstatusmap(g,gstatus) = pGenData(g,'status')=gstatIndex(gstatus);


committed(g) = gstatusmap(g,'Committed');
eg(g)  = gstatusmap(g,'Existing');

pCapacity(g,y) = 0;
pCapacity(committed,y)$(y.val >= pGenData(committed,"StYr"))  = pGenData(committed,"Capacity");
pCapacity(eg,y)$(y.val >= pGenData(eg,"StYr"))  = pGenData(eg,"Capacity");


sFirstYear(y) = y.first;
sFinalYear(y) = y.last;
gimap(g,i) = sum((z,tech,f),gmap(g,i,z,tech,f)); 
gzmap(g,z) = sum((i,tech,f),gmap(g,i,z,tech,f));
gfmap(g,f) = sum((i,tech,z),gmap(g,i,z,tech,f));
gtechmap(g,tech) = sum((i,z,f),gmap(g,i,z,tech,f));

* Calculating VarCost in $/MWh
pVarCost(g,y) = pGenData(g,'VOM') + (pGenData(g,'HeatRate') * sum(f$(gfmap(g,f)), 
                        pFuelPrice(f,y)))$(sum(f$gfmap(g,f), pFuelPrice(f,y)));
                        
pVREgenProfile(q,d,t,g)$(pGenData(g,'VRE')) =
    sum(z$(gzmap(g,z)), sum(tech$gtechmap(g,tech), pVREgenProfileTech(z,tech,q,d,t)));
                        

* Calculate the weighting of each year for economic calculations, accounting for the first year, subsequent years, and the final year.
pWeightYear(y)$(sFirstYear(y)) = 1;
pWeightYear(y)$(ord(y) = 2) = y.val - sum(yy$(ord(yy)=1), yy.val);
pWeightYear(y)$(NOT sFinalYear(y) AND ord(y)>2) = sum(sameas(yy-1,y), yy.val) - y.val;
pWeightYear(y)$(sFinalYear(y) AND ord(y)>2) = sFinalYear.val - sum(sameas(yy+1,sFinalYear), yy.val);

* Compute the revenue requirement (RR) based on discount rate and the weight of each year.
pRR(y) = 1/[(1+pScalars("DR"))**(sum(yy$(ord(yy)<ord(y)), pWeightYear(yy)))];


pFixedDemand(z,y,q,d,t) = pDemandProfile(z,q,d,t) * pDemandForecast(z,'Peak',y);

execute_unload '%GDX_INPUT%_common' gimap, gzmap, gfmap, gtechmap, pVarCost, pFixedDemand, pWeightYear, pRR;


*----------------------------------------------------------------------------------------
*                 UNIVERSAL RESULTS OUTPUT PARAMETERS FOR BOTH SCENARIOS
*----------------------------------------------------------------------------------------



$if not set SCENARIO $set SCENARIO Least-cost

$ifThen.notValidScenario "%SCENARIO%" <> "Least-cost" and "%SCENARIO%" <> "Cournot" and "%SCENARIO%" <> "Clearing"
   $abort "Invalid scenario. Choose from 'Least-cost', 'Cournot', 'Clearing'"
$endIf.notValidScenario

$log ### Debugging scenario: %SCENARIO%

*----------------------------------------------------------------------------------------
*                          SOLVING MODEL DEPENDING ON OPTION
*----------------------------------------------------------------------------------------

option optcr = 0.01;
option qcp = cplex;
option limcol = 0;


$ifThenI %SCENARIO% == Least-cost
    PerfectCompetition.solprint = 2;
    PerfectCompetition.OptFile = 1;
    PerfectCompetition.dictfile = 0;

    solve PerfectCompetition using LP maximizing vObjVal;
    
    display vGenSupply.L, eDemandSupply_FixedDemand.M;

$endIf



$ifThenI %SCENARIO% == Cournot

    Cournot.solprint = 2;
    Cournot.OptFile = 1;
    Cournot.dictfile=0;

    solve Cournot using QCP maximizing vObjVal ;

$endIf


$include %REPORT_FILE%

$ifThenI %SCENARIO% == Least-cost
    execute_unload 'ResultsPerfectCompetition.gdx' pPrice, pSupplyFirm, pSupplyMarket, pDemand, pFixedDemand, pGenSupply;
$endIf

$ifThenI %SCENARIO% == Cournot
    execute_unload 'ResultsCournot.gdx' pPrice, pSupplyFirm, pSupplyMarket, pDemand, pGenSupply;
$endIf

