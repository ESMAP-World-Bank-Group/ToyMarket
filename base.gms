
*----------------------------------------------------------------------------------------
*                           MARKET ANALYSIS MODEL FOR South Africa
*----------------------------------------------------------------------------------------
* version: 1
* date 12.02.2023

*----------------------------------------------------------------------------------------
*                          SETS INTODUCTION
*----------------------------------------------------------------------------------------

SETS

z                              'Zones considered in CM module'
y                              'Years'
q                              'quarters or seasons'
d                              'day types'
t                              'hours of day'
i                              'Firms'
g                              'Generators considered in CM module'
f                              'Fuel'
tech                            'Technologies'

gzmap(g,z)                     'Map generators to zones'
gfmap(g,f)                     'Map generators to fuels'
gimap(g,i)                     'Map generators to firms'
gmap(g<,i<,z<,tech<,f<)           'Map generators to firms, zones, technologies and fuels'
gtechmap(g,tech)                     'Map generators to technologies'

eg(g)                          'existing generators'
committed(g)                          'committed generators'
sto(g)                          'Storage generators'
fringefirms(i)                       'Fringe generators'
cournotfirms(i)                      'Cournot-behaving generators'

sGenMap                        'Set of parameters for Generators tab'       /StYr, ReYr, Capacity, Status, HeatRate, FOM, VOM, RampUpRate,
                                                                            RampDownRate, VRE/
sStorageMap                    'Set of parameters for Generators tab'       /StorageDuration, Efficiency/
sFirmMap                        'Set of parameters for Generators tab'      /Fringe, ContractLevel/


gstatus                         'generator status' / Existing, Candidate, Committed /
gstatusmap(g,gstatus)           'Generator status map'
istatus                         'firm status' /Fringe, Cournot/
istatusmap(i, istatus)          'Firm status map'
;

ALIAS (y,yy)
;

SINGLETON SET sFirstYear(y);           
SINGLETON SET sFinalYear(y); 

;
*----------------------------------------------------------------------------------------
*                              PARAMETERS INTODUCTION
*----------------------------------------------------------------------------------------

PARAMETERS

pDuration(q<,d<,t<)            'Duration of each time segment'
A(z,y,q,d,t)                       'Coefficient A of the inverse demand function'
B(z,y,q,d,t)                       'Coefficient B of the inverse demand function'
pGenDatax(g,i,z,tech,f, sGenMap)              'Generation data'
pGenData(g,sGenMap)              'Generation data'
pStorageData(g,sStorageMap)       'Storage data'
pFirmData(i,sFirmMap)            'Firm data'
pDemandProfile(z,q,d,t)              'Demand profile'
pDemandForecast(z,*,y)                'Demand forecast'
pFixedDemand(z,y,q,d,t)         'Fixed demand profile for model with perfect competition'
pVarCost(g,y)                    'Variable costs for various products'
pFuelPrice(f,y)                  'Fuel price'
pContractVolume(i,z,y,q,d,t)       'Contract volumes of firms'
pFixedCapacity(g,y)              'Total fixed capacity in dispatch module'
pVREgenProfileTech(z,tech,q,d,t) 'VRE generation profile by hour and tech -- normalized (per MW of solar and wind capacity)'
pVREgenProfile(q,d,t,g)         'VRE generation profile by hour -- normalized (per MW of solar and wind capacity)'
pAvailability(g,q)               'Availability by generation type and season or quarter in percentage'
pMinGen(i,q)                     'Minimum generation by firm per season in percentage'
pSettings(*)                      'Parameter with scalars'
 

pRR(y)                          'Discount rate'
pWeightYear(y)

gstatIndex(gstatus) / Existing 0, Candidate 2, Committed 1 /
istatIndex(istatus) / Cournot 0, Fringe 1 /;
;

*----------------------------------------------------------------------------------------
*                              VARIABLES INTRODUCTION
*----------------------------------------------------------------------------------------

VARIABLES

vObjVal                          'Objective value'
vPrice(z,y,q,d,t)                'Price of the product in specific zone and time'


;

POSITIVE VARIABLES 
vSupplyMarket(z,y,q,d,t)          'Total supply to the zone'
vUnmetDemand(z,y,q,d,t)           'Unmet demand under the fixed demand setting'
vGenSupply(y,g,q,d,t)             'Supply of each generator'
vStorageLevel(y,g,q,d,t)          'Storage level for storage generators'
vStorageCharging(y,g,q,d,t)       'Storage charging for storage generators'

;

EQUATION

eObjFunCournot                   'Objective function for Cournot model'
eObjFunFixedDemand               'Objective function for perfect competition '
eJointCap                        'Supply is bounded by capacity'
eVRE_Limit                       'VRE production is bounded by capacity factors'
eCF                              'Seasonal (weekly) capacity factor constraint'
eDemandSupply                    'Demand-supply balance for each product and node (zone)'
eDemandSupply_FixedDemand           'Demand-supply balance for each product and node (zone)'
ePrice(z,y,q,d,t)                'Price definition as the point at the inverse demand curve'
ePriceCap(z,y,q,d,t)             'Price cap'
eMinGen(i,y,q)                   'Minimal level of energy availability, estimated per season'
eRampUP(g,y,q,d,t)                'Ramp up constraint'
eRampDOWN(g,y,q,d,t)               'Ramp down constraint'

eStorageChargingLimit(y,g,q,d,t)     'Charging cannot be higher than installed capacity'
eStorageCap(y,g,q,d,t)             'Stored energy cannot be higher than installed energy capacity'
eStorageBalance(y,g,q,d,t)        'Storage balance equation'
eStorageBalance1(y,g,q,d,t)       'Storage balance equation in the first hour'

;


*---------------- COURNOT EQUATIONS ------------------------

eObjFunCournot..
            vObjVal =E=
            sum((y,z,q,d,t), pWeightYear(y) * pRR(y) * pDuration(q,d,t) * (A(z,y,q,d,t)-0.5*B(z,y,q,d,t)*vSupplyMarket(z,y,q,d,t))*vSupplyMarket(z,y,q,d,t)) 
            - sum((y,g,q,d,t), pWeightYear(y) * pRR(y) * pDuration(q,d,t) * pVarCost(g,y)*vGenSupply(y,g,q,d,t)) 
            - sum((cournotfirms,z,y,q,d,t), pWeightYear(y) * pRR(y) * pDuration(q,d,t) * 0.5*B(z,y,q,d,t)*sqr((sum(g$(gzmap(g,z) AND gimap(g,cournotfirms)), vGenSupply(y,g,q,d,t) - vStorageCharging(y,g,q,d,t)$(sto(g))) - pContractVolume(cournotfirms,z,y,q,d,t))));
                   

eJointCap(g,y,q,d,t)..
            vGenSupply(y,g,q,d,t) =L= pFixedCapacity(g,y);
            
eVRE_Limit(g,y,q,d,t)$(pGenData(g,"VRE"))..
            vGenSupply(y,g,q,d,t) =E= pVREgenProfile(q,d,t,g)*pFixedCapacity(g,y);
            
            
eCF(g,y,q)$(NOT pGenData(g,"VRE") AND pAvailability(g,q))..
            sum((d,t), vGenSupply(y,g,q,d,t) * pDuration(q,d,t)) =L= pAvailability(g,q) * sum((d,t), pDuration(q,d,t)) * pFixedCapacity(g,y);
            
            
eDemandSupply(y,z,q,d,t)..
            sum(g$(gzmap(g,z)), vGenSupply(y,g,q,d,t)) - sum(g$(gzmap(g,z) AND sto(g)), vStorageCharging(y,g,q,d,t)) - vSupplyMarket(z,y,q,d,t) =E= 0 ;

ePrice(z,y,q,d,t)..
            vPrice(z,y,q,d,t) =E= A(z,y,q,d,t)-B(z,y,q,d,t)*(vSupplyMarket(z,y,q,d,t)) ;
            
ePriceCap(z,y,q,d,t)..
            vPrice(z,y,q,d,t) =L= pSettings('PriceCap') ;
            

eMinGen(i,y,q)$(pMinGen(i,q))..
            sum((d,t), sum(g$(gimap(g,i)), vGenSupply(y,g,q,d,t))) =G= pMinGen(i,q) * sum(g$(gimap(g,i)),pFixedCapacity(g,y)) * sum((d,t), pDuration(q,d,t));
            
            
eRampUP(g,y,q,d,t)$(ord(t)>1 AND pGenData(g,"RampUpRate") AND NOT pGenData(g,'VRE'))..
         vGenSupply(y,g,q,d,t) - vGenSupply(y,g,q,d,t-1) =L= (pFixedCapacity(g,y)*pGenData(g,"RampUpRate"));
         

eRampDOWN(g,y,q,d,t)$(ord(t)>1 AND pGenData(g,"RampDownRate") AND NOT pGenData(g,'VRE'))..
         vGenSupply(y,g,q,d,t-1) - vGenSupply(y,g,q,d,t) =L= (pFixedCapacity(g,y)*pGenData(g,"RampUpRate"));
         
    
*---------------- STORAGE EQUATIONS ------------------------
     
* Add ramp up and down for storage injection
* Storage inj should be limited with capacity
eStorageChargingLimit(y,sto,q,d,t)..
            vStorageCharging(y,sto,q,d,t) =L=  pFixedCapacity(sto,y);
            
eStorageCap(y,sto,q,d,t)..
            vStorageLevel(y,sto,q,d,t) =L= pStorageData(sto,'StorageDuration')*pFixedCapacity(sto,y);

eStorageBalance(y,sto,q,d,t)$(ord(t)>1)..
            vStorageLevel(y,sto,q,d,t)  =E= vStorageLevel(y,sto,q,d,t-1) + vStorageCharging(y,sto,q,d,t)*pStorageData(sto,'Efficiency') - vGenSupply(y,sto,q,d,t);
            
eStorageBalance1(y,sto,q,d,t)$(ord(t)=1)..
            vStorageLevel(y,sto,q,d,t)  =E= vStorageCharging(y,sto,q,d,t)*pStorageData(sto,'Efficiency') - vGenSupply(y,sto,q,d,t);

* Add CSP specific equations: need to understand their specificities

*---------------- FIXED DEMAND EQUATIONS ------------------------


eObjFunFixedDemand..
            vObjVal =E=
            - sum((y,g,q,d,t), pWeightYear(y) * pRR(y) * pDuration(q,d,t) * pVarCost(g,y)*vGenSupply(y,g,q,d,t))
            - pSettings('VOLL') * sum((z,y,q,d,t),pWeightYear(y) * pRR(y) * pDuration(q,d,t) * vUnmetDemand(z,y,q,d,t))
            ;
            

eDemandSupply_FixedDemand(y,z,q,d,t)..
            sum(g$(gzmap(g,z)), vGenSupply(y,g,q,d,t))
            - sum(g$(gzmap(g,z) AND sto(g)), vStorageCharging(y,g,q,d,t))
*            - sum(z2$sTopology(z,z2), vPowerFlow(pr,z,z2,y,w,t))
*            + sum(z2$sTopology(z,z2), vPowerFlow(pr,z2,z,y,w,t))*0.999
            + vUnmetDemand(z,y,q,d,t) - pFixedDemand(z,y,q,d,t)=E= 0 ;


model Cournot
/
eObjFunCournot
eJointCap
eVRE_Limit
eCF
eDemandSupply
ePrice
eMinGen
eRampUP
eRampDOWN
ePriceCap
eStorageChargingLimit
eStorageCap
eStorageBalance
eStorageBalance1
/ ;


model PerfectCompetition
/
eObjFunFixedDemand
eJointCap
eVRE_Limit
eCF
eDemandSupply_FixedDemand
eMinGen
eRampUP
eRampDOWN
eStorageChargingLimit
eStorageCap
eStorageBalance
eStorageBalance1
/ ;