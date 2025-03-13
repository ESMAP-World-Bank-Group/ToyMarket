
SET s /'Least-cost', 'Cournot', 'Clearing'/ 
    tr /tr1*tr3/ ;
    


PARAMETER
        pPrice(s,z,y,q,d,t)                      ' Price by scenario, period, year, week, zone, and time slice'
        pPriceSystem(s,y,q,d,t)
        pPriceCM(s,y)
        pTotalSupply(s,y,q,d,t)
        pSupplyMarket(s,z,y,q,d,t)               ' Market supply by scenario, period, zone, year, week, and time slice'
        pGenSupply(s,y,g,q,d,t)                  ' Generation supply by scenario, generator, period, year, week, and time slice'
        pSupplyFirm(s,i,z,y,q,d,t)               ' Firm supply by scenario, period, market player, zone, year, week, and time slice'
        pSupplyFirmTotal(s,i,z,y,q,d,t)            ' Total firm supply by scenario, period, market player, year, and week'
        pSupplyFirmAnnual(s,i,z,y,q,d,t)                ' Annual firm supply by scenario, period, and market player'
        pDemand(s,z,y,q,d,t)                    'Demand under the Cournot scenario'
        pDispatch(s,z,y,q,d,t,*)                'Dispatch information, including unserved demand.'
             
;

*$ifThenI %SCENARIO% == Cournot
*    pPrice(scen,p,z,y,w,t) = (-eDemandSupply_FixedDemand.M(p,y,w,z,t)/pDuration(w,t));      
*$elseIf
*    pPrice("%SCENARIO%",z,y,q,d,t) = vPrice.L(y,z,q,d,t);
*$endIf

$ifThenI %SCENARIO% == Least-cost
    pPrice("%SCENARIO%",z,y,q,d,t) = (-eDemandSupply_FixedDemand.M(y,z,q,d,t)/pDuration(q,d,t));
$else
    pPrice("%SCENARIO%",z,y,q,d,t) = vPrice.L(z,y,q,d,t);
$endIf


pSupplyFirm("%SCENARIO%",i,z,y,q,d,t) = Sum((g)$(gimap(g,i) and gzmap(g,z)), vGenSupply.L(y,g,q,d,t));

pGenSupply("%SCENARIO%",y,g,q,d,t) = vGenSupply.L(y,g,q,d,t);

$ifThenI %SCENARIO% == Cournot
    pDemand("%SCENARIO%",z,y,q,d,t) = vSupplyMarket.L(z,y,q,d,t);
$else
    pDemand("%SCENARIO%",z,y,q,d,t) = pFixedDemand(z,y,q,d,t) - vUnmetDemand.L(z,y,q,d,t);
$endIf

pDispatch("%SCENARIO%",z,y,q,d,t,'Generation') = sum(gzmap(g,z),pGenSupply("%SCENARIO%",y,g,q,d,t));
pDispatch("%SCENARIO%",z,y,q,d,t,'Demand') = pDemand("%SCENARIO%",z,y,q,d,t);
$ifThenI %SCENARIO% == Least-cost
    pDispatch("%SCENARIO%",z,y,q,d,t,'Unmet demand') = vUnmetDemand.L(z,y,q,d,t);
$endIf


*
*pSupplyMarket("%SCENARIO%",z,y,q,d,t) = vSupplyMarket.L(y,z,q,d,t);    
*
** Calculating system price for each scenario
*pPriceSystem("%SCENARIO%",y,q,d,t)$((sum((z),pSupplyMarket("%SCENARIO%",z,y,q,d,t)))) = 
*    sum((z), pPrice("%SCENARIO%",z,y,q,d,t) * pSupplyMarket("%SCENARIO%",z,y,q,d,t))/(sum((z),pSupplyMarket("%SCENARIO%",z,y,q,d,t)));      
*Weighted average of zonal prices