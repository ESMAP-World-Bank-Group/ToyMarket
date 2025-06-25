import numpy as np
import scipy.optimize as opt
import pandas as pd
from pathlib import Path
import gams.transfer as gt
import os


def input_checks(scenario):
    """Checking inputs for inconsistencies."""
    pGenData = pd.read_csv(scenario['pGenData'])
    pFirmData = pd.read_csv(scenario['pFirmData'])

    missing_in_firm = set(pGenData.firm.unique()) - set(pFirmData.firm.unique())

    if missing_in_firm:
        raise ValueError(f"Mismatch in firm definitions!\n"
                         f"Firms in pGenData but not in pFirmData: {missing_in_firm}\n")

    pAvailability = pd.read_csv(scenario['pAvailability'], index_col=0)

    pGenDataNonVRE = pGenData.loc[pGenData.VRE != 1]
    missing_in_gendata = set(pGenDataNonVRE.plants.unique()) - set(pAvailability.index.unique())

    if missing_in_gendata:
        raise ValueError(f"Mismatch in generator names in pAvailability!\n"
                         f"Generators in pAvailability do not match names in pGendata: {missing_in_gendata}\n")

    pVREProfile = pd.read_csv(scenario['pVREgenProfile'], index_col=[0,1,2,3])
    pDuration = pd.read_csv(scenario['pDuration'], index_col=[0,1])
    zone = pVREProfile.index.get_level_values('zone')[0]
    pVREProfile = pVREProfile.loc[pVREProfile.index.get_level_values('zone') == zone]
    for fuel in pVREProfile.index.get_level_values('fuel').unique():
        tmp = pVREProfile.loc[pVREProfile.index.get_level_values('fuel') == fuel, :].droplevel([0,1])
        if set(tmp.index) != set(pDuration.index):
            raise ValueError(f"VRE profile is not defined with the correct granularity. \n"
                             f"Index of pVREprofile does not match index of pDuration\n")

    pDemandProfile = pd.read_csv(scenario['pDemandProfile'], index_col=[0,1,2])
    zone = pDemandProfile.index.get_level_values('zone')[0]
    pDemandProfile = pDemandProfile.loc[pDemandProfile.index.get_level_values('zone') == zone]
    tmp = pDemandProfile.droplevel([0])
    if set(tmp.index) != set(pDuration.index):
        raise ValueError(f"Demand profile is not defined with the correct granularity. \n"
                         f"Index of pDemandProfile does not match index of pDuration\n")

    print("Input check passed.")


def extract_gdx(file, process_sets=False):
    """
    Extract information as pandas DataFrame from a gdx file.

    Parameters
    ----------
    file: str or Path
        Path to the gdx file

    Returns
    -------
    epm_result: dict
        Dictionary containing the extracted information
    """

    df = {}
    container = gt.Container(file)
    for param in container.getParameters():
        if container.data[param.name].records is not None:
            df[param.name] = container.data[param.name].records.copy()

    if process_sets:
        for gams_set in container.getSets():
            if container.data[gams_set.name].records is not None:
                df[gams_set.name] = container.data[gams_set.name].records.copy()

                if 'element_text' in df[gams_set.name].columns:
                    df[gams_set.name].drop(columns=['element_text'], inplace=True)

    return df

def process_outputs(epm_results, scenarios_rename=None, keys=None, folder=None, additional_processing=False):
    rename_columns = {'c': 'zone', 'country': 'zone', 'y': 'year', 'v': 'value', 's': 'competition', 'uni': 'attribute',
                      'z': 'zone', 'g': 'generator', 'f': 'fuel', 'q': 'season', 'd': 'day', 't': 't', 'i': 'firm',
                      'istatus': 'firmstatus'}
    if keys is None:
        keys = ['pPrice', 'pDemand', 'pSupplyFirm', 'pGenSupply', 'pDispatch', 'pSupplyFirm', 'pPlantCapacity', 'pCapacity',
                'gfmap', 'gimap', 'gtechmap', 'gzmap', 'pVarCost', 'pFirmData', 'istatusmap']
    epm_dict = {k: i.rename(columns=rename_columns) for k, i in epm_results.items() if
                k in keys and k in epm_results.keys()}

    if scenarios_rename is not None:
        for k, i in epm_dict.items():
            if 'scenario' in i.columns:
                i['scenario'] = i['scenario'].replace(scenarios_rename)

    # Convert columns to the right type
    for k, i in epm_dict.items():
        if 'year' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'year': 'int'})
        if 'value' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'value': 'float'})
        if 'zone' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'zone': 'str'})
        if 'country' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'country': 'str'})
        if 'generator' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'generator': 'str'})
        if 'fuel' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'fuel': 'str'})
        if 'tech' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'tech': 'str'})
        # if 'season' in i.columns:
        #     epm_dict[k] = epm_dict[k].astype({'season': 'str'})
        # if 'day' in i.columns:
        #     epm_dict[k] = epm_dict[k].astype({'day': 'str'})
        # if 't' in i.columns:
        #     epm_dict[k] = epm_dict[k].astype({'t': 'str'})
        if 'scenario' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'scenario': 'str'})
        if 'competition' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'competition': 'str'})
        if 'attribute' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'attribute': 'str'})
        if 'firmstatus' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'firmstatus': 'str'})

    if additional_processing:
        folder_output = Path(folder) / Path('dataframes')
        if not folder_output.exists():
            folder_output.mkdir()

        for k in keys:
            try:
                epm_dict[k].to_csv(Path(folder_output) / Path(f'{k}.csv'), float_format='%.3f')
            except Exception as e:
                print(f"Skipping {k} due to error: {e}")
        epm_dict = process_additional_dataframe(epm_dict, folder=folder, scenarios_rename=scenarios_rename)
    return epm_dict



def filter_dataframe(df, conditions):
    """
    Filters a DataFrame based on a dictionary of conditions.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be filtered.
    - conditions (dict): Dictionary specifying filtering conditions.
      - Keys: Column names in the DataFrame.
      - Values: Either a single value (exact match) or a list of values (keeps only matching rows).

    Returns:
    - pd.DataFrame: The filtered DataFrame.

    Example Usage:
    ```
    conditions = {'scenario': 'Baseline', 'year': 2050}
    filtered_df = filter_dataframe(df, conditions)
    ```
    """
    for col, value in conditions.items():
        if isinstance(value, list):
            df = df[df[col].isin(value)]
        else:
            df = df[df[col] == value]
    return df


def process_additional_dataframe(epm_results, folder, scenarios_rename=None):
    """Postprocessing existing dataframes to obtain additional dataframes"""
    scenario_df = []
    # Get scenario
    for new_folder in folder.iterdir():
        if (new_folder.is_dir()) & ('simulation' in str(new_folder) ) & ('img' not in str(new_folder)) & ('dataframes' not in str(new_folder)):
            scenario_file = new_folder / 'input' / 'scenario.csv'
            scenario = pd.read_csv(scenario_file)
            scenario_name = str(new_folder).split('/')[-1]
            scenario.loc[:, 'scenario'] = scenario_name
            scenario_df.append(scenario)

    # TODO: the best way to do that would probably be to extract dataframes from the gdx of inputs! TODO
    scenario_df = pd.concat(scenario_df, axis=0)
    pDuration_df = []
    pGenData_df = []
    for scenario in scenario_df.scenario.unique():
        tmp = scenario_df.loc[scenario_df['scenario'] == scenario].drop(columns=['scenario'])
        duration_file = tmp.set_index('paramNames').loc['pDuration'].values[0]
        pDuration = pd.read_csv(duration_file, index_col=[0,1])
        pDuration.index.names = ['season', 'day']
        pDuration = pDuration.reset_index()
        pDuration['scenario'] = scenario
        pDuration_df.append(pDuration)
        gendata_file = tmp.set_index('paramNames').loc['pGenData'].values[0]
        pGenData = pd.read_csv(gendata_file).rename(columns={'plants': 'generator'})
        pGenData = pGenData[['generator', 'Capacity']]
        pGenData.loc[:, 'scenario'] = scenario
        pGenData_df.append(pGenData)
    pDuration_df = pd.concat(pDuration_df, axis=0)
    pGenData_df = pd.concat(pGenData_df, axis=0)

    if scenarios_rename is not None:
        pDuration_df['scenario'] = pDuration_df['scenario'].replace(scenarios_rename)
        pGenData_df['scenario'] = pGenData_df['scenario'].replace(scenarios_rename)

    hours_in_season = pDuration_df.drop(columns=['day']).groupby(['season', 'scenario']).sum().sum(
        axis=1).reset_index().rename(columns={0: 'hours_in_season'})
    pDuration_reshaped =pDuration_df.set_index(['season', 'day', 'scenario']).stack().reset_index().rename(columns={'level_3': 't', 0: 'nb_hours'})

    # Energy information
    gensupply_with_zone = epm_results['pGenSupply'].copy().merge(epm_results['gzmap'], on=['scenario', 'generator'], how='left')
    pEnergyByGenerator = gensupply_with_zone.merge(pDuration_reshaped, on=['season', 'day', 't', 'scenario'], how='left').assign(
        generation=lambda df: df['value'] * df['nb_hours']
    ).groupby(['scenario', 'competition', 'zone', 'year', 'generator'], observed=False)['generation'].sum().reset_index().rename(columns={'generation': 'value'})
    pEnergyByGeneratorAndSeason = gensupply_with_zone.merge(pDuration_reshaped, on=['season', 'day', 't', 'scenario'], how='left').assign(
        generation=lambda df: df['value'] * df['nb_hours']
    ).groupby(['scenario', 'competition', 'zone', 'year', 'season', 'generator'], observed=False)['generation'].sum().reset_index().rename(columns={'generation': 'value'})

    pEnergyByFirm = epm_results['pSupplyFirm'].copy().merge(pDuration_reshaped, on=['season', 'day', 't', 'scenario'], how='left').assign(
        generation=lambda df: df['value'] * df['nb_hours']
    ).groupby(['scenario', 'competition', 'zone', 'year', 'firm'], observed=False)['generation'].sum().reset_index().rename(columns={'generation': 'value'})
    pEnergyByFirmAndSeason = epm_results['pSupplyFirm'].copy().merge(pDuration_reshaped, on=['season', 'day', 't', 'scenario'], how='left').assign(
        generation=lambda df: df['value'] * df['nb_hours']
    ).groupby(['scenario', 'competition', 'zone', 'year', 'season', 'firm'], observed=False)['generation'].sum().reset_index().rename(columns={'generation': 'value'})

    pEnergyByFuel = gensupply_with_zone.merge(epm_results['gfmap'], on=['scenario', 'generator'], how='left')
    pEnergyByFuel = pEnergyByFuel.merge(pDuration_reshaped, on=['season', 'day', 't', 'scenario'], how='left').assign(
        generation=lambda df: df['value'] * df['nb_hours']
    ).groupby(['scenario', 'competition', 'zone', 'year', 'fuel'], observed=False)['generation'].sum().reset_index().rename(columns={'generation': 'value'})

    pEnergyByFuelDispatch = gensupply_with_zone.merge(epm_results['gfmap'], on=['scenario', 'generator'], how='left')
    pEnergyByFuelDispatch = pEnergyByFuelDispatch.groupby(['scenario', 'competition', 'zone', 'year', 'season', 'day', 't', 'fuel'], observed=False)['value'].sum().reset_index()

    pEnergyByTech = gensupply_with_zone.merge(epm_results['gtechmap'], on=['scenario', 'generator'], how='left')
    pEnergyByTech = pEnergyByTech.merge(pDuration_reshaped, on=['season', 'day', 't', 'scenario'], how='left').assign(
        generation=lambda df: df['value'] * df['nb_hours']
    ).groupby(['scenario', 'competition', 'zone', 'year', 'tech'], observed=False)['generation'].sum().reset_index().rename(columns={'generation': 'value'})

    pEnergyByTechDispatch = gensupply_with_zone.merge(epm_results['gtechmap'], on=['scenario', 'generator'], how='left')
    pEnergyByTechDispatch = pEnergyByTechDispatch.groupby(['scenario', 'competition', 'zone', 'year', 'season', 'day', 't', 'tech'], observed=False)['value'].sum().reset_index()

    pEnergyFullDispatch = gensupply_with_zone.merge(epm_results['pVarCost'], on=['scenario', 'generator', 'year'], how='left').rename(columns={'value_x': 'generation', 'value_y': 'VarCost'})
    pEnergyFullDispatch = pEnergyFullDispatch.merge(epm_results['gfmap'], on=['scenario', 'generator'], how='left')
    pEnergyFullDispatch = pEnergyFullDispatch.merge(epm_results['gtechmap'], on=['scenario', 'generator'], how='left')
    pEnergyFullDispatch = pEnergyFullDispatch.merge(epm_results['gimap'], on=['scenario', 'generator'], how='left')
    try:
        pEnergyFullDispatch = pEnergyFullDispatch.merge(epm_results['istatusmap'], on=['scenario', 'firm'], how='left')
    except Exception as e:
        print(f"Skipping pEnergyFullDispatch due to error: {e}")

    pEnergyFull = pEnergyFullDispatch.merge(pDuration_reshaped, on=['season', 'day', 't', 'scenario'], how='left').assign(
        generation=lambda df: df['generation'] * df['nb_hours']
    ).groupby(['scenario', 'competition', 'zone', 'year', 'fuel', 'tech', 'firm', 'firmstatus'], observed=False)['generation'].sum().reset_index().rename(columns={'generation': 'value'})

    pPlantCapacityFactor = gensupply_with_zone.copy()
    pPlantCapacityFactor = pPlantCapacityFactor.merge(pDuration_reshaped, on=['season', 'day', 't', 'scenario'], how='left')
    pPlantCapacityFactor.loc[:, 'generation'] = pPlantCapacityFactor.loc[:, 'value'] * pPlantCapacityFactor.loc[:, 'nb_hours']

    pPlantCapacityFactor = pPlantCapacityFactor.groupby(['scenario', 'competition', 'zone', 'generator', 'season'])[
        'generation'].sum().reset_index()
    # pPlantCapacityFactor = pPlantCapacityFactor.merge(pGenData[['generator', 'Capacity']], on='generator', how='left')
    pPlantCapacityFactor = pPlantCapacityFactor.merge(pGenData_df, on=['generator', 'scenario'], how='left')
    pPlantCapacityFactor = pPlantCapacityFactor.merge(hours_in_season, on=['season', 'scenario'], how='left')
    pPlantCapacityFactor.loc[:, 'capacity_factor'] = pPlantCapacityFactor.loc[:, 'generation'] / (pPlantCapacityFactor.loc[:, 'Capacity'] * pPlantCapacityFactor.loc[:, 'hours_in_season'])

    # Demand data
    pDemandTotal = epm_results['pDemand'].copy().merge(pDuration_reshaped, on=['season', 'day', 't', 'scenario'], how='left').assign(
        demand=lambda df: df['value'] * df['nb_hours']
    ).groupby(['scenario', 'competition', 'zone', 'year'], observed=False)['demand'].sum().reset_index().rename(columns={'demand': 'value'})

    # Price information
    pPrice = epm_results['pPrice'].copy()
    pPriceWeighted = pPrice.rename(columns={'value': 'price'}).merge(
        epm_results['pDemand'].rename(columns={'value': 'demand'}), on=['scenario', 'competition', 'zone', 'year', 'season', 'day', 't'],
    how='left')
    pPriceWeighted = (
        pPriceWeighted
        .groupby(['scenario', 'competition', 'year', 'season', 'day', 't'])
        .apply(lambda x: (x['price'] * x['demand']).sum() / x['demand'].sum())
        .reset_index(name='weighted_price')
    )

    additional_df = {
        'pEnergyByGenerator': pEnergyByGenerator,
        'pEnergyByGeneratorAndSeason': pEnergyByGeneratorAndSeason,
        'pEnergyByFirm': pEnergyByFirm,
        'pEnergyByFirmAndSeason': pEnergyByFirmAndSeason,
        'pEnergyByFuel': pEnergyByFuel,
        'pEnergyByFuelDispatch': pEnergyByFuelDispatch,
        'pEnergyByTech': pEnergyByTech,
        'pEnergyByTechDispatch': pEnergyByTechDispatch,
        'pEnergyFullDispatch': pEnergyFullDispatch,
        'pEnergyFull': pEnergyFull,
        'pPlantCapacityFactor': pPlantCapacityFactor,
        'pDemandTotal': pDemandTotal,
        'pPriceWeighted': pPriceWeighted
    }

    for key, item in additional_df.items():
        epm_results[key] = additional_df[key]

    if folder is not None:
        for key, item in additional_df.items():
            additional_df[key].to_csv(Path(folder)/  Path('dataframes') / Path(f'{key}.csv'), float_format='%.3f',
                                      index=False)

    return epm_results


def extract_perfect_competition(folder, file='ResultsPerfectCompetition.gdx'):
    """Processing results from perfect competition."""
    if file in os.listdir(folder):
        file = Path(folder) / Path(file)
        epm_results = extract_gdx(file=file, process_sets=True)
        epm_dict = process_outputs(epm_results, additional_processing=False)
        epm_dict['pPrice'] = (epm_dict['pPrice'].loc[epm_dict['pPrice'].competition == 'Least-cost']).drop(
            columns=['competition'])
        epm_dict['pDemand'] = (epm_dict['pDemand'].loc[epm_dict['pDemand'].competition == 'Least-cost']).drop(
            columns=['competition'])
    else:
        print('File not found:', file)
        epm_dict = None

    return epm_dict


def extract_simulation_folder(results_folder):
    required_files = {'ResultsPerfectCompetition.gdx', 'ResultsCournot.gdx', 'MarketModelInput_common.gdx'}
    dict_df = {}
    for scenario in [i for i in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, i))]:
        scenario_files = set(os.listdir(Path(results_folder) / Path(scenario)))
        if required_files.issubset(scenario_files):
            dict_df.update({scenario: extract_scenario_folder(Path(results_folder) / Path(scenario))})

    inverted_dict = {
        k: {outer: inner[k] for outer, inner in dict_df.items() if k in inner}
        for k in {key for inner in dict_df.values() for key in inner}
    }

    inverted_dict = {k: pd.concat(v, names=['scenario']).reset_index('scenario') for k, v in inverted_dict.items()}

    return inverted_dict


def extract_scenario_folder(folder):
    file = 'ResultsPerfectCompetition.gdx'
    if file in os.listdir(folder):
        file = Path(folder) / Path(file)
        epm_results_competition = extract_gdx(file=file)
        # epm_dict = process_outputs(epm_results)

    file = 'ResultsCournot.gdx'
    if file in os.listdir(folder):
        file = Path(folder) / Path(file)
        epm_results_cournot = extract_gdx(file=file)
        # epm_dict = process_outputs(epm_results)

    epm_results = {}
    for key in epm_results_cournot.keys():
        if key in epm_results_competition.keys():
            epm_results[key] = pd.concat([epm_results_competition[key], epm_results_cournot[key]], axis=0)

    file = 'MarketModelInput_common.gdx'
    if file in os.listdir(folder):
        file = Path(folder) / Path(file)
        epm_results_common = extract_gdx(file=file, process_sets=True)

    epm_results.update(epm_results_common)

    return epm_results



def estimate_demand_slope_intercept(demand, price, elasticity, folder, scenario):
    """Calculates slope and intercept of linear inverse demand, relying on perfect competition data on price and demand,
    and on elasticity values."""
    B = (1/(-elasticity)) * (price / demand)
    A = price + demand * B

    if not (Path(folder) / Path('input')).is_dir():
        os.mkdir(Path(folder) / Path('input'))

    A.to_csv(Path(folder) / Path('input') / Path('A.csv'), float_format='%.3f')
    B.to_csv(Path(folder) / Path('input') / Path('B.csv'), float_format='%.3f')

    # adding path to files
    scenario.loc['A'] = Path(folder) / Path('input') / Path('A.csv')
    scenario.loc['B'] = Path(folder) / Path('input') / Path('B.csv')
    return scenario

def estimate_hourly_contracting(pSupplyFirm, pFirmData, folder, scenario):
    """Calculates hourly contracted volume per firm based on supply under perfect competition setting.
    This will only estimate contracted volume for Cournot firms."""
    pFirmData = pFirmData.copy()
    if 'ContractPrice' in pFirmData.columns:
        pFirmData = pFirmData.drop(columns=['ContractPrice'])
    pContractVolume = (pFirmData * pSupplyFirm).dropna()

    if not (Path(folder) / Path('input')).is_dir():
        os.mkdir(Path(folder) / Path('input'))

    pContractVolume.to_csv(Path(folder) / Path('input') / Path('pContractVolume.csv'), float_format='%.3f')

    scenario.loc['pContractVolume'] = Path(folder) / Path('input') / Path('pContractVolume.csv')
    return scenario


def load_additional_data(scenario):
    """Gets additional data required."""
    pScalars = pd.read_csv(scenario['pSettings'], index_col=0)
    elasticity = pScalars.loc['elasticity', 'Value']
    additional_data = {
        'elasticity': elasticity
    }

    pFirmData = pd.read_csv(scenario['pFirmData'], index_col=[0])
    pFirmData = pFirmData.dropna().drop(columns='Fringe').rename(columns={'ContractLevel': 'value'})
    additional_data['pFirmData'] = pFirmData
    return additional_data

def prepare_data_for_cournot(d, scenario, additional_data, folder, contract):
    """
    Prepares data for a Cournot competition model by estimating demand parameters
    and determining contract levels at either the firm or plant level.

    Parameters:
    -----------
    d : dict
        Dictionary containing input datasets as Pandas DataFrames:
        - 'pDemand': Demand data indexed by ['zone', 'year', 'season', 'day', 't'].
        - 'pPrice': Price data indexed by ['zone', 'year', 'season', 'day', 't'].
        - 'pSupplyFirm' (if contract='firm'): Supply data at the firm level, indexed by ['firm', 'zone', 'year', 'season', 'day', 't'].
        - 'pGenSupply' (if contract='plant'): Supply data at the generator level, indexed by ['generator', 'year', 'season', 'day', 't'].
        - 'gimap': Mapping between generators and firms.

    scenario : dict
        Dictionary containing scenario-specific information that will be updated.

    additional_data : dict
        Additional data needed for demand and contracting estimation:
        - 'elasticity': Demand elasticity values.
        - 'pFirmData': Data specifying firm-level contract information.

    folder : str
        Path to the folder where intermediate or output data may be stored.

    contract : str
        Specifies the contract level, must be either:
        - 'firm': Contracting is handled at the firm level.
        - 'plant': Contracting is handled at the plant level.

    Returns:
    --------
    scenario : dict
        Updated scenario dictionary with estimated demand parameters and contract allocations.

    Raises:
    -------
    AssertionError:
        If the `contract` parameter is not 'firm' or 'plant'.

    Notes:
    ------
    - The function first estimates demand slope and intercept using `estimate_demand_slope_intercept`.
    - If contracting is at the firm level, supply data is processed from 'pSupplyFirm'.
    - If contracting is at the plant level, supply data is processed from 'pGenSupply'
      and mapped using 'gimap'.
    - Contracting at the plant level is actually only relevant if the level of contracting differs across plants belong to the same firm.
    If the level of contracting is the same, then working directly at the firm level will yield the same results.
    """
    assert contract in ['firm', 'plant'], 'Method used to define contract level is not yet implemented.'
    demand = d['pDemand'].set_index([ 'zone', 'year', 'season', 'day', 't'])
    price = d['pPrice'].set_index([ 'zone', 'year', 'season', 'day', 't'])
    scenario = estimate_demand_slope_intercept(demand, price, additional_data['elasticity'], folder, scenario)
    if contract == 'firm':  # contract level is defined at the firm level
        supplyfirm = d['pSupplyFirm'].drop(columns=['competition']).set_index(['firm', 'zone', 'year', 'season', 'day', 't'])
        scenario = estimate_hourly_contracting(supplyfirm, additional_data['pFirmData'], folder, scenario)
    else:  # contract level is defined at the plant level
        contract_data = d['gimap'].merge(additional_data['pFirmData'].reset_index()[['firm', 'value']], on='firm', how='left').fillna(0).drop(columns=['firm']).set_index(['generator'])
        contract_data = contract_data[contract_data['value'] > 0]
        gensupply = d['pGenSupply'].drop(columns=['competition']).set_index(['generator', 'year', 'season', 'day', 't'])
        scenario = estimate_hourly_contracting(gensupply, contract_data, folder, scenario)

    return scenario

def get_duration_curve(value_df, duration_df):
    """
    Generate a duration curve by expanding time slices using weights from the duration_df
    and sorting the values within each scenario and competition group.

    Parameters
    ----------
    value_df : pandas.DataFrame
        A DataFrame containing at least the following columns:
        - 'scenario': scenario name
        - 'competition': competition identifier
        - 'season', 'day', 't': indices for time slice
        - 'value': numerical value (e.g., price, dispatch)

    duration_df : pandas.DataFrame
        A multi-index DataFrame (with levels ['season', 'day', 't']) containing a single value per cell,
        representing the weight (e.g., number of hours) of each time slice.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each original value has been repeated according to its duration weight,
        sorted in descending order of 'value' within each ['scenario', 'competition'] group,
        and annotated with an 'hour' index (from 0 to N-1) indicating its position in the sorted curve.

    Notes
    -----
    This function is used to build duration curves (e.g., load duration, price duration, etc.)
    from time-slice-based model outputs. The result is useful for visualizing distributions
    over a representative year, accounting for the relative frequency (duration) of each time slice.
    """
    value_df = value_df.copy()

    # Reshape duration to merge properly
    duration_df = duration_df.stack().to_frame().reset_index().rename(
        columns={0: 'weight', 'level_0': 'season', 'level_1': 'day', 'level_2': 't'})

    # Merge price and duration_df
    value_df = value_df.merge(duration_df, on=['season', 'day', 't'], how='left')

    # Function to expand and add an hour index
    def expand_group(df):
        expanded_df = df.loc[df.index.repeat(df['weight'])].reset_index(drop=True)
        expanded_df = expanded_df.sort_values(by='value', ascending=False)
        expanded_df['hour'] = range(len(expanded_df))  # Assign duration from 0 to n for each group
        return expanded_df

    # Apply function per group
    expanded_value_df = value_df.groupby(['scenario', 'competition', 'zone'], group_keys=False).apply(expand_group)
    return expanded_value_df


def adjust_demand(demand_forecast, demand_profile, duration, hours=None):
    """
    Adjusts demand to match peak and energy forecast while minimizing deviation.
    """
    demand_profile = demand_profile.stack().rename_axis(index=['zone', 'season', 'day', 'time'])
    duration = duration.stack().rename_axis(index=['season', 'day', 'time'])
    pmax = demand_profile.groupby(level='zone').max()
    pmin = demand_profile.loc[demand_profile > 0].groupby(level='zone').min()

    demand_forecast_energy = (demand_forecast.loc[demand_forecast.index.get_level_values('type') == 'Energy']).droplevel('type')
    demand_forecast_peak = (demand_forecast.loc[demand_forecast.index.get_level_values('type') == 'Peak']).droplevel('type')

    demand_forecast_peak = demand_forecast_peak.reindex(demand_profile.index, level='zone')
    ptempdemand = demand_forecast_peak.mul(demand_profile, axis=0)

    pdiff = demand_forecast_energy * 1e3 - ptempdemand.mul(duration, axis=0).groupby('zone').sum()

    cardinalities = [duration.index.get_level_values(name).nunique() for name in duration.index.names]
    divisor_lo = np.prod(cardinalities)

    z_size = demand_profile.index.get_level_values('zone').nunique()
    q_size = demand_profile.index.get_level_values('season').nunique()  # Assuming 'season' ≈ q
    d_size = demand_profile.index.get_level_values('day').nunique()
    t_size = demand_profile.index.get_level_values('time').nunique()
    y_size = demand_forecast_energy.shape[1]  # Assuming y is a column index

    # Define decision variable sizes
    adjustment_factor_size = (z_size, q_size, d_size, y_size, t_size)
    divisor_size = (z_size, y_size)

    # Flatten initial values
    x0_adjustment = np.zeros(adjustment_factor_size).ravel()
    x0_divisor = np.full(divisor_size, divisor_lo).ravel()  # Start at lower bound
    x0 = np.concatenate([x0_adjustment, x0_divisor])

    # Indices for separating variables in `x`
    split_idx = x0_adjustment.size

    # adjustment_factor = ((2 * pdiff / divisor).mul((pmax - demand_profile), axis=0)).div(((pmax - pmin) ** 2), axis=0)
    # adjustment_factor = (
    #     adjustment_factor.stack()
    #     .rename_axis(index=['zone', 'season', 'day', 'time', 'year'])
    #     .reorder_levels(['zone', 'season', 'day', 'year', 'time'])  # Move 'year' before 'time'
    # )
    # adjustment_factor = adjustment_factor.to_numpy().reshape(z_size, q_size, d_size, y_size, t_size)

    pdiff = pdiff.stack().to_numpy().reshape(z_size, y_size)
    duration = duration.to_numpy().reshape(q_size, d_size, t_size)
    demand_profile = demand_profile.to_numpy().reshape(z_size, q_size, d_size, t_size)
    pmax = pmax.to_numpy()
    pmin = pmin.to_numpy()

    # Define objective function (sum of adjustments)
    def objective(x):
        return np.sum(x[:split_idx])
        # return np.sum(x)

    # Constraint to match energy balance
    def area_constraint(x):
        adjustment_factor = x[:split_idx].reshape(adjustment_factor_size)
        lhs = np.sum(adjustment_factor * duration[np.newaxis, :, :, np.newaxis, :], axis=(1, 2, 4))
        return (lhs - pdiff).ravel()
        # x_reshaped = x.reshape(adjustment_factor.shape)
        # lhs = np.sum(x_reshaped * duration[np.newaxis, :, :, np.newaxis, :], axis=(1, 2, 4))
        # return (lhs - pdiff).ravel()

    def divisor_constraint(x):
        adjustment_factor = x[:split_idx].reshape(adjustment_factor_size)
        divisor = x[split_idx:].reshape(divisor_size)

        # Constraint:
        lhs = adjustment_factor - (2 * pdiff[:, np.newaxis, np.newaxis, :, np.newaxis] / divisor[:, np.newaxis, np.newaxis, :, np.newaxis]) * \
              (pmax[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] - demand_profile[:, :, :, np.newaxis,:]) / \
              (pmax[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] - pmin[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis])**2

        return lhs.ravel()
        # x_reshaped = x.reshape(adjustment_factor.shape)
        # return (x_reshaped - adjustment_factor).ravel()

    def divisor_lower_bound(x):
        divisor = x[split_idx:].reshape(divisor_size)
        return (divisor - divisor_lo).ravel()  # Ensure divisor ≥ divisor_lo

    # # Bounds: Adjustment variables must be within pmin and pmax
    # bounds_adjustment = [(pmin[i, j, k] - demand_profile[i, j, k],
    #                       pmax[i, j, k] - demand_profile[i, j, k])
    #                      for i in range(z_size) for j in range(q_size) for k in range(d_size * y_size * t_size)]
    #
    # bounds_divisor = [(divisor_lo, None)] * np.prod(divisor_size)  # Ensure divisor ≥ divisor_lo
    #
    # bounds = bounds_adjustment + bounds_divisor

    # Solve optimization
    constraints = [
        {'type': 'eq', 'fun': divisor_constraint},  # pyval equation
        {'type': 'eq', 'fun': area_constraint},         # Energy balance
        {'type': 'ineq', 'fun': divisor_lower_bound}    # divisor >= divisor_lo
    ]

    # TODO: code bugging, needs to be solved

    result = opt.minimize(objective, x0, constraints=constraints)

    return 0


