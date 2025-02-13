import numpy as np
import scipy.optimize as opt
import pandas as pd
from pathlib import Path
import gams.transfer as gt
import os

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
                      'z': 'zone', 'g': 'generator', 'f': 'fuel', 'q': 'season', 'd': 'day', 't': 't', 'i': 'firm'}
    if keys is None:
        keys = ['pPrice', 'pDemand', 'pSupplyFirm', 'pGenSupply', 'pSupplyFirm', 'gfmap', 'gimap', 'gtechmap', 'pVarCost']
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
        if 'scenario' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'scenario': 'str'})
        if 'competition' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'competition': 'str'})
        if 'attribute' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'attribute': 'str'})

    if additional_processing:
        folder = Path(folder) / Path('output')
        if not folder.exists():
            folder.mkdir()

        for k in keys:
            epm_dict[k].to_csv(Path(folder) / Path(f'{k}.csv'), float_format='%.3f')
        epm_dict = process_additional_dataframe(epm_dict, folder=folder)
    return epm_dict


def process_additional_dataframe(epm_results, folder=None):
    """Postprocessing existing dataframes to obtain additional dataframes"""
    pEnergyByGenerator = epm_results['pGenSupply'].copy().groupby(['scenario', 'competition', 'year', 'generator'], observed=False)['value'].sum().reset_index()
    pEnergyByGeneratorAndSeason = epm_results['pGenSupply'].copy().groupby(['scenario', 'competition', 'year', 'season', 'generator'], observed=False)['value'].sum().reset_index()
    pEnergyByFirm = epm_results['pSupplyFirm'].copy().groupby(['scenario', 'competition', 'year', 'firm'], observed=False)['value'].sum().reset_index()
    pEnergyByFirmAndSeason = epm_results['pSupplyFirm'].copy().groupby(['scenario', 'competition', 'year', 'season', 'firm'], observed=False)['value'].sum().reset_index()

    pEnergyByFuel = epm_results['pGenSupply'].merge(epm_results['gfmap'], on=['scenario', 'generator'], how='left')
    pEnergyByFuel = pEnergyByFuel.groupby(['scenario', 'competition', 'year', 'fuel'], observed=False)['value'].sum().reset_index()

    pEnergyByFuelDispatch = epm_results['pGenSupply'].merge(epm_results['gfmap'], on=['scenario', 'generator'], how='left')
    pEnergyByFuelDispatch = pEnergyByFuelDispatch.groupby(['scenario', 'competition', 'year', 'season', 'day', 't', 'fuel'], observed=False)['value'].sum().reset_index()

    pEnergyByTech = epm_results['pGenSupply'].merge(epm_results['gtechmap'], on=['scenario', 'generator'], how='left')
    pEnergyByTech = pEnergyByTech.groupby(['scenario', 'competition', 'year', 'tech'], observed=False)['value'].sum().reset_index()

    pEnergyByTechDispatch = epm_results['pGenSupply'].merge(epm_results['gtechmap'], on=['scenario', 'generator'], how='left')
    pEnergyByTechDispatch = pEnergyByTechDispatch.groupby(['scenario', 'competition', 'year', 'season', 'day', 't', 'tech'], observed=False)['value'].sum().reset_index()

    pGenSupplyWithCost = epm_results['pGenSupply'].merge(epm_results['pVarCost'], on=['scenario', 'generator', 'year'], how='left').rename(columns={'value_x': 'generation', 'value_y': 'VarCost'})
    pGenSupplyWithCost = pGenSupplyWithCost.merge(epm_results['gfmap'], on=['scenario', 'generator'], how='left')
    pGenSupplyWithCost = pGenSupplyWithCost.merge(epm_results['gtechmap'], on=['scenario', 'generator'], how='left')
    pGenSupplyWithCost = pGenSupplyWithCost.merge(epm_results['gimap'], on=['scenario', 'generator'], how='left')

    additional_df = {
        'pEnergyByGenerator': pEnergyByGenerator,
        'pEnergyByGeneratorAndSeason': pEnergyByGeneratorAndSeason,
        'pEnergyByFirm': pEnergyByFirm,
        'pEnergyByFirmAndSeason': pEnergyByFirmAndSeason,
        'pEnergyByFuel': pEnergyByFuel,
        'pEnergyByFuelDispatch': pEnergyByFuelDispatch,
        'pEnergyByTech': pEnergyByTech,
        'pEnergyByTechDispatch': pEnergyByTechDispatch,
        'pGenSupplyWithCost': pGenSupplyWithCost
    }

    for key, item in additional_df.items():
        epm_results[key] = additional_df[key]

    if folder is not None:
        for key, item in additional_df.items():
            additional_df[key].to_csv(Path(folder) / Path(f'{key}.csv'), float_format='%.3f')

    return epm_results


def extract_perfect_competition(folder, file='ResultsPerfectCompetition.gdx'):
    """Processing results from perfect competition."""
    if file in os.listdir(folder):
        file = Path(folder) / Path(file)
        epm_results = extract_gdx(file=file)
        epm_dict = process_outputs(epm_results, additional_processing=False)
        epm_dict['pPrice'] = (epm_dict['pPrice'].loc[epm_dict['pPrice'].competition == 'Least-cost']).drop(
            columns=['competition'])
        epm_dict['pDemand'] = (epm_dict['pDemand'].loc[epm_dict['pDemand'].competition == 'Least-cost']).drop(
            columns=['competition'])
    else:
        print('File not found:', file)
        epm_dict = None

    return epm_dict


def extract_simulation_folder(folder):
    dict_df = {}
    for scenario in [i for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))]:
        if 'simulation' in scenario:
            dict_df.update({scenario: extract_scenario_folder(Path(folder) / Path(scenario))})

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
    pContractVolume = (pFirmData * pSupplyFirm).dropna()

    if not (Path(folder) / Path('input')).is_dir():
        os.mkdir(Path(folder) / Path('input'))

    pContractVolume.to_csv(Path(folder) / Path('input') / Path('pContractVolume.csv'), float_format='%.3f')

    scenario.loc['pContractVolume'] = Path(folder) / Path('input') / Path('pContractVolume.csv')
    return scenario


def load_additional_data(scenario):
    """Gets additional data required."""
    pScalars = pd.read_csv(scenario['pScalars'], index_col=0)
    elasticity = pScalars.loc['elasticity', 'Value']
    additional_data = {
        'elasticity': elasticity
    }

    pFirmData = pd.read_csv(scenario['pFirmData'], index_col=[0,1,2])
    pFirmData = pFirmData.dropna().drop(columns='Fringe').rename(columns={'ContractLevel': 'value'})
    additional_data['pFirmData'] = pFirmData
    return additional_data

def prepare_data_for_cournot(d, scenario, additional_data, folder):

    demand = d['pDemand'].set_index([ 'zone', 'year', 'season', 'day', 't'])
    price = d['pPrice'].set_index([ 'zone', 'year', 'season', 'day', 't'])
    scenario = estimate_demand_slope_intercept(demand, price, additional_data['elasticity'], folder, scenario)
    supplyfirm = d['pSupplyFirm'].drop(columns=['competition']).set_index(['firm', 'zone', 'year', 'season', 'day', 't'])
    scenario = estimate_hourly_contracting(supplyfirm, additional_data['pFirmData'], folder, scenario)
    return scenario


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


if __name__ == '__main__':
    # pDemandProfile = pd.read_csv('input/pDemandProfile.csv', index_col=[0,1,2])
    # pDemandForecast = pd.read_csv('input/pDemandForecast.csv', index_col=[0,1])
    # pDuration = pd.read_csv('input/pDuration.csv', index_col=[0,1])
    #
    # pDemandData = adjust_demand(
    #     pDemandForecast,
    #     pDemandProfile,
    #     pDuration
    # )
    folder = Path('output') / Path('simulations_run_20250211_165709')
    epm_results = extract_simulation_folder(folder)
    epm_results = process_outputs(epm_results)
    epm_results = process_additional_dataframe(epm_results)
