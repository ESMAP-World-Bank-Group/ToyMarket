from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
from matplotlib.patches import Patch
from itertools import cycle
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from utils import get_duration_curve, filter_dataframe
# import ipywidgets as widgets
# from IPython.display import display


NAME_COLUMNS = {
    'pGenSupply': 'generator',
    'pEnergyByTechDispatch': 'tech',
    'pEnergyByFuelDispatch': 'fuel',
    'pEnergyFullDispatch': 'fuel_firm',
    'pDemand': 'demand',
    'pDispatch': 'attribute',
    'pCostSummary': 'attribute',
    'pCapacityByFuel': 'fuel',
    'pEnergyByFuel': 'fuel'
}

COLORS = 'static/colors_SA.csv'
FUELS = 'static/fuels.csv'
TECHS = 'static/technologies.csv'

def read_plot_specs():
    """
    Read the specifications for the plots from the static files.

    Returns:
    -------
    dict_specs: dict
        Dictionary containing the specifications for the plots
    """

    colors = pd.read_csv(COLORS)
    fuel_mapping = pd.read_csv(FUELS)
    tech_mapping = pd.read_csv(TECHS)

    dict_specs = {
        'colors': colors.set_index('Processing')['Color'].to_dict(),
        'fuel_mapping': fuel_mapping.set_index('EPM_Fuel')['Processing'].to_dict(),
        'tech_mapping': tech_mapping.set_index('EPM_Tech')['Processing'].to_dict()
    }
    return dict_specs


def create_folders_imgs(folder):
    """
    Creating folders for images

    Parameters:
    ----------
    folder: str
        Path to the folder where the images will be saved.

    """
    for p in [path for path in Path(folder).iterdir() if path.is_dir()]:
        if 'simulation' in str(p):
            if not (p / Path('images')).is_dir():
                os.mkdir(p / Path('images'))
    if not (Path(folder) / Path('images')).is_dir():
        os.mkdir(Path(folder) / Path('images'))


def standardize_names(dict_df, key, mapping, column='fuel', sum=True):
    """
    Standardize the names of fuels in the dataframes.

    Only works when dataframes have fuel and value (with numerical values) columns.

    Parameters
    ----------
    dict_df: dict
        Dictionary containing the dataframes
    key: str
        Key of the dictionary to modify
    mapping: dict
        Dictionary mapping the original fuel names to the standardized names
    column: str, optional, default='fuel'
        Name of the column containing the fuels
    """

    if key in dict_df.keys():
        temp = dict_df[key].copy()
        temp[column] = temp[column].replace(mapping)
        if sum:
            temp = temp.groupby([i for i in temp.columns if i != 'value'], observed=False).sum().reset_index()

        new_fuels = [f for f in temp[column].unique() if f not in mapping.values()]
        if new_fuels:
            raise ValueError(f'New fuels found in {key}: {new_fuels}. '
                             f'Add fuels to the mapping in the /static folder and add in the colors.csv file.')

        dict_df[key] = temp.copy()
    else:
        print(f'{key} not found in epm_dict')


def process_for_labels(epm_dict, dict_specs):
    if dict_specs is not None:
        standardize_names(epm_dict, 'pEnergyByFuel', dict_specs['fuel_mapping'], column='fuel')
        standardize_names(epm_dict, 'pEnergyFullDispatch', dict_specs['fuel_mapping'], column='fuel', sum=False)
        standardize_names(epm_dict, 'pEnergyFull', dict_specs['fuel_mapping'], column='fuel', sum=False)
        standardize_names(epm_dict, 'pEnergyByTech', dict_specs['tech_mapping'], column='tech')
        standardize_names(epm_dict, 'pEnergyByFuelDispatch', dict_specs['fuel_mapping'], column='fuel')
    return epm_dict


def make_stacked_bar_subplots(df, filename, dict_colors, figsize=(10,6), selected_zone=None, selected_year=None, column_subplots='year',
                              column_stacked='fuel', column_multiple_bars='scenario',
                              column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None,
                              order_subplots=None, dict_scenarios=None,
                              format_y=lambda y, _: '{:.0f} MW'.format(y),  annotation_format="{:.0f}",
                              order_stacked=None, cap=2, annotate=True,
                              show_total=False, fonttick=12, rotation=0, title=None):
    """
    Subplots with stacked bars. Can be used to explore the evolution of capacity over time and across scenarios.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with results.
    filename : str
        Path to save the figure.
    dict_colors : dict
        Dictionary with color arguments.
    selected_zone : str
        Zone to select.
    column_subplots : str
        Column for choosing the subplots.
    column_stacked : str
        Column name for choosing the column to stack values.
    column_multiple_bars : str
        Column for choosing the type of bars inside a given subplot.
    column_value : str
        Column name for the values to be plotted.
    select_xaxis : list, optional
        Select a subset of subplots (e.g., a number of years).
    dict_grouping : dict, optional
        Dictionary for grouping variables and summing over a given group.
    order_scenarios : list, optional
        Order of scenarios for plotting.
    dict_scenarios : dict, optional
        Dictionary for renaming scenarios.
    format_y : function, optional
        Function for formatting y-axis labels.
    order_stacked : list, optional
        Reordering the variables that will be stacked.
    cap : int, optional
        Under this cap, no annotation will be displayed.
    annotate : bool, optional
        Whether to annotate the bars.
    show_total : bool, optional
        Whether to show the total value on top of each bar.

    Example
    -------
    Stacked bar subplots for capacity (by fuel) evolution:
    filename = Path(RESULTS_FOLDER) / Path('images') / Path('CapacityEvolution.png')
    fuel_grouping = {
        'Battery Storage 4h': 'Battery',
        'Battery Storage 8h': 'Battery',
        'Hydro RoR': 'Hydro',
        'Hydro Storage': 'Hydro'
    }
    scenario_names = {
        'baseline': 'Baseline',
        'HydroHigh': 'High Hydro',
        'DemandHigh': 'High Demand',
        'LowImport_LowThermal': 'LowImport_LowThermal'
    }
    make_stacked_bar_subplots(epm_dict['pCapacityByFuel'], filename, dict_specs['colors'], selected_zone='Liberia',
                              select_xaxis=[2025, 2028, 2030], dict_grouping=fuel_grouping, dict_scenarios=scenario_names,
                              order_scenarios=['Baseline', 'High Hydro', 'High Demand', 'LowImport_LowThermal'],
                              format_y=lambda y, _: '{:.0f} MW'.format(y))

    Stacked bar subplots for reserve evolution:
    filename = Path(RESULTS_FOLDER) / Path('images') / Path('ReserveEvolution.png')
    make_stacked_bar_subplots(epm_dict['pReserveByPlant'], filename, dict_colors=dict_specs['colors'], selected_zone='Liberia',
                              column_subplots='year', column_stacked='fuel', column_multiple_bars='scenario',
                              select_xaxis=[2025, 2028, 2030], dict_grouping=dict_grouping, dict_scenarios=scenario_names,
                              order_scenarios=['Baseline', 'High Hydro', 'High Demand', 'LowImport_LowThermal'],
                              format_y=lambda y, _: '{:.0f} GWh'.format(y),
                              order_stacked=['Hydro', 'Oil'], cap=2)
    """
    if selected_zone is not None:
        df = df[(df['zone'] == selected_zone)]
        df = df.drop(columns=['zone'])

    if selected_year is not None:
        df = df[(df['year'] == selected_year)]
        df = df.drop(columns=['year'])

    if dict_grouping is not None:
        for key, grouping in dict_grouping.items():
            assert key in df.columns, f'Grouping parameter with key {key} is used but {key} is not in the columns.'
            df[key] = df[key].replace(grouping)  # case-specific, according to level of preciseness for dispatch plot

    if column_subplots is not None:
        df = (df.groupby([column_subplots, column_stacked, column_multiple_bars], observed=False)[
                  column_value].sum().reset_index())
        df = df.set_index([column_stacked, column_multiple_bars, column_subplots]).squeeze().unstack(column_subplots)
    else:  # no subplots in this case
        df = (df.groupby([column_stacked, column_multiple_bars], observed=False)[column_value].sum().reset_index())
        df = df.set_index([column_stacked, column_multiple_bars])

    if select_xaxis is not None:
        df = df.loc[:, [i for i in df.columns if i in select_xaxis]]

    if order_subplots is not None:  # order subplots
        new_order = [c for c in order_subplots if c in df.columns] + [c for c in df.columns if
                                                                          c not in order_subplots]
        df = df.loc[:, new_order]


    stacked_bar_subplot(df, column_stacked, filename, dict_colors, figsize=figsize, format_y=format_y, annotation_format=annotation_format,
                        rotation=rotation, order_scenarios=order_scenarios, dict_scenarios=dict_scenarios,
                        order_columns=order_stacked, cap=cap, annotate=annotate, show_total=show_total,
                        fonttick=fonttick, title=title)



def stacked_bar_subplot(df, column_group, filename, dict_colors=None, figsize=(10,6), year_ini=None, order_scenarios=None,
                        order_columns=None,
                        dict_scenarios=None, rotation=0, fonttick=14, legend=True,
                        format_y=lambda y, _: '{:.0f} GW'.format(y), annotation_format="{:.0f}",
                        cap=6, annotate=True, show_total=False, title=None):
    """
    Create a stacked bar subplot from a DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    column_group : str
        Column name to group by for the stacked bars.
    filename : str
        Path to save the plot image. If None, the plot is shown instead.
    dict_colors : dict, optional
        Dictionary mapping column names to colors for the bars. Default is None.
    figsize : tuple, optional
        Size of the figure (width, height). Default is (10, 6).
    year_ini : str, optional
        Initial year to highlight in the plot. Default is None.
    order_scenarios : list, optional
        List of scenario names to order the bars. Default is None.
    order_columns : list, optional
        List of column names to order the stacked bars. Default is None.
    dict_scenarios : dict, optional
        Dictionary mapping scenario names to new names for the plot. Default is None.
    rotation : int, optional
        Rotation angle for x-axis labels. Default is 0.
    fonttick : int, optional
        Font size for tick labels. Default is 14.
    legend : bool, optional
        Whether to display the legend. Default is True.
    format_y : function, optional
        Function to format y-axis labels. Default is a lambda function formatting as '{:.0f} GW'.
    cap : int, optional
        Minimum height of bars to annotate. Default is 6.
    annotate : bool, optional
        Whether to annotate each bar with its height. Default is True.
    show_total : bool, optional
        Whether to show the total value on top of each bar. Default is False.
    Returns
    -------
    None
    """

    list_keys = list(df.columns)
    n_scenario = df.index.get_level_values([i for i in df.index.names if i != column_group][0]).unique()
    num_subplots = int(len(list_keys))
    n_columns = min(3, num_subplots)  # Limit to 3 columns per row
    n_rows = int(np.ceil(num_subplots / n_columns))
    if year_ini is not None:
        width_ratios = [1] + [len(n_scenario)] * (n_columns - 1)
    else:
        width_ratios = [1] * n_columns
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(figsize[0], figsize[1] * n_rows), sharey='all',
                             gridspec_kw={'width_ratios': width_ratios})
    if n_rows * n_columns == 1:  # If only one subplot, `axes` is not an array
        axes = [axes]  # Convert to list to maintain indexing consistency
    else:
        axes = np.array(axes).flatten()  # Ensure it's always a 1D array

    handles, labels = None, None
    for k, key in enumerate(list_keys):
        ax = axes[k]

        try:
            df_temp = df[key].unstack(column_group)

            if key == year_ini:
                df_temp = df_temp.iloc[0, :]
                df_temp = df_temp.to_frame().T
                df_temp.index = ['Initial']
            else:
                if dict_scenarios is not None:  # Renaming scenarios for plots
                    df_temp.index = df_temp.index.map(lambda x: dict_scenarios.get(x, x))
                if order_scenarios is not None:  # Reordering scenarios
                    df_temp = df_temp.loc[[c for c in order_scenarios if c in df_temp.index], :]
                if order_columns is not None:
                    new_order = [c for c in order_columns if c in df_temp.columns] + [c for c in df_temp.columns if
                                                                                      c not in order_columns]
                    df_temp = df_temp.loc[:, new_order]

            df_temp.plot(ax=ax, kind='bar', stacked=True, linewidth=0,
                         color=dict_colors if dict_colors is not None else None)

            # Annotate each bar
            if annotate:
                for container in ax.containers:
                    for bar in container:
                        height = bar.get_height()
                        if height > cap:  # Only annotate bars with a height
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,  # X position: center of the bar
                                bar.get_y() + height / 2,  # Y position: middle of the bar
                                annotation_format.format(height),  # Annotation text (formatted value)
                                ha="center", va="center",  # Center align the text
                                fontsize=10, color="black"  # Font size and color
                            )

            if show_total:
                df_total = df_temp.sum(axis=1)
                for x, y in zip(df_temp.index, df_total.values):
                    # Put the total at the y-position equal to the total
                    ax.text(x, y * (1 + 0.02), f"{y:,.0f}", ha='center', va='bottom', fontsize=10,
                            color='black', fontweight='bold')
                ax.scatter(df_temp.index, df_total, color='black', s=20)

            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
            # put tick label in bold
            ax.tick_params(axis='both', which=u'both', length=0)
            ax.set_xlabel('')

            if len(list_keys) > 1:
                title = key
                if isinstance(key, tuple):
                    title = '{}-{}'.format(key[0], key[1])
                ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)
            else:
                if title is not None:
                    if isinstance(title, tuple):
                        title = '{}-{}'.format(title[0], title[1])
                    ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)

            if k == 0:
                handles, labels = ax.get_legend_handles_labels()
                labels = [l.replace('_', ' ') for l in labels]
                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
            if k % n_columns != 0:
                ax.set_ylabel('')
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.get_legend().remove()

            # Add a horizontal line at 0
            ax.axhline(0, color='black', linewidth=0.5)

        except IndexError:
            ax.axis('off')

        if legend:
            fig.legend(handles[::-1], labels[::-1], loc='center left', frameon=False, ncol=1,
                       bbox_to_anchor=(1, 0.5))

    # Hide unused subplots
    for j in range(k + 1, len(axes)):
        fig.delaxes(axes[j])

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()




def make_multiple_lines_subplots(df, filename, dict_colors, selected_zone=None, selected_year=None, column_subplots='scenario',
                              column_multiple_lines='competition', column_xaxis='t',
                              column_value='value', select_subplots=None, order_scenarios=None,
                              dict_scenarios=None, figsize=(10,6),
                              format_y=lambda y, _: '{:.0f} MW'.format(y),  annotation_format="{:.0f}",
                              order_stacked=None, max_ticks=10, annotate=True,
                              show_total=False, fonttick=12, rotation=0, title=None):
    """
    Subplots with stacked bars. Can be used to explore the evolution of capacity over time and across scenarios.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with results.
    filename : str
        Path to save the figure.
    dict_colors : dict
        Dictionary with color arguments.
    selected_zone : str
        Zone to select.
    column_xaxis : str
        Column for choosing the subplots.
    column_stacked : str
        Column name for choosing the column to stack values.
    column_multiple_bars : str
        Column for choosing the type of bars inside a given subplot.
    column_value : str
        Column name for the values to be plotted.
    select_xaxis : list, optional
        Select a subset of subplots (e.g., a number of years).
    dict_grouping : dict, optional
        Dictionary for grouping variables and summing over a given group.
    order_scenarios : list, optional
        Order of scenarios for plotting.
    dict_scenarios : dict, optional
        Dictionary for renaming scenarios.
    format_y : function, optional
        Function for formatting y-axis labels.
    order_stacked : list, optional
        Reordering the variables that will be stacked.
    cap : int, optional
        Under this cap, no annotation will be displayed.
    annotate : bool, optional
        Whether to annotate the bars.
    show_total : bool, optional
        Whether to show the total value on top of each bar.

    Example
    -------

    """
    if selected_zone is not None:
        df = df[(df['zone'] == selected_zone)]
        df = df.drop(columns=['zone'])

    if selected_year is not None:
        df = df[(df['year'] == selected_year)]
        df = df.drop(columns=['year'])

    if column_subplots is not None:
        df = (df.groupby([column_subplots, column_multiple_lines, column_xaxis], observed=False)[
                  column_value].mean().reset_index())
        df = df.set_index([column_multiple_lines, column_xaxis, column_subplots]).squeeze().unstack(column_subplots)
    else:  # no subplots in this case
        df = (df.groupby([column_multiple_lines, column_xaxis], observed=False)[column_value].mean().reset_index())
        df = df.set_index([column_multiple_lines, column_xaxis])

    # TODO: change select_axis name
    if select_subplots is not None:
        df = df.loc[:, [i for i in df.columns if i in select_subplots]]

    multiple_lines_subplot(df, column_multiple_lines, filename, figsize=figsize, dict_colors=dict_colors,  format_y=format_y,
                           annotation_format=annotation_format,  rotation=rotation, order_scenarios=order_scenarios, dict_scenarios=dict_scenarios,
                           order_columns=order_stacked, max_ticks=max_ticks, annotate=annotate, show_total=show_total,
                           fonttick=fonttick, title=title)


def multiple_lines_subplot(df, column_multiple_lines, filename, figsize=(10,6), dict_colors=None, order_scenarios=None,
                            order_columns=None, dict_scenarios=None, rotation=0, fonttick=14, legend=True,
                           format_y=lambda y, _: '{:.0f} GW'.format(y), annotation_format="{:.0f}",
                           max_ticks=10, annotate=True, show_total=False, title=None):
    """
    Create a stacked bar subplot from a DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    column_group : str
        Column name to group by for the stacked bars.
    filename : str
        Path to save the plot image. If None, the plot is shown instead.
    dict_colors : dict, optional
        Dictionary mapping column names to colors for the bars. Default is None.
    figsize : tuple, optional
        Size of the figure (width, height). Default is (10, 6).
    year_ini : str, optional
        Initial year to highlight in the plot. Default is None.
    order_scenarios : list, optional
        List of scenario names to order the bars. Default is None.
    order_columns : list, optional
        List of column names to order the stacked bars. Default is None.
    dict_scenarios : dict, optional
        Dictionary mapping scenario names to new names for the plot. Default is None.
    rotation : int, optional
        Rotation angle for x-axis labels. Default is 0.
    fonttick : int, optional
        Font size for tick labels. Default is 14.
    legend : bool, optional
        Whether to display the legend. Default is True.
    format_y : function, optional
        Function to format y-axis labels. Default is a lambda function formatting as '{:.0f} GW'.
    cap : int, optional
        Minimum height of bars to annotate. Default is 6.
    annotate : bool, optional
        Whether to annotate each bar with its height. Default is True.
    show_total : bool, optional
        Whether to show the total value on top of each bar. Default is False.
    Returns
    -------
    None
    """

    list_keys = list(df.columns)
    n_scenario = df.index.get_level_values([i for i in df.index.names if i != column_multiple_lines][0]).unique()
    num_subplots = int(len(list_keys))
    n_columns = min(3, num_subplots)  # Limit to 3 columns per row
    n_rows = int(np.ceil(num_subplots / n_columns))
    width_ratios = [1] * n_columns
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(figsize[0], figsize[1] * n_rows), sharey='all',
                             gridspec_kw={'width_ratios': width_ratios})

    if n_rows * n_columns == 1:  # If only one subplot, `axes` is not an array
        axes = [axes]  # Convert to list to maintain indexing consistency
    else:
        axes = np.array(axes).flatten()  # Ensure it's always a 1D array


    handles, labels = None, None
    for k, key in enumerate(list_keys):
        ax = axes[k]

        try:
            df_temp = df[key].unstack(column_multiple_lines)
            if dict_scenarios is not None:  # Renaming scenarios for plots
                df_temp.index = df_temp.index.map(lambda x: dict_scenarios.get(x, x))
            if order_scenarios is not None:  # Reordering scenarios
                df_temp = df_temp.loc[[c for c in order_scenarios if c in df_temp.index], :]
            if order_columns is not None:
                new_order = [c for c in order_columns if c in df_temp.columns] + [c for c in df_temp.columns if
                                                                                  c not in order_columns]
                df_temp = df_temp.loc[:, new_order]

            df_temp.plot(ax=ax, kind='line',
                         color=dict_colors if dict_colors is not None else None)

            num_xticks = min(len(df_temp.index), max_ticks)  # Set a reasonable max number of ticks
            xticks_positions = np.linspace(0, len(df_temp.index) - 1, num_xticks, dtype=int)

            ax.set_xticks(xticks_positions)  # Set tick positions
            ax.set_xticklabels(df_temp.index[xticks_positions], rotation=rotation)

            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
            # put tick label in bold
            ax.tick_params(axis='both', which=u'both', length=0)
            ax.set_xlabel('')

            if len(list_keys) > 1:
                title = key
                if isinstance(key, tuple):
                    title = '{}-{}'.format(key[0], key[1])
                ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)
            else:
                if title is not None:
                    if isinstance(title, tuple):
                        title = '{}-{}'.format(title[0], title[1])
                    ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)

            if k == 0:
                handles, labels = ax.get_legend_handles_labels()
                labels = [l.replace('_', ' ') for l in labels]
                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
            if k % n_columns != 0:
                ax.set_ylabel('')
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.get_legend().remove()

            # Add a horizontal line at 0
            ax.axhline(0, color='black', linewidth=0.5)

        except IndexError:
            ax.axis('off')

        if legend:
            fig.legend(handles[::-1], labels[::-1], loc='center left', frameon=False, ncol=1,
                       bbox_to_anchor=(1, 0.5))

    # Hide unused subplots
    for j in range(k + 1, len(axes)):
        fig.delaxes(axes[j])

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()



def remove_na_values(df):
    """Removes na values from a dataframe, to avoind unnecessary labels in plots."""
    df = df.where((df > 1e-6) | (df < -1e-6),
                  np.nan)
    df = df.dropna(axis=1, how='all')
    return df


def select_time_period(df, select_time):
    """Select a specific time period in a dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        Columns contain season and day
    select_time: dict
        For each key, specifies a subset of the dataframe

    Returns
    -------
    pd.DataFrame: Dataframe with the selected time period
    str: String with the selected time period
    """
    temp = ''
    if 'season' in select_time.keys():
        df = df.loc[df.season.isin(select_time['season'])]
        temp += '_'.join(select_time['season'])
    if 'day' in select_time.keys():
        df = df.loc[df.day.isin(select_time['day'])]
        temp += '_'.join(select_time['day'])
    return df, temp


def clean_dataframe(df, zone, year, scenario, competition, column_stacked=None, fuel_grouping=None, select_time=None):
    """
    Transforms a dataframe from the results GDX into a dataframe with season, day, and time as the index, and format ready for plot.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the results.
    zone : str
        The zone to filter the data for.
    year : int
        The year to filter the data for.
    scenario : str
        The scenario to filter the data for.
    column_stacked : str
        Column to use for stacking values in the transformed dataframe.
    fuel_grouping : dict, optional
        A dictionary mapping fuels to their respective groups and to sum values over those groups.
    select_time : dict or None, optional
        Specific time filter to apply (e.g., "summer").

    Returns
    -------
    pd.DataFrame
        A transformed dataframe with multi-level index (season, day, time).

    Example
    -------
    df = epm_dict['FuelDispatch']
    column_stacked = 'fuel'
    select_time = {'season': ['m1'], 'day': ['d21', 'd22', 'd23', 'd24', 'd25', 'd26', 'd27', 'd28', 'd29', 'd30']}
    df = clean_dataframe(df, zone='Liberia', year=2025, scenario='Baseline', column_stacked='fuel', fuel_grouping=None, select_time=select_time)
    """
    if 'zone' in df.columns:
        df = df[(df['zone'] == zone) & (df['year'] == year) & (df['scenario'] == scenario) & (df['competition'] == competition)]
        df = df.drop(columns=['zone', 'year', 'scenario', 'competition'])
    else:
        df = df[(df['year'] == year) & (df['scenario'] == scenario) & (df['competition'] == competition)]
        df = df.drop(columns=['year', 'scenario', 'competition'])

    if column_stacked == 'fuel':
        if fuel_grouping is not None:
            df['fuel'] = df['fuel'].replace(
                fuel_grouping)  # case-specific, according to level of preciseness for dispatch plot

    if column_stacked is not None:
        df = (df.groupby(['season', 'day', 't', column_stacked], observed=False).sum().reset_index())
        # if df.dtypes['t'] == 'object':  # column t is a string, and should be reordered to match hourly order
        #
        #     df['t_numeric'] = df['t'].str.extract(r'(\d+)').astype(int)
        #     df = df.sort_values(by=['season', 'day', 't_numeric'])
        #     df = df.drop(columns=['t_numeric'])  # Remove helper column if not needed

    if select_time is not None:
        df, temp = select_time_period(df, select_time)
    else:
        temp = None

    if column_stacked is not None:
        df = df.set_index(['season', 'day', 't', column_stacked]).unstack(column_stacked)
    else:
        df = df.set_index(['season', 'day', 't'])
    return df, temp


def make_complete_fuel_dispatch_plot(dfs_area, dfs_line, dict_colors, zone, year, scenario, competition,
                                     name_columns_spec=None, filename=None, fuel_grouping=None, select_time=None, reorder_dispatch=None,
                                     legend_loc='bottom', figsize=(10, 6), interactive=False):
    """
    Generates and saves a fuel dispatch plot, including only generation plants.

    Parameters
    ----------
    dfs_area : dict
        Dictionary containing dataframes for area plots.
    dfs_line : dict
        Dictionary containing dataframes for line plots.
    graph_folder : str
        Path to the folder where the plot will be saved.
    dict_colors : dict
        Dictionary mapping fuel types to colors.
    fuel_grouping : dict
        Mapping to create aggregate fuel categories, e.g.,
        {'Battery Storage 4h': 'Battery Storage'}.
    select_time : dict
        Time selection parameters for filtering the data.
    dfs_line_2 : dict, optional
        Optional dictionary containing dataframes for a secondary line plot.

    Returns
    -------
    None

    Example
    -------
    Generate and save a fuel dispatch plot:
    dfs_to_plot_area = {
        'pFuelDispatch': epm_dict['pFuelDispatch'],
        'pCurtailedVRET': epm_dict['pCurtailedVRET'],
        'pDispatch': subset_dispatch
    }
    subset_demand = epm_dict['pDispatch'].loc[epm_dict['pDispatch'].attribute.isin(['Demand'])]
    dfs_to_plot_line = {
        'pDispatch': subset_demand
    }
    fuel_grouping = {
        'Battery Storage 4h': 'Battery Discharge',
        'Battery Storage 8h': 'Battery Discharge',
        'Battery Storage 2h': 'Battery Discharge',
        'Battery Storage 3h': 'Battery Discharge',
    }
    make_complete_fuel_dispatch_plot(dfs_to_plot_area, dfs_to_plot_line, folder_results / Path('images'), dict_specs['colors'],
                                 zone='Liberia', year=2030, scenario=scenario, fuel_grouping=fuel_grouping,
                                 select_time=select_time, reorder_dispatch=['MtCoffee', 'Oil', 'Solar'], season=False)
    """
    # TODO: Add ax2 to show other data. For example prices would be interesting to show in the same plot.

    if name_columns_spec is not None:
        NAME_COLUMNS.update(name_columns_spec)
    tmp_concat_area = []
    for key in dfs_area:
        df = dfs_area[key]
        column_stacked = NAME_COLUMNS[key]
        df, temp = clean_dataframe(df, zone, year, scenario, competition, column_stacked, fuel_grouping=fuel_grouping,
                                   select_time=select_time)
        tmp_concat_area.append(df)

    tmp_concat_line = []
    for key in dfs_line:
        df = dfs_line[key]

        df, temp = clean_dataframe(df, zone, year, scenario, competition, None, fuel_grouping=fuel_grouping,
                                   select_time=select_time)
        column_stacked = NAME_COLUMNS[key]
        df = df.rename(columns={'value': column_stacked})
        tmp_concat_line.append(df)

    df_tot_area = pd.concat(tmp_concat_area, axis=1)
    df_tot_area = df_tot_area.droplevel(0, axis=1)
    df_tot_area = remove_na_values(df_tot_area)

    if len(tmp_concat_line) > 0:
        df_tot_line = pd.concat(tmp_concat_line, axis=1)
        df_tot_line = remove_na_values(df_tot_line)

        # Making sure both dataframes have the same ordered index
        df_tot_line = df_tot_line.reindex(df_tot_area.index)
    else:
        df_tot_line = None


    if reorder_dispatch is not None:
        new_order = [col for col in reorder_dispatch if col in df_tot_area.columns] + [col for col in
                                                                                       df_tot_area.columns if
                                                                                       col not in reorder_dispatch]
        df_tot_area = df_tot_area[new_order]

    # if select_time is None:
    #     temp = 'all'
    # temp = f'{year}_{temp}'
    # if filename is not None:
    #     filename = filename.split('.')[0] + f'_{temp}.png'

    if not interactive:
        dispatch_plot(df_tot_area, filename, df_line=df_tot_line, dict_colors=dict_colors, legend_loc=legend_loc,
                      figsize=figsize)
    else:
        # TODO: not working currently
        dispatch_plot_interactive(df_area=df_tot_area, df_line=df_tot_line, dict_colors=dict_colors)


def dispatch_plot(df_area=None, filename=None, dict_colors=None, df_line=None, figsize=(10, 6), legend_loc='bottom',
                  ymin=0):
    """
    Generate and display or save a dispatch plot with area and line plots.


    Parameters
    ----------
    df_area : pandas.DataFrame, optional
        DataFrame containing data for the area plot. If provided, the area plot will be stacked.
    filename : str, optional
        Path to save the plot image. If not provided, the plot will be displayed.
    dict_colors : dict, optional
        Dictionary mapping column names to colors for the plot.
    df_line : pandas.DataFrame, optional
        DataFrame containing data for the line plot. If provided, the line plot will be overlaid on the area plot.
    figsize : tuple, default (10, 6)
        Size of the figure in inches.
    legend_loc : str, default 'bottom'
        Location of the legend. Options are 'bottom' or 'right'.
    ymin : int or float, default 0
        Minimum value for the y-axis.
    Raises
    ------
    ValueError
        If neither `df_area` nor `df_line` is provided.
    Notes
    -----
    The function will raise an assertion error if `df_area` and `df_line` are provided but do not share the same index.

    Examples
    --------

    """

    fig, ax = plt.subplots(figsize=figsize)

    if dict_colors is None:
        palette = sns.color_palette("viridis", n_colors=len(df_area.columns))
        final_colors = {col: color for col, color in zip(df_area.columns, palette)}
    else:
        # Start with the provided colors.
        final_colors = dict_colors.copy()
        # Identify missing columns.
        missing = [col for col in df_area.columns if col not in final_colors]
        if missing:
            palette = sns.color_palette("Set2", n_colors=len(missing))
            for col, color in zip(missing, palette):
                final_colors[col] = color

    if df_area is not None:
        df_area.plot.area(ax=ax, stacked=True, color=final_colors, linewidth=0)
        pd_index = df_area.index
    if df_line is not None:
        if df_area is not None:
            assert df_area.index.equals(
                df_line.index), 'Dataframes used for area and line do not share the same index. Update the input dataframes.'
        df_line.plot(ax=ax, color=dict_colors)
        pd_index = df_line.index

    if (df_area is None) and (df_line is None):
        raise ValueError('No dataframes provided for the plot. Please provide at least one dataframe.')

    format_dispatch_ax(ax, pd_index)

    # Add axis labels and title
    ax.set_ylabel('Generation (MWh)', fontweight='bold')
    # set ymin to 0
    if ymin is not None:
        ax.set_ylim(bottom=ymin)

    # Add legend bottom center
    legend_ncol = max(1, min(len(df_area.columns), 5))

    if legend_loc == 'bottom':
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=legend_ncol, frameon=False)
    elif legend_loc == 'right':
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), ncol=1, frameon=False)

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def dispatch_plot_interactive(df_area=None, df_line=None, dict_colors=None, legend_loc='bottom', ymin=0):
    """
    Generate an interactive Plotly dispatch plot.
    """

    fig = go.Figure()

    # **Stacked Area Plots (Ensuring Proper Hover & Order)**
    if df_area is not None:
        for col in df_area.columns:
            fig.add_trace(go.Scatter(
                x=df_area.index, y=df_area[col], mode='lines', fill='tonexty',
                name=col, stackgroup='one', line=dict(width=0),  # No borders
                marker=dict(color=dict_colors.get(col, None)),
                hoverinfo='x+y+name'  # Show value per area individually
            ))

    # **Line Plots (Overlay)**
    if df_line is not None:
        for col in df_line.columns:
            fig.add_trace(go.Scatter(
                x=df_line.index, y=df_line[col], mode='lines', name=col,
                line=dict(color=dict_colors.get(col, 'black'), width=2, dash='dash'),
                hoverinfo='x+y+name'  # Show demand separately
            ))

    # **Extract Time Structure**
    days = df_area.index.get_level_values('day').unique()
    seasons = df_area.index.get_level_values('season').unique()
    total_days = len(seasons) * len(days)
    y_max = df_area.max().max() * 1.05  # Extend ymax slightly for labels

    # **Vertical Lines for Day Separators**
    for d, day in enumerate(days):
        fig.add_vline(x=f"{day} - 0h", line_width=1, line_dash="dash", line_color="slategrey")

        # **Add day labels (d1, d2, ...) at 12h**
        fig.add_annotation(
            x=f"{day} - 12h", y=y_max,
            text=day, showarrow=False, font=dict(size=10, color="black"),
            yshift=5
        )

    # **Add Season Labels**
    season_x_positions = [f"d{d + 1} - 12h" for d in range(len(days))]
    fig.update_xaxes(
        tickmode="array",
        tickvals=season_x_positions,
        ticktext=[s for s in seasons],
        title_text="Time"
    )

    # **Legend Positioning**
    legend_x, legend_y = (0.5, -0.2) if legend_loc == 'bottom' else (1.1, 0.5)

    # **Update Layout**
    fig.update_layout(
        title="Dispatch Plot",
        xaxis_title="Time",
        yaxis_title="Generation (MWh)",
        legend=dict(x=legend_x, y=legend_y, orientation="h" if legend_loc == 'bottom' else "v"),
        template="plotly_white",
        hovermode="x unified",  # Keep x-axis hover synchronized
        yaxis=dict(range=[ymin, y_max]),  # Adjust based on max value
    )

    fig.show()


def format_dispatch_ax(ax, pd_index):
    # Adding the representative days and seasons
    n_rep_days = len(pd_index.get_level_values('day').unique())
    dispatch_seasons = pd_index.get_level_values('season').unique()
    total_days = len(dispatch_seasons) * n_rep_days
    y_max = ax.get_ylim()[1]

    for d in range(total_days):
        x_d = 24 * d

        # Add vertical lines to separate days
        is_end_of_season = d % n_rep_days == 0
        linestyle = '-' if is_end_of_season else '--'
        ax.axvline(x=x_d, color='slategrey', linestyle=linestyle, linewidth=0.8)

        # Add day labels (d1, d2, ...)
        ax.text(
            x=x_d + 12,  # Center of the day (24 hours per day)
            y=y_max * 0.99,
            s=f'd{(d % n_rep_days) + 1}',
            ha='center',
            fontsize=7
        )

    # Add season labels
    season_x_positions = [24 * n_rep_days * s + 12 * n_rep_days for s in range(len(dispatch_seasons))]
    ax.set_xticks(season_x_positions)
    ax.set_xticklabels(dispatch_seasons, fontsize=8)
    ax.set_xlim(left=0, right=24 * total_days)
    ax.set_xlabel('Time')
    # Remove grid
    ax.grid(False)
    # Remove top spine to let days appear
    ax.spines['top'].set_visible(False)


def subplot_scatter(df, column_xaxis, column_yaxis, column_color, color_dict, figsize=(12, 8),
                    ymax=None, xmax=None, title='', legend=None, filename=None,
                    size_scale=None, annotate_thresh=None, column_annotate=None, subplot_column=None, column_scale=None):
    """
    Creates scatter plots with points colored based on the values in a specific column.
    Supports optional subplots based on a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    column_xaxis : str
        Column name for x-axis values.
    column_yaxis : str
        Column name for y-axis values.
    column_color : str
        Column name for categorical values determining color.
    color_dict : dict
        Dictionary mapping values in column_color to specific colors.
    ymax : float, optional
        Maximum y-axis value.
    xmax : float, optional
        Maximum x-axis value.
    title : str, optional
        Title of the plot.
    legend : str, optional
        Title for the legend.
    filename : str, optional
        File name to save the plot. If None, the plot is displayed.
    size_scale : float, optional
        Scaling factor for point sizes.
    annotate_thresh : float, optional
        Threshold for annotating points with generator names.
    subplot_column : str, optional
        Column name to split the data into subplots.

    Returns
    -------
    None
        Displays the scatter plots.
    """
    # If subplots are required
    if subplot_column is not None:
        unique_values = df[subplot_column].unique()
        n_subplots = len(unique_values)
        ncols = min(3, n_subplots)  # Limit to 3 columns per row
        nrows = int(np.ceil(n_subplots / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows),
                                 sharex=True, sharey=True)
        axes = np.array(axes).flatten()  # Ensure axes is an iterable 1D array

        for i, val in enumerate(unique_values):
            ax = axes[i]
            subset_df = df[df[subplot_column] == val]

            scatter_plot_on_ax(ax, subset_df, column_xaxis, column_yaxis, column_color, color_dict,
                               ymax, xmax, title=f"{title} - {subplot_column}: {val}",
                               legend=False, size_scale=size_scale, column_scale=column_scale,
                               annotate_thresh=annotate_thresh, column_annotate=column_annotate)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        handles, labels = scatter_plot_on_ax(None, df, column_xaxis, column_yaxis, column_color, color_dict,
                                             legend=True, size_scale=size_scale, column_scale=column_scale)

        fig.legend(handles, labels, title=legend, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        # plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit the legend

        plt.tight_layout()
    else:
        # If no subplots, plot normally
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter_plot_on_ax(ax, df, column_xaxis, column_yaxis, column_color, color_dict,
                           ymax, xmax, title=title, legend=legend,
                           size_scale=size_scale, column_scale=column_scale,
                           annotate_thresh=annotate_thresh, column_annotate=column_annotate)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def scatter_plot_on_ax(ax, df, column_xaxis, column_yaxis, column_color, color_dict,
                       ymax=None, xmax=None, title='', legend=None,
                       size_scale=None, column_scale=None, annotate_thresh=None, column_annotate=None):
    """
    Helper function to create a scatter plot on a given matplotlib Axes.
    """
    unique_values = df[column_color].unique()
    handles = []
    labels = []

    if ax is not None:
        for val in unique_values:
            if val not in color_dict:
                raise ValueError(f"No color specified for value '{val}' in {column_color}")

        color_dict = {val: color_dict[val] for val in unique_values}

        # Determine sizes of points
        sizes = 50
        if size_scale is not None:
            assert column_scale is not None, "Size scale is required but column to scale not provided."
            sizes = df[column_scale] * size_scale

        # Plot each category separately
        for value, color in color_dict.items():
            subset = df[df[column_color] == value]
            scatter = ax.scatter(subset[column_xaxis], subset[column_yaxis],
                                 label=value, color=color, alpha=0.7,
                                 s=sizes[subset.index] if size_scale else sizes)

            # Annotate points above a certain threshold
            if annotate_thresh is not None:
                assert column_annotate is not None, 'Annotate is required but no column is provided for the threshold used in the annotation.'
                for i, txt in enumerate(subset['generator']):
                    if abs(subset[column_annotate].iloc[i]) > annotate_thresh:
                        x_value, y_value = subset[column_xaxis].iloc[i], subset[column_yaxis].iloc[i]
                        ax.annotate(
                            txt,
                            (x_value, y_value),  # Point location
                            xytext=(5, 10),  # Offset in points (x, y)
                            textcoords='offset points',  # Use an offset from the data point
                            fontsize=9,
                            color='black',
                            ha='left'
                        )
                        # ax.annotate(txt, (subset[column_xaxis].iloc[i], subset[column_yaxis].iloc[i]), color='black')

        if ymax is not None:
            ax.set_ylim(0, ymax)

        if xmax is not None:
            ax.set_xlim(0, xmax)

        ax.set_xlabel(column_xaxis)
        ax.set_ylabel(column_yaxis)
        ax.set_title(title)

        # # Remove legend from each subplot to avoid redundancy
        # if legend is not None:
        #     ax.legend(title=legend, frameon=False)

        ax.grid(True, linestyle='--', alpha=0.5)

    for value in unique_values:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=value,
                                  markerfacecolor=color_dict[value], markersize=8))
        labels.append(value)

    if legend and ax is not None:
        ax.legend(handles, labels, title=legend, frameon=False)

    return handles, labels


def subplot_pie_new(df, index, dict_colors=None, subplot_column=None, share_column='value', title='', figsize=(16, 4),
                    percent_cap=1, filename=None, rename=None, bbox_to_anchor=(0.5, -0.1), loc='lower center',
                    font_title=16, annotation_font=10):
    """
    Creates pie charts for data grouped by a column, or a single pie chart if no grouping is specified.

    Parameters:
    ----------
    df: pd.DataFrame
        DataFrame containing the data
    index: str
        Column to use for the pie chart
    dict_colors: dict
        Dictionary mapping the index values to colors
    subplot_column: str, optional
        Column to use for subplots. If None, a single pie chart is created.
    title: str, optional
        Title of the plot
    figsize: tuple, optional, default=(16, 4)
        Size of the figure
    percent_cap: float, optional, default=1
        Minimum percentage to show in the pie chart
    filename: str, optional
        Path to save the plot
    bbox_to_anchor: tuple
        Position of the legend compared to the figure
    loc: str
        Localization of the legend
    """
    if rename is not None:
        df[index] = df[index].replace(rename)

    if dict_colors is None:
        unique_labels = df[index].unique()
        color_cycle = cycle(plt.cm.get_cmap("tab10").colors)  # Choose a colormap
        dict_colors = {label: next(color_cycle) for label in unique_labels}

    if subplot_column is not None:
        # Group by the column for subplots
        groups = df.groupby(subplot_column)

        # Calculate the number of subplots
        num_subplots = len(groups)
        ncols = min(3, num_subplots)  # Limit to 3 columns per row
        nrows = int(np.ceil(num_subplots / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0], figsize[1] * nrows))
        axes = np.array(axes).flatten()  # Ensure axes is iterable 1D array

        all_labels = set()  # Collect all labels for the combined legend
        for ax, (name, group) in zip(axes, groups):
            colors = [dict_colors[f] for f in group[index]]
            handles, labels = plot_pie_on_ax(ax, group, index, share_column, percent_cap, colors,
                                             title=f"{title} - {subplot_column}: {name}", fontsize=font_title, annotation_size=annotation_font)
            all_labels.update(group[index])  # Collect unique labels

        # Hide unused subplots
        for j in range(len(groups), len(axes)):
            fig.delaxes(axes[j])

        # Create a shared legend below the graphs
        all_labels = sorted(all_labels)  # Sort labels for consistency
        handles = [plt.Line2D([0], [0], marker='o', color=dict_colors[label], linestyle='', markersize=10)
                   for label in all_labels]
        fig.legend(
            handles,
            all_labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=1,  # Adjust number of columns based on subplots
            frameon=False, fontsize=16
        )

        # Add title for the whole figure
        fig.suptitle(title, fontsize=16)

    else:  # Create a single pie chart if no subplot column is specified
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [dict_colors[f] for f in df[index]]
        handles, labels = plot_pie_on_ax(ax, df, index, share_column, percent_cap, colors, title, fontsize=font_title,
                                         annotation_size=annotation_font)

        fig.legend(
            handles,
            labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=1,  # Adjust number of columns based on subplots
            frameon=False, fontsize=16
        )

    # Save the figure if filename is provided
    plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pie_on_ax(ax, df, index, share_column, percent_cap, colors, title, radius=None, annotation_size=8, fontsize=16):
    """Pie plot on a single axis."""
    if radius is not None:
        df.plot.pie(
            ax=ax,
            y=share_column,
            autopct=lambda p: f'{p:.0f}%' if p > percent_cap else '',
            startangle=140,
            legend=False,
            colors=colors,
            labels=None,
            radius=radius
        )
    else:
        df.plot.pie(
            ax=ax,
            y=share_column,
            autopct=lambda p: f'{p:.0f}%' if p > percent_cap else '',
            startangle=140,
            legend=False,
            colors=colors,
            labels=None
        )
    ax.set_ylabel('')
    ax.set_title(title, fontsize=fontsize)

    # Adjust annotation font sizes
    for text in ax.texts:
        if text.get_text().endswith('%'):  # Check if the text is a percentage annotation
            text.set_fontsize(annotation_size)

    # Generate legend handles and labels manually
    handles = [Patch(facecolor=color, label=label) for color, label in zip(colors, df[index])]
    labels = list(df[index])
    return handles, labels


def make_automatic_plots(epm_results, zone, dict_specs, folder, scenarios_to_remove = None):

    # Price plot
    if not (Path(folder) / Path('images')).is_dir():
        os.mkdir(Path(folder) / Path('images'))

    df = epm_results['pPrice'].copy()
    df["hour"] = df["t"].str.extract(r"(\d+)").astype(int).astype(str) + "h"

    df["day_hour"] = df["day"].astype(str) + " - " + df["hour"]
    df["season_day_hour"] = df["season"].astype(str) + " - " + df["day"].astype(str) + " - " + df["hour"]

    if 'baseline' in df.scenario.unique():
        df.loc[(df.scenario == 'baseline') & (df.competition == 'Least-cost'), 'scenario'] = 'Least-cost'
    elif 'Baseline' in df.scenario.unique():
        df.loc[(df.scenario == 'Baseline') & (df.competition == 'Least-cost'), 'scenario'] = 'Least-cost'
    else:
        df.loc[(df.scenario ==df.scenario.unique()[0]) & (df.competition == 'Least-cost'), 'scenario'] = 'Least-cost'

    df = df[~((df["scenario"] != "Least-cost") & (df["competition"] == "Least-cost"))]

    df = df.drop(columns=['competition'])

    if scenarios_to_remove is not None:
        df = df[~df["scenario"].isin(scenarios_to_remove)]

    filename = Path(folder) / Path('images') / Path('prices.png')

    make_multiple_lines_subplots(df, filename, dict_colors=None, figsize=(6, 6), column_subplots=None,
                                 column_xaxis='season_day_hour',
                                 column_value='value', column_multiple_lines='scenario',
                                 format_y=lambda y, _: '{:.0f} US$ /MWh'.format(y), annotation_format="{:.0f}",
                                 max_ticks=10, rotation=45)

    # Price duration curves plots

    df = epm_results['pPrice'].copy()
    scenarios = pd.read_csv(Path(folder) / Path('simulation_scenarios.csv'), index_col=0)
    if 'baseline' in scenarios.columns :
        pDurationpath = scenarios.loc['pDuration', 'baseline']
    elif 'Baseline' in scenarios.columns :
        pDurationpath = scenarios.loc['pDuration', 'Baseline']
    else:
        pDurationpath = scenarios.loc['pDuration', scenarios.columns [0]]

    pDuration = pd.read_csv(pDurationpath, index_col=[0, 1])

    if 'baseline' in df.scenario.unique():
        df.loc[(df.scenario == 'baseline') & (df.competition == 'Least-cost'), 'scenario'] = 'Least-cost'
    elif 'Baseline' in df.scenario.unique():
        df.loc[(df.scenario == 'Baseline') & (df.competition == 'Least-cost'), 'scenario'] = 'Least-cost'
    else:
        df.loc[(df.scenario == df.scenario.unique()[0]) & (df.competition == 'Least-cost'), 'scenario'] = 'Least-cost'

    df = df[~((df["scenario"] != "Least-cost") & (df["competition"] == "Least-cost"))]

    if scenarios_to_remove is not None:
        df = df[~df["scenario"].isin(scenarios_to_remove)]

    expanded_price = get_duration_curve(df, pDuration)

    filename = Path(folder) / Path('images') / Path('price_duration_curve.png')

    make_multiple_lines_subplots(expanded_price, filename, None, column_subplots=None, column_xaxis='hour',
                                 column_value='value', column_multiple_lines='scenario',
                                 format_y=lambda y, _: '{:.0f} US$ /MWh'.format(y), annotation_format="{:.0f}",
                                 max_ticks=10, rotation=45, figsize=(6, 6))

    # Energy by fuel subplot

    df = epm_results['pEnergyByFuel'].copy()
    df['value'] = df['value'] / 1e6

    if scenarios_to_remove is not None:
        df = df[~df["scenario"].isin(scenarios_to_remove)]

    filename = Path(folder) / Path('images') / Path('energy_subplots.png')
    make_stacked_bar_subplots(df, filename, dict_specs['colors'], column_stacked='fuel', column_subplots='scenario',
                              column_value='value', column_multiple_bars='competition',
                              format_y=lambda y, _: '{:.0f} TWh'.format(y), annotation_format="{:.0f}", cap=5,
                              show_total=True)

    # Energy by fuel plot

    df = epm_results['pEnergyByFuel'].copy()
    df['value'] = df['value'] / 1e6

    df.loc[(df.scenario == 'baseline') & (df.competition == 'Least-cost'), 'scenario'] = 'Least-cost'

    df = df[~((df["scenario"] != "Least-cost") & (df["competition"] == "Least-cost"))]

    df = df.drop(columns=['competition'])

    filename = Path(folder) / Path('images') / Path('energy.png')
    make_stacked_bar_subplots(df, filename, dict_specs['colors'], column_stacked='fuel', column_subplots=None,
                              column_value='value', column_multiple_bars='scenario',
                              format_y=lambda y, _: '{:.0f} TWh'.format(y), annotation_format="{:.0f}", cap=5,
                              show_total=True)

    # Dispatch plots
    if not (Path(folder) / Path('images') / Path('dispatch')).is_dir():
        os.mkdir(Path(folder) / Path('images') / Path('dispatch'))

    dispatch_df = epm_results['pEnergyByFuelDispatch'].copy()
    dispatch_df['fuel'] = dispatch_df['fuel'].replace({'Uranium': 'Nuclear'})

    dfs_to_plot_area = {
        'pEnergyByFuelDispatch': dispatch_df,
        'pDispatch': filter_dataframe(epm_results['pDispatch'], {'attribute': ['Unmet demand']})
    }

    dfs_to_plot_line = {
        'pDemand': epm_results['pDemand']
    }

    for selected_scenario in dispatch_df.scenario.unique():
        for competition in ['Least-cost', 'Cournot']:
            for year in (dispatch_df.loc[dispatch_df.scenario == selected_scenario].groupby("year").filter(lambda x: x["value"].sum() > 0)).year.unique():
                for season in dispatch_df.loc[dispatch_df.scenario == selected_scenario].season.unique():

                    select_time = {
                        'season': [season],
                        'day': ['d1', 'd2', 'd3', 'd4', 'd5']
                    }

                    filename = Path(folder) / Path('images') / Path('dispatch') / Path(f'dispatch_{selected_scenario}_{competition}_{year}_{season}.png')

                    make_complete_fuel_dispatch_plot(dfs_to_plot_area, dfs_to_plot_line, dict_colors=dict_specs['colors'],
                                                     year=year, scenario=selected_scenario, competition=competition, zone=zone,
                                                     fuel_grouping=None, select_time=select_time, filename=filename,
                                                     reorder_dispatch=['Solar', 'Concentrated Solar', 'Hydro', 'Nuclear', 'Coal', 'Gas',
                                                                       'Oil'],
                                                     figsize=(10, 4))

