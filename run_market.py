from os import remove

import pandas as pd
import os
import subprocess
import datetime
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
from requests.auth import HTTPBasicAuth
import gams.engine
from requests import post, get
import re
from multiprocessing import Pool

from utils import *
from postprocessing.utils_plots import *
import argparse


PATH_GAMS = {
    'path_main_file': 'main.gms',
    'path_base_file': 'base.gms',
    'path_report_file': 'report.gms',
    'path_cplex_file': 'cplex.opt',
    'path_reader_file': 'input_readers.gms',
    'path_reader_cournot_file': 'input_readers_cournot.gms'
}

URL_ENGINE = "https://engine.gams.com/api"

def get_auth_engine():
    user_name = "CeliaEscribe"
    password = "cv86aWE30TG"
    auth = HTTPBasicAuth(user_name, password)
    return auth


def get_configuration():
    configuration = gams.engine.Configuration(
        host='https://engine.gams.com/api',
        username='CeliaEscribe',
        password='cv86aWE30TG')
    return configuration


def post_job_engine(scenario_name, path_zipfile):
    """
    Post a job to the GAMS Engine.

    Parameters
    ----------
    scenario_name
    path_zipfile

    Returns
    -------

    """

    auth = get_auth_engine()

    # Send the job to the server
    query_params = {
        "model": 'engine_{}'.format(scenario_name),
        "namespace": "wb",
        "labels": "instance=GAMS_z1d.2xlarge_282_S"
    }
    job_files = {"model_data": open(path_zipfile, "rb")}
    req = post(
        URL_ENGINE + "/jobs/", params=query_params, files=job_files, auth=auth
    )
    return req

def launch_market(scenario,
                  scenario_name='',
                  path_main_file='main.gms',
                  path_base_file='base.gms',
                  path_report_file='report.gms',
                  path_reader_file='input_readers.gms',
                  path_reader_cournot_file='input_readers_cournot.gms',
                  path_cplex_file='cplex.opt',
                  path_engine_file=False):
    """

    :param scenario_name: str
        Specify the scenario name for the output folder
    :param path_gams: dict
        If given, specifies which files to use in the gams command. Otherwise, default files are used from PATH_GAMS
    :param path_engine_file: str
        If given, code is run on ENGINE.
    :param run_excel: bool
        If True, uses input reading with CONNECT (compatible with Mac). Otherwise, uses default option GDXXRW.
    :return:
    """

    if scenario_name == '':
        scenario_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    folder = '{}'.format(scenario_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    cwd = os.path.join(os.getcwd(), folder)

    shutil.copy(path_cplex_file, cwd)

    path_args = ['--{} {}'.format(k, i) for k, i in scenario.items()]

    input_checks(scenario)

    options = []
    if path_engine_file:
        print('Save file only to prepare running simulation on remote server')

        options = ['a=c', 'xs=engine_{}'.format(scenario_name)]

    command = ["gams", path_main_file] + options + ["--READER CONNECT_CSV_PYTHON",
                                                    "--BASE_FILE {}".format(path_base_file),
                                                    "--REPORT_FILE {}".format(path_report_file),
                                                    "--READER_FILE {}".format(path_reader_file),
                                                    "--SCENARIO Least-Cost"
                                                    ] + path_args

    # Print the command
    print("Command to execute:", command)

    subprocess.run(command, cwd=cwd)

    scenario_cournot = scenario.copy()

    results_perfect_competition = extract_perfect_competition(cwd)
    additional_data = load_additional_data(scenario)
    scenario_cournot = prepare_data_for_cournot(results_perfect_competition, scenario_cournot, additional_data, folder=cwd, contract='firm')

    scenario_cournot.to_csv(Path(cwd) / Path('input/scenario.csv'))

    path_args = ['--{} {}'.format(k, i) for k, i in scenario_cournot.items()]

    options = []
    if path_engine_file:
        print('Save file only to prepare running simulation on remote server')

        options = ['a=c', 'xs=engine_{}'.format(scenario_name)]

    # TODO: it could be interesting to add some preprocessing in python here (for instance, for calculating the demand profile from demand forecast+hourly demand profile
    command = ["gams", path_main_file] + options + ["--READER CONNECT_CSV_PYTHON",
                                                    "--BASE_FILE {}".format(path_base_file),
                                                    "--REPORT_FILE {}".format(path_report_file),
                                                    "--READER_FILE {}".format(path_reader_file),
                                                    "--READER_FILE_COURNOT {}".format(path_reader_cournot_file),
                                                    "--SCENARIO Cournot"
                                                    ] + path_args

    # Print the command
    print("Command to execute:", command)

    subprocess.run(command, cwd=cwd)

    return 0


def launch_market_multiprocess(df, scenario_name, path_gams, path_engine_file=False):
    return launch_market(df, scenario_name=scenario_name, path_engine_file=path_engine_file, **path_gams)


def launch_market_multiple_scenarios(scenario_baseline='scenario_baseline.csv',
                                     scenarios_specification='scenarios_specification.csv',
                                     selected_scenarios=None,
                                     removed_scenarios=None,
                                     cpu=1, path_gams=None,
                                     path_engine_file=False):
    """
    Launch market model with multiple scenarios based on scenarios_specification

    Parameters
    ----------
    scenario_baseline: str, optional, default 'scenario_baseline.csv'
        Path to the CSV file with the baseline scenario
    scenarios_specification: str, optional, default 'scenarios_specification.csv'
        Path to the CSV file with the scenarios specification
    cpu: int, optional, default 1
        Number of CPUs to use
    selected_scenarios: list, optional, default None
        List of scenarios to run
    path_engine_file: str, optional, default False
    """

    working_directory = os.getcwd()

    # Add the full path to the files
    if path_engine_file:
        path_engine_file = os.path.join(working_directory, path_engine_file)

    # Read the scenario CSV file
    if path_gams is not None:  # path for required gams file is provided
        path_gams = {k: os.path.join(working_directory, i) for k, i in path_gams.items()}
    else:  # use default configuration
        path_gams = {k: os.path.join(working_directory, i) for k, i in PATH_GAMS.items()}

    # Read scenario baseline
    scenario_baseline = pd.read_csv(scenario_baseline).set_index('paramNames').squeeze()

    # Read scenarios specification
    if scenarios_specification is not None:
        scenarios = pd.read_csv(scenarios_specification).set_index('paramNames')

        # Generate scenario pd.Series for alternative scenario
        s = {k: scenario_baseline.copy() for k in scenarios}
        for k in s.keys():
            s[k].update(scenarios[k].dropna())
    else:
        s = {}

    s.update({'baseline': scenario_baseline})

    if selected_scenarios is not None:
        s = {k: s[k] for k in selected_scenarios}
    if removed_scenarios is not None:
        s = {k: v for k, v in s.items() if k not in removed_scenarios}

    # Add full path to the files
    for k in s.keys():
        s[k] = s[k].apply(lambda i: os.path.join(working_directory, 'input', i))

    # Create dir for simulation and change current working directory
    if 'output' not in os.listdir():
        os.mkdir('output')

    folder = 'simulations_run_{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder = os.path.join('output', folder)
    if not os.path.exists(folder):
        os.mkdir(folder)
        print('Folder created:', folder)
    os.chdir(folder)

    with Pool(cpu) as pool:
        result = pool.starmap(launch_market_multiprocess,
                              [(s[k], k, path_gams, path_engine_file) for k in s.keys()])

    if path_engine_file:
        pd.DataFrame(result).to_csv('tokens_simulation.csv', index=False)

    # Collect scenario.csv files and merge them
    scenario_files = [os.path.join(scenario, 'input/scenario.csv') for scenario in s.keys()]
    scenario_data = []

    for scenario, file in zip(s.keys(), scenario_files):
        if os.path.exists(file):
            df = pd.read_csv(file)
            df = df.rename(columns={'file': scenario}).set_index('paramNames')
            scenario_data.append(df)

    if scenario_data:
        final_df = pd.concat(scenario_data, axis=1)
        final_df.to_csv(os.path.join('simulation_scenarios.csv'), index=True)

    os.chdir(working_directory)

    return folder, result


def main(test_args=None):
    parser = argparse.ArgumentParser(description="Process some configurations.")

    parser.add_argument(
        "--baseline",
        type=str,
        default="input/scenario_baseline.csv",
        help="Baseline scenario file (default: input/scenario_baseline.csv)"
    )

    parser.add_argument(
        "--spec",
        type=str,
        default=None,
        help="Specifications scenario file (default: None)"
    )

    parser.add_argument(
        "--selected_scenarios",
        nargs="+",  # Accepts one or more values
        type=str,
        default=None,
        help="List of selected scenarios (default: None). Example usage: --selected_scenarios baseline HighDemand"
    )

    parser.add_argument(
        "--removed_scenarios",
        nargs="+",  # Accepts one or more values
        type=str,
        default=None,
        help="List of scenarios to remove (default: None). Example usage: --removed_scenarios HighDemand"
    )

    parser.add_argument(
        "--cpu",
        type=int,
        default=1,
        help="Number of CPUs (default: 1)"
    )

    parser.add_argument(
        "--postprocess",
        type=str,
        default=None,
        help="Run only postprocess with folder (default: None)"
    )

    parser.add_argument(
        "--multizone",
        action="store_true",
        help="Enable reduced output (default: False)"
    )


    args = parser.parse_args()  # Normal command-line parsing

    # If none do not run EPM
    if args.postprocess is None:
        folder, result = launch_market_multiple_scenarios(scenario_baseline=args.baseline,
                                         scenarios_specification=args.spec,
                                         selected_scenarios=args.selected_scenarios,
                                         removed_scenarios=args.removed_scenarios,
                                         cpu=args.cpu,
                                         path_gams=None,
                                         path_engine_file=None)

    else:
        print(f"Project folder: {args.postprocess}")
        print("EPM does not run again but use the existing simulation within the folder" )
        folder = args.postprocess

    postprocess(folder, multizone=args.multizone)


if __name__ == '__main__':

    main()
