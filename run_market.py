import pandas as pd
import os
import subprocess
import datetime
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
from requests.auth import HTTPBasicAuth
import gams.engine
from requests import post, get

from multiprocessing import Pool

from utils import *


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

    folder = 'simulation_{}'.format(scenario_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    cwd = os.path.join(os.getcwd(), folder)

    shutil.copy(path_cplex_file, cwd)

    # TODO : add scenario spec here


    path_args = ['--{} {}'.format(k, i) for k, i in scenario.items()]

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

    # TODO: create new scenario config file for cournot, that includes additional files
    scenario_cournot = scenario.copy()

    # TODO: extract_perfect_competition(path) with path = cwd, ResultsPerfectCompetition.gdx for gdx
    results_perfect_competition = extract_perfect_competition(cwd)
    additional_data = load_additional_data(scenario)
    scenario_cournot = prepare_data_for_cournot(results_perfect_competition, scenario_cournot, additional_data, folder=cwd)

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
                                     selected_scenarios=['baseline'],
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

    # Add the full path to the files
    if path_engine_file:
        path_engine_file = os.path.join(os.getcwd(), path_engine_file)

    # Read the scenario CSV file
    if path_gams is not None:  # path for required gams file is provided
        path_gams = {k: os.path.join(os.getcwd(), i) for k, i in path_gams.items()}
    else:  # use default configuration
        path_gams = {k: os.path.join(os.getcwd(), i) for k, i in PATH_GAMS.items()}

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

    # Add the baseline scenario
    s.update({'baseline': scenario_baseline})

    if selected_scenarios is not None:
        s = {k: s[k] for k in selected_scenarios}

    # Add full path to the files
    for k in s.keys():
        s[k] = s[k].apply(lambda i: os.path.join(os.getcwd(), 'input', i))

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
    return folder, result


if __name__ == '__main__':
    launch_market_multiple_scenarios(scenario_baseline='input/scenario_baseline.csv',
                                     scenarios_specification='input/scenario_spec_elasticity.csv',
                                     selected_scenarios=['Elasticity0p8'],
                                     cpu=1, path_gams=None,
                                     path_engine_file=None)
