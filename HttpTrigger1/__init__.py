import logging

import azure.functions as func

from azure.storage.filedatalake import DataLakeServiceClient, DataLakeLeaseClient

import pyspark
from pyspark.sql import SparkSession

# Load packages for parser
import os.path
import pandas as pd
import numpy as np
from copy import copy, deepcopy
from datetime import datetime as dt
import re


frequent_rms_categories = ['4000', '100', 'env_400']


class SensorData:
    """
    This class contains data for each sensor in a .med file.
    The parsing method is based on the premise that each section in the Sensor is structured in the following way:
        - A line with an unsigned integer (n) that represents the amount of lines that encompass the following section
        in the document.
        - n lines containing the information for the section.
    Sensor data are structured as follows:
        - O number of observations.
        FOR EACH OBSERVATION:
        - Timestamp of the observation and the quality of the observation.
        - Metadata for the observation:
            · RPM.
            · Power in kW.
            · Unknown.
            · Unknown.
        - RMS data
            · 0-4000 range.
            · 0-100 range.
            · 0-400 envelope.
        - Binary encoded FFT data:
            · 0-4000 range.
            · 0-100 range.
            · 0-400 envelope.
    The parser runs through the sensor data string, split by \r\n. FFT data are not split by \r\n,as they are encoded.
    This makes the division between one observation and the next more difficult to parse. To fix it, we rejoin all the
    remaining lines in the sensor data as a string and count characters (4*b being b the number of bytes for each
    FFT section). We split the data again afterwards to repeat the process in the next FFT section.
    """

    def __init__(self, name, data, rms_categories):

        self.name = name[:name.index(',')].lstrip('"').rstrip('"')

        # Number of observations for the date range in the sensor
        self.num_obs = int(data[0])
        self.data_quality = 'OK' if int(str(data[1], 'UTF-8').split(',')[-1]) >= 0 else 'NOT OK'
        timestamp = str(data[1], 'UTF-8').split(',')[1:3]
        timestamp = [dt.strptime(el.lstrip('#').rstrip('#')[:18], '%Y-%m-%d %H:%M:%S') for el in timestamp]
        self.data = data[2:]
        self.remaining_data = deepcopy(self.data)

        self.observations = []
        self.rms_categories = rms_categories

        # noinspection SpellCheckingInspection
        for el in range(1, self.num_obs):
            observation = {'metadata': [],
                           'rms_data': {},
                           'fft_data': {},
                           'timestamp': timestamp}
            for line in self.remaining_data[1:int(self.remaining_data[0])+1]:
                observation['metadata'].append(str(line, 'UTF-8').split(',')[1])
            self.remaining_data = self.remaining_data[int(self.remaining_data[0])+1:]
            observation['metadata'] = {'rpm': observation['metadata'][0], 'kw': observation['metadata'][1]}
            for i in range(0, len(self.rms_categories)):
                rms_category = None

                # There may be sensor data without RMS data, so we protect the parser against that.
                if self.remaining_data[0] != b'0':
                    for line in self.remaining_data[1:int(self.remaining_data[0])+1]:
                        if rms_category is None:
                            observation['rms_data'].update({str(line, 'UTF-8').split(',')[3]: [','.join(str(line, 'UTF-8').split(',')[1:])]})
                            rms_category = str(line, 'UTF-8').split(',')[3]
                        else:
                            observation['rms_data'][rms_category].append(','.join(str(line, 'UTF-8').split(',')[1:]))
                self.remaining_data = self.remaining_data[int(self.remaining_data[0])+1:]

            # FFT data parsing
            match = [re.match("b'.'$", str(el)) for el in self.remaining_data]
            fft_end_index = None
            for m in match:
                if m is not None:
                    fft_end_index = match.index(m)
                    break
            if fft_end_index is None:
                raise Exception('Error parsing FFT data for {}'.format(self.name))

            # Iteration through fft string

            # The amount of binary encoded data is referenced in the  fft_counter preceding the actual binary string. We
            # retrieve that counter and then decode the string. This is done once for every rms_category
            fft_counter = str(self.remaining_data[0], 'UTF-8').split(',')[0]
            fft_category = str(self.remaining_data[0], 'UTF-8').split(',')[1]
            self.remaining_data = b'\r\n'.join(self.remaining_data[1:])
            last_obs = None
            for i in range(1, len(self.rms_categories)+1):
                y_values = np.frombuffer(self.remaining_data[:4*int(fft_counter)], dtype='<f4')
                if len(y_values) != 0:
                    try:
                        data_arrays = {'x': np.arange(0, int(fft_category), int(fft_category)/len(y_values)), 'y': y_values}
                    except Exception as e:
                        pass
                else:
                    data_arrays = {'x': [], 'y': []}
                observation['fft_data'].update({fft_category: data_arrays})
                self.remaining_data = self.remaining_data[4*(int(fft_counter)+1):]
                self.remaining_data = self.remaining_data.split(b'\r\n')
                fft_counter = str(self.remaining_data[0], 'UTF-8').split(',')[0]
                fft_category = str(self.remaining_data[0], 'UTF-8').split(',')[1]
                last_obs = self.remaining_data[0]
                self.remaining_data = b'\r\n'.join(self.remaining_data[1:])

            self.remaining_data = self.remaining_data.split(b'\r\n')
            if last_obs is not None:
                timestamp = str(last_obs).split(',#')[1:]
                timestamp = [dt.strptime(el.lstrip('#').rstrip('#')[:18], '%Y-%m-%d %H:%M:%S') for el in timestamp]
            else:
                timestamp = ''

            self.observations.append(observation)


    def rms_data_as_df(self):
        """
        Deprecated. Functionality contained in ImportedFiles.
        :return:
        """
        out = dict()
        for n in self.observations:
            if len(out) == 0:
                for k in n['rms_data'].keys():
                    out.update({k: pd.DataFrame()})
            record = {'timestamp': n.get('timestamp')[0],
                      'kW': float(n.get('metadata').get('kw'))}
            for cat, value in n.get('rms_data').items():
                row = dict()
                row.update(record)
                for i, el in enumerate(value):
                    ide = el.split(',')
                    col = {'_'.join([str(i), ide[1], ide[2]]): float(ide[-1])}
                    row.update(col)
                df = pd.DataFrame(row, index=[row['timestamp']])
                df.drop('timestamp', axis=1, inplace=True)
                if out[cat].empty:
                    out[cat] = df
                else:
                    out[cat] = out[cat].append(df)
        return out

    def get_observations(self):
        return self.observations
    


class MEDData:
    """
    This class contains sensor data read from a .med file. It parses the files and stores and classifies its data.
    """

    def __init__(self, path, from_file=True, filename=None):
        if filename is None or not isinstance(filename, str):
            self.alias = ''
            self.filename = ''
        else:
            self.filename = filename
            self.alias = filename
        self.file = path
        self.windfarm = ''
        self.turbine = ''
        self.file_data = None
        self.data_info = None
        self.turbine_info = None

        # Remaining data contains the string of data that has not been parsed yet.
        self.remaining_data = None
        self.sensor_data = dict()
        self.parse_file(from_file=from_file)
        self.parse_sensor_data()

    def parse_data_info(self):
        """
        Parse general information and metadata from the file.
        """
        self.data_info = str(self.file_data[:self.file_data.index(b'\r\n"SENSOR')],
                             'UTF-8').split('\r\n')[0].replace('"', '').split(',')
        self.turbine_info = str(self.file_data[:self.file_data.index(b'\r\n"SENSOR')],
                                'UTF-8').split('\r\n')[1].replace('"', '').split(',')
        self.turbine = self.turbine_info[0]
        self.remaining_data = self.file_data[self.file_data.index(b'"SENSOR'):]

    def parse_sensor_data(self):
        """
        Iterate through the sensors in the file and store their data in the sensor_data dictionary
        :return:
        """
        aux_string = self.remaining_data
        data = aux_string.split(b'\r\n')
        sensor_data = dict()
        current_sensor = None
        for el in data:
            if b'"SENSOR' in el:

                sensor_data.update({str(el, 'UTF-8'): []})
                current_sensor = str(el, 'UTF-8')
            else:
                sensor_data[current_sensor].append(el)

        for k, v in sensor_data.items():
            try:
                s_data = SensorData(k, v, ['4000', '100', 'env_400'])
                self.sensor_data.update({s_data.name: s_data})
            except Exception as e:
                print('Error parsing {0}: {1}'.format(k, e))

    def parse_file(self, from_file=True):
        if from_file:
            with open(self.file, 'rb') as file:
                data = file.readlines()

            self.file_data = b"".join(data)
        else:
            self.file_data = self.file
        self.parse_data_info()

    def set_alias(self, alias):
        self.alias = alias

    def get_alias(self):
        return self.alias

    def set_turbine(self, turbine):
        self.turbine = turbine

    def get_turbine(self):
        return self.turbine

    def set_windfarm(self, windfarm):
        self.windfarm = windfarm

    def get_windfarm(self):
        return self.windfarm

    def get_filename(self):
        return self.filename

    def get_sensor_data(self, sensor=None):
        if sensor is None:
            return self.sensor_data
        else:
            return self.sensor_data[sensor]

    def to_dataframe(self, rms=True):
        """
        Output med parsed data into a dataframe.
        :param bool rms: If rms is false, information regarding rms is not included.
        :return: pd.DataFrame.
        """
        output_rows = list()
        for sensor, s_data in self.sensor_data.items():
            if s_data.data_quality != 'OK':
                continue
            sensor_rows = list()
            for o in s_data.observations:
                rows = list()
                main_obs = dict()
                main_obs['timestamp'] = o['timestamp'][0]
                main_obs['site'] = self.get_windfarm()
                main_obs['turbine_id'] = self.get_turbine()
                main_obs['sensor'] = sensor
                try:
                    main_obs['rpm'] = int(o['metadata']['rpm'])
                except:
                    main_obs['rpm'] = o['metadata']['rpm']
                try:
                    main_obs['kw'] = int(o['metadata']['kw'])
                except:
                    main_obs['kw'] = o['metadata']['kw']
                if rms:
                    main_obs['rms'] = o['rms_data']
                for k, fft in o['fft_data'].items():
                    r = copy(main_obs)
                    r['measurement'] = 'fft_{}'.format(k)
                    r['y_value'] = fft['y']
                    if len(fft['y']) == 0:
                        #if there is no fft data the row will be skipped.
                        continue
                    r['dF'] = fft['x'][1]
                    rows.append(r)
                sensor_rows.extend(rows)
            output_rows.extend(sensor_rows)
        output = pd.DataFrame(output_rows)

        return output




def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Collect the name of the .MED file to process (string: T001_H_20201001_MEDIDAS.MED)
    file = req.params.get('file')
    if not file:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            file = req_body.get('file')


    #connection_string = "DefaultEndpointsProtocol=https;AccountName=dlalex;AccountKey=ve15i2Y7cNoZMiQKdYa3FjL298efwf5HS9s9U53ddfKtdej4Ed4eWyspAOWqx4shiBYKOTjZaknw+ASthbPPBw==;EndpointSuffix=core.windows.net"
    storage_account_key = "ve15i2Y7cNoZMiQKdYa3FjL298efwf5HS9s9U53ddfKtdej4Ed4eWyspAOWqx4shiBYKOTjZaknw+ASthbPPBw=="
    storage_account_name = "dlalex"

    contenedor = "smp"
    source_dir = "raw_data"
    sink_dir = "processed_data"
    sink_file = file.split(".")[0] + ".csv"

    # TODO hacerlo con Azure Key Vault
    # Generate the class DataLakeServiceClient for the ADL Gen2 defined.
    service_client_sink = DataLakeServiceClient(account_url="{}://{}.dfs.core.windows.net".format("https", storage_account_name), credential=storage_account_key)

    # Generate the instances where files are stored in the ADL.
    # Define the ADL container.
    file_system_client = service_client_sink.get_file_system_client(file_system=contenedor)
    # Define the source and sink directories.
    directory_client_source = file_system_client.get_directory_client(source_dir)
    directory_client_sink = file_system_client.get_directory_client(sink_dir)
    # Define the source and sink files (.MED y .csv, respectively).
    file_client_source = directory_client_source.get_file_client(file)
    file_client_sink = directory_client_sink.create_file(sink_file)

    # Request a new lease for the sink file. This serves that it cannot be
    # modified/deleted by another process (for example, another call from 
    # Azure Functions).
    sink_lease_client = DataLakeLeaseClient(file_client_sink)
    sink_lease_client.acquire()

    # Define where to store the temporary files. In this case, it'll
    # be in "/temp/", because it's a Linux machine.
    local_raw_path = "/tmp/"+ file
    local_sink_path = "/tmp/"+ sink_file

    # Create on the local machine the .MED file (empty).
    local_file_prueba = open(local_raw_path, 'wb')

    # Download from the ADL the .MED file and store it on the path "/tmp/"
    # for the file just created.
    download = file_client_source.download_file()
    downloaded_bytes = download.readall()
    local_file_prueba.write(downloaded_bytes)
    
    # TODO controlar las excepciones (aunque no debería pasar nada puesto
    # que se ejecutan los parsers uno a uno).
    # Apply .med parser.
    smp_class = MEDData(local_raw_path)
    smp_class.set_windfarm(file.split(".")[0])
    smp = smp_class.to_dataframe()

    # Change 'y_value' column to standard python float
    smp['y_value'] = [[float(y) for y in list(x)] for x in smp['y_value']]

    # Store the generated .csv in the local path "/tmp/"
    smp_csv = smp.to_csv(local_sink_path) 

    # Open on read mode the .csv file created.
    local_file = open(local_sink_path,'r')
    file_contents = local_file.read()

    # Break the lease opened beforhand so it can be modified.
    sink_lease_client.break_lease()

    # Upload de data to the ADL.
    file_client_sink.upload_data(data=file_contents, overwrite=True, length=len(file_contents))
    # Commit the data uploaded.
    file_client_sink.flush_data(len(file_contents))

    
    if file:
        return func.HttpResponse(f"The file {file} processed successfully. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a file in the query string or in the request body for a personalized response.",
             status_code=200
        )
