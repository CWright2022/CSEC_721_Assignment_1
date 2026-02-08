import csv
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import lambertw
import plotly.graph_objects as go
import argparse

'''
CSEC-721 Lab 1: Location Privacy
Cayden Wright
'''


BASE_DIR = Path(os.path.dirname(__file__) or '.')
INPUT = BASE_DIR / 'data_input'
OUTPUT = BASE_DIR / 'data_output'
PRIVACY_BUDGET = 0.1 #epsilon

parser = argparse.ArgumentParser()
parser.add_argument('--plot_only', action='store_true')
parser.add_argument('--pertrub_only', action='store_true')


def generate_perturbation(lat: float, long: float) -> tuple[float, float]:
    '''generate pertubations based on laplace'''
    theta = np.random.uniform(0, 2 * np.pi)
    p = np.random.uniform(0, 1)
    r = (-1 / PRIVACY_BUDGET) * (lambertw((p - 1) / np.e, k=-1) + 1)
    # Take the real part to avoid complex numbers
    r = float(np.real(r))

    meters_per_degree = 111320.0
    new_lat = lat + (r * np.cos(theta)) / meters_per_degree
    new_long = long + (r * np.sin(theta)) / (meters_per_degree*np.cos(lat))

    return float(new_lat), float(new_long)

def process_csvs(input_dir: Path, output_dir: Path) -> None:
    '''
    main function to process csvs in input_dir, generate pertrubations, and write to output_dir
    '''
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob('*.csv'))
    if not csv_files:
        # in case i forgot to import the data
        print(f'No CSV files found in {input_dir}')
        return

    for csv_path in csv_files:
        out_path = output_dir / csv_path.name
        # check for headers (there shouldn't be)
        with csv_path.open('r', newline='', encoding='utf-8') as infp:
            reader = csv.reader(infp)
            first = next(reader, None)
            if first is None:
                print(f'Skipping empty file {csv_path}')
                continue
            # CSV format is (longitude, latitude); output as (latitude, longitude) for plotly
            core_names = ['latitude', 'longitude']

            fieldnames = core_names + ['perturbed_lat', 'perturbed_long']

            with out_path.open('w', newline='', encoding='utf-8') as outfp:
                writer = csv.writer(outfp)
                writer.writerow(fieldnames)
                row_count = 0
                # process the first row (numeric)
                # remember longitude, latitude, not the other way around
                try:
                    lon = float(first[0])
                    lat = float(first[1])
                except Exception:
                    continue
                #
                pert_lat, pert_lon = generate_perturbation(lat, lon)
                writer.writerow([lat, lon, pert_lat, pert_lon])
                row_count += 1

                # remaining rows
                for row in reader:
                    if not row:
                        continue
                    try:
                        lon = float(row[0])
                        lat = float(row[1])
                    except Exception:
                        # skip malformed rows
                        continue
                    pert_lat, pert_lon = generate_perturbation(lat, lon)
                    writer.writerow([lat, lon, pert_lat, pert_lon])
                    row_count += 1
        print(f'Wrote {row_count} rows to {out_path}')

def plot_csvs(output_dir: Path) -> None:
    '''
    plots pertrubed vs actual csvs using Plotly/mapbox
    '''
    csv_files = sorted(output_dir.glob('*.csv'))
    if not csv_files:
        print(f'No CSV files found in {output_dir}')
        return

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        # Create figure with original points (blue)
        figure = go.Figure()
        figure.add_trace(go.Scattermapbox(
            lat=df['latitude'],
            lon=df['longitude'],
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='Original',
        ))
        
        # Add perturbed points in red
        figure.add_trace(go.Scattermapbox(
            lat=df['perturbed_lat'],
            lon=df['perturbed_long'],
            mode='markers',
            marker=dict(size=8, color='red'),
            name='Perturbed',
        ))
        
        # Update layout with mapbox settings
        figure.update_layout(
            mapbox=dict(
                style='open-street-map',
                zoom=10,
                center=dict(
                    lat=df['latitude'].mean(),
                    lon=df['longitude'].mean()
                )
            ),
            margin=dict(r=0, t=0, l=0, b=0),
            height=700
        )
        
        figure.show()
        input(f"Plotted {csv_path.name} ({df.shape[0]} points). Press Enter to see the next plot...")
        
if __name__ == '__main__':
    args = parser.parse_args()
    if args.plot_only:
        plot_csvs(OUTPUT)
    elif args.pertrub_only:
        process_csvs(INPUT, OUTPUT)
    else:
        process_csvs(INPUT, OUTPUT)
        plot_csvs(OUTPUT)
    
