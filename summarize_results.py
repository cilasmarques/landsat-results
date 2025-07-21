#!/usr/bin/env python3
"""
Script para sumarizar dados de tempo e utilização de recursos das estratégias de otimização Landsat.

Este script processa todos os arquivos CSV de tempo, CPU, GPU, SSD I/O e CPU Power de todas as estratégias
e algoritmos, calculando estatísticas (mediana, média, desvio padrão, etc.) e salvando
os resultados em arquivos CSV dentro de cada diretório de estratégia/algoritmo.

Adaptado para a estrutura output-csv-lsd-0707.
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import argparse
import sys

def get_experiment_dirs(base_path, strategy_algorithm):
    """Obtém todos os diretórios de experimentos para uma estratégia/algoritmo específicos"""
    pattern = base_path / strategy_algorithm / 'experiment*'
    experiment_dirs = [d for d in glob.glob(str(pattern)) if os.path.isdir(d)]
    return sorted(experiment_dirs)

def load_time_data(experiment_dir):
    """Carrega dados de tempo de um experimento"""
    time_file = Path(experiment_dir) / 'time.csv'
    if time_file.exists():
        try:
            df = pd.read_csv(time_file)
            experiment_name = Path(experiment_dir).name
            df['experiment'] = experiment_name
            return df
        except Exception as e:
            print(f"Erro ao carregar {time_file}: {e}")
            return None
    return None

def load_cpu_data(experiment_dir):
    """Carrega dados de CPU de um experimento"""
    cpu_file = Path(experiment_dir) / 'cpu_metrics.csv'
    if cpu_file.exists():
        try:
            df = pd.read_csv(cpu_file)
            # Limpa os nomes das colunas removendo espaços extras
            df.columns = df.columns.str.strip()
            
            # Converte colunas numéricas, tratando valores inválidos
            numeric_columns = ['CPU_USAGE_PERCENTAGE', 'MEM_USAGE_PERCENTAGE', 'MEM_USAGE_MB']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            experiment_name = Path(experiment_dir).name
            df['experiment'] = experiment_name
            return df
        except Exception as e:
            print(f"Erro ao carregar {cpu_file}: {e}")
            return None
    return None

def load_gpu_data(experiment_dir):
    """Carrega dados de GPU de um experimento"""
    gpu_file = Path(experiment_dir) / 'gpu_metrics.csv'
    if gpu_file.exists():
        try:
            df = pd.read_csv(gpu_file)
            # Limpa os nomes das colunas removendo espaços extras
            df.columns = df.columns.str.strip()
            
            # Converte colunas numéricas, tratando valores inválidos
            numeric_columns = ['GPU_USAGE_PERCENTAGE', 'MEM_USAGE_PERCENTAGE', 'MEM_USAGE_MB', 'POWER_W', 'TEMP_C']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            experiment_name = Path(experiment_dir).name
            df['experiment'] = experiment_name
            return df
        except Exception as e:
            print(f"Erro ao carregar {gpu_file}: {e}")
            return None
    return None

def load_ssd_io_data(experiment_dir):
    """Carrega dados de SSD I/O de um experimento"""
    ssd_io_file = Path(experiment_dir) / 'ssd_io_metrics.csv'
    if ssd_io_file.exists():
        try:
            df = pd.read_csv(ssd_io_file)
            # Limpa os nomes das colunas removendo espaços extras
            df.columns = df.columns.str.strip()
            
            # Converte colunas numéricas, tratando valores inválidos
            numeric_columns = ['READ_TIME_MS', 'WRITE_TIME_MS', 'READ_BYTES', 'WRITE_BYTES', 
                             'READ_OPERATIONS', 'WRITE_OPERATIONS', 'IO_WAIT_TIME_MS', 
                             'AVG_READ_TIME_MS', 'AVG_WRITE_TIME_MS']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            experiment_name = Path(experiment_dir).name
            df['experiment'] = experiment_name
            return df
        except Exception as e:
            print(f"Erro ao carregar {ssd_io_file}: {e}")
            return None
    return None

def load_cpu_power_data(experiment_dir):
    """Carrega dados de CPU Power de um experimento"""
    cpu_power_file = Path(experiment_dir) / 'cpu_power_metrics.csv'
    if cpu_power_file.exists():
        try:
            df = pd.read_csv(cpu_power_file)
            # Limpa os nomes das colunas removendo espaços extras
            df.columns = df.columns.str.strip()
            
            # Converte colunas numéricas, tratando valores inválidos
            numeric_columns = ['CPU_POWER_W', 'CPU_FREQ_MHZ', 'CPU_TEMP_C', 'PACKAGE_POWER_W', 
                             'CORES_ACTIVE', 'CPU_UTIL_PERCENT']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            experiment_name = Path(experiment_dir).name
            df['experiment'] = experiment_name
            return df
        except Exception as e:
            print(f"Erro ao carregar {cpu_power_file}: {e}")
            return None
    return None

def map_resources_to_phases(time_df, resource_df, resource_type):
    """Mapeia dados de recursos (CPU/GPU/SSD/CPU_POWER) para cada fase baseado nos timestamps"""
    if time_df is None or resource_df is None:
        return None
    
    phase_resource_data = []
    
    for _, phase_row in time_df.iterrows():
        phase_name = phase_row['PHASE']
        start_time = phase_row['START_TIME']
        end_time = phase_row['END_TIME']
        experiment = phase_row['experiment']
        strategy = phase_row['STRATEGY']
        
        # Converte os timestamps das fases para a mesma escala dos recursos
        # Os timestamps das fases estão em nanosegundos, os recursos em segundos
        start_time_sec = start_time // 1000000000  # Converte para segundos
        end_time_sec = end_time // 1000000000      # Converte para segundos
        
        # Filtra dados de recursos que estão dentro do intervalo da fase
        phase_resources = resource_df[
            (resource_df['experiment'] == experiment) &
            (resource_df['TIMESTAMP'] >= start_time_sec) &
            (resource_df['TIMESTAMP'] <= end_time_sec)
        ].copy()
        
        if not phase_resources.empty:
            # Calcula estatísticas para cada métrica disponível
            stats = {}
            
            if resource_type == 'cpu':
                metrics = ['CPU_USAGE_PERCENTAGE', 'MEM_USAGE_PERCENTAGE', 'MEM_USAGE_MB']
            elif resource_type == 'gpu':
                metrics = ['GPU_USAGE_PERCENTAGE', 'MEM_USAGE_PERCENTAGE', 'MEM_USAGE_MB', 'POWER_W', 'TEMP_C']
            elif resource_type == 'ssd_io':
                metrics = ['READ_TIME_MS', 'WRITE_TIME_MS', 'READ_BYTES', 'WRITE_BYTES', 
                          'READ_OPERATIONS', 'WRITE_OPERATIONS', 'IO_WAIT_TIME_MS', 
                          'AVG_READ_TIME_MS', 'AVG_WRITE_TIME_MS']
            elif resource_type == 'cpu_power':
                metrics = ['CPU_POWER_W', 'CPU_FREQ_MHZ', 'CPU_TEMP_C', 'PACKAGE_POWER_W', 
                          'CORES_ACTIVE', 'CPU_UTIL_PERCENT']
            else:
                metrics = []
            
            for metric in metrics:
                if metric in phase_resources.columns:
                    values = phase_resources[metric].dropna()
                    if not values.empty:
                        stats[f'{metric}_median'] = values.median()
                        stats[f'{metric}_mean'] = values.mean()
                        stats[f'{metric}_max'] = values.max()
                        stats[f'{metric}_min'] = values.min()
                        stats[f'{metric}_std'] = values.std()
                        stats[f'{metric}_count'] = len(values)
                    else:
                        stats[f'{metric}_median'] = np.nan
                        stats[f'{metric}_mean'] = np.nan
                        stats[f'{metric}_max'] = np.nan
                        stats[f'{metric}_min'] = np.nan
                        stats[f'{metric}_std'] = np.nan
                        stats[f'{metric}_count'] = 0
            
            # Adiciona informações da fase
            stats['PHASE'] = phase_name
            stats['STRATEGY'] = strategy
            stats['experiment'] = experiment
            stats['START_TIME'] = start_time
            stats['END_TIME'] = end_time
            stats['PHASE_DURATION'] = end_time - start_time
            
            phase_resource_data.append(stats)
    
    if phase_resource_data:
        return pd.DataFrame(phase_resource_data)
    return None

def save_summary_to_csv(summary_df, results_dir, strategy_algorithm, metric_type):
    """Salva o resumo em um arquivo CSV dentro do diretório results"""
    if summary_df is not None and not summary_df.empty:
        # Define o caminho do diretório da estratégia/algoritmo em results
        strategy_dir = Path(results_dir) / strategy_algorithm
        
        # Cria o diretório se não existir
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Define o nome do arquivo baseado no tipo de métrica
        if metric_type == 'time':
            filename = 'final-time.csv'
        elif metric_type == 'cpu':
            filename = 'cpu-time.csv'
        elif metric_type == 'gpu':
            filename = 'gpu-time.csv'
        elif metric_type == 'ssd_io':
            filename = 'ssd-io-time.csv'
        elif metric_type == 'cpu_power':
            filename = 'cpu-power-time.csv'
        elif metric_type == 'cpu_by_phase':
            filename = 'cpu-by-phase.csv'
        elif metric_type == 'gpu_by_phase':
            filename = 'gpu-by-phase.csv'
        elif metric_type == 'ssd_io_by_phase':
            filename = 'ssd-io-by-phase.csv'
        elif metric_type == 'cpu_power_by_phase':
            filename = 'cpu-power-by-phase.csv'
        else:
            filename = f'{metric_type}-time.csv'
        
        output_file = strategy_dir / filename
        summary_df.to_csv(output_file, index=False)
        print(f"  Salvo: {output_file}")
    else:
        print(f"  Nenhum dado para salvar em {metric_type}")

def summarize_time_data(base_path, strategy_algorithm):
    """Sumariza dados de tempo para uma estratégia/algoritmo"""
    print(f"Processando tempo para {strategy_algorithm}...")
    
    experiment_dirs = get_experiment_dirs(base_path, strategy_algorithm)
    print(f"  Encontrados {len(experiment_dirs)} experimentos")
    
    all_time_data = []
    
    for exp_dir in experiment_dirs:
        time_df = load_time_data(exp_dir)
        if time_df is not None:
            all_time_data.append(time_df)
    
    if not all_time_data:
        print(f"  Nenhum dado de tempo encontrado para {strategy_algorithm}")
        return None
    
    # Combina todos os dados
    combined_df = pd.concat(all_time_data, ignore_index=True)
    
    # Obtém a ordem original das fases do primeiro experimento
    first_experiment = all_time_data[0]
    phase_order = first_experiment[['STRATEGY', 'PHASE']].drop_duplicates()
    
    # Calcula estatísticas por fase
    summary = combined_df.groupby(['STRATEGY', 'PHASE']).agg({
        'TIMESTAMP': ['count', 'median', 'mean', 'std', 'min', 'max']
    }).round(6)
    
    # Flatten das colunas
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    # Mantém a ordem original das fases usando merge com how='left' e preserve_order=True
    # Cria um índice para manter a ordem original
    phase_order['original_order'] = range(len(phase_order))
    summary = summary.merge(phase_order, on=['STRATEGY', 'PHASE'], how='right')
    summary = summary.sort_values('original_order').reset_index(drop=True)
    summary = summary.drop('original_order', axis=1)
    
    # Adiciona informações da estratégia e algoritmo
    summary['strategy_algorithm'] = strategy_algorithm
    
    print(f"  Processados {len(combined_df)} registros de tempo")
    return summary

def summarize_cpu_data(base_path, strategy_algorithm):
    """Sumariza dados de CPU para uma estratégia/algoritmo"""
    print(f"Processando CPU para {strategy_algorithm}...")
    
    experiment_dirs = get_experiment_dirs(base_path, strategy_algorithm)
    print(f"  Encontrados {len(experiment_dirs)} experimentos")
    
    all_cpu_data = []
    
    for exp_dir in experiment_dirs:
        cpu_df = load_cpu_data(exp_dir)
        if cpu_df is not None:
            all_cpu_data.append(cpu_df)
    
    if not all_cpu_data:
        print(f"  Nenhum dado de CPU encontrado para {strategy_algorithm}")
        return None
    
    # Combina todos os dados
    combined_df = pd.concat(all_cpu_data, ignore_index=True)
    
    # Verifica se as colunas necessárias existem
    required_columns = ['CPU_USAGE_PERCENTAGE', 'MEM_USAGE_PERCENTAGE', 'MEM_USAGE_MB']
    available_columns = [col for col in required_columns if col in combined_df.columns]
    
    if not available_columns:
        print(f"  Nenhuma coluna de CPU encontrada. Colunas disponíveis: {list(combined_df.columns)}")
        return None
    
    # Calcula estatísticas por experimento apenas para as colunas disponíveis
    agg_dict = {}
    for col in available_columns:
        agg_dict[col] = ['median', 'mean', 'std', 'min', 'max']
    
    summary = combined_df.groupby('experiment').agg(agg_dict).round(6)
    
    # Flatten das colunas
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    # Adiciona informações da estratégia e algoritmo
    summary['strategy_algorithm'] = strategy_algorithm
    
    print(f"  Processados {len(combined_df)} registros de CPU")
    return summary

def summarize_gpu_data(base_path, strategy_algorithm):
    """Sumariza dados de GPU para uma estratégia/algoritmo"""
    print(f"Processando GPU para {strategy_algorithm}...")
    
    experiment_dirs = get_experiment_dirs(base_path, strategy_algorithm)
    print(f"  Encontrados {len(experiment_dirs)} experimentos")
    
    all_gpu_data = []
    
    for exp_dir in experiment_dirs:
        gpu_df = load_gpu_data(exp_dir)
        if gpu_df is not None:
            all_gpu_data.append(gpu_df)
    
    if not all_gpu_data:
        print(f"  Nenhum dado de GPU encontrado para {strategy_algorithm}")
        return None
    
    # Combina todos os dados
    combined_df = pd.concat(all_gpu_data, ignore_index=True)
    
    # Verifica se as colunas necessárias existem
    required_columns = ['GPU_USAGE_PERCENTAGE', 'MEM_USAGE_PERCENTAGE', 'MEM_USAGE_MB', 'POWER_W', 'TEMP_C']
    available_columns = [col for col in required_columns if col in combined_df.columns]
    
    if not available_columns:
        print(f"  Nenhuma coluna de GPU encontrada. Colunas disponíveis: {list(combined_df.columns)}")
        return None
    
    # Calcula estatísticas por experimento apenas para as colunas disponíveis
    agg_dict = {}
    for col in available_columns:
        agg_dict[col] = ['median', 'mean', 'std', 'min', 'max']
    
    summary = combined_df.groupby('experiment').agg(agg_dict).round(6)
    
    # Flatten das colunas
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    # Adiciona informações da estratégia e algoritmo
    summary['strategy_algorithm'] = strategy_algorithm
    
    print(f"  Processados {len(combined_df)} registros de GPU")
    return summary

def summarize_ssd_io_data(base_path, strategy_algorithm):
    """Sumariza dados de SSD I/O para uma estratégia/algoritmo"""
    print(f"Processando SSD I/O para {strategy_algorithm}...")
    
    experiment_dirs = get_experiment_dirs(base_path, strategy_algorithm)
    print(f"  Encontrados {len(experiment_dirs)} experimentos")
    
    all_ssd_io_data = []
    
    for exp_dir in experiment_dirs:
        ssd_io_df = load_ssd_io_data(exp_dir)
        if ssd_io_df is not None:
            all_ssd_io_data.append(ssd_io_df)
    
    if not all_ssd_io_data:
        print(f"  Nenhum dado de SSD I/O encontrado para {strategy_algorithm}")
        return None
    
    # Combina todos os dados
    combined_df = pd.concat(all_ssd_io_data, ignore_index=True)
    
    # Verifica se as colunas necessárias existem
    required_columns = ['READ_TIME_MS', 'WRITE_TIME_MS', 'READ_BYTES', 'WRITE_BYTES', 
                       'READ_OPERATIONS', 'WRITE_OPERATIONS', 'IO_WAIT_TIME_MS', 
                       'AVG_READ_TIME_MS', 'AVG_WRITE_TIME_MS']
    available_columns = [col for col in required_columns if col in combined_df.columns]
    
    if not available_columns:
        print(f"  Nenhuma coluna de SSD I/O encontrada. Colunas disponíveis: {list(combined_df.columns)}")
        return None
    
    # Calcula estatísticas por experimento apenas para as colunas disponíveis
    agg_dict = {}
    for col in available_columns:
        agg_dict[col] = ['median', 'mean', 'std', 'min', 'max']
    
    summary = combined_df.groupby('experiment').agg(agg_dict).round(6)
    
    # Flatten das colunas
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    # Adiciona informações da estratégia e algoritmo
    summary['strategy_algorithm'] = strategy_algorithm
    
    print(f"  Processados {len(combined_df)} registros de SSD I/O")
    return summary

def summarize_cpu_power_data(base_path, strategy_algorithm):
    """Sumariza dados de CPU Power para uma estratégia/algoritmo"""
    print(f"Processando CPU Power para {strategy_algorithm}...")
    
    experiment_dirs = get_experiment_dirs(base_path, strategy_algorithm)
    print(f"  Encontrados {len(experiment_dirs)} experimentos")
    
    all_cpu_power_data = []
    
    for exp_dir in experiment_dirs:
        cpu_power_df = load_cpu_power_data(exp_dir)
        if cpu_power_df is not None:
            all_cpu_power_data.append(cpu_power_df)
    
    if not all_cpu_power_data:
        print(f"  Nenhum dado de CPU Power encontrado para {strategy_algorithm}")
        return None
    
    # Combina todos os dados
    combined_df = pd.concat(all_cpu_power_data, ignore_index=True)
    
    # Verifica se as colunas necessárias existem
    required_columns = ['CPU_POWER_W', 'CPU_FREQ_MHZ', 'CPU_TEMP_C', 'PACKAGE_POWER_W', 
                       'CORES_ACTIVE', 'CPU_UTIL_PERCENT']
    available_columns = [col for col in required_columns if col in combined_df.columns]
    
    if not available_columns:
        print(f"  Nenhuma coluna de CPU Power encontrada. Colunas disponíveis: {list(combined_df.columns)}")
        return None
    
    # Calcula estatísticas por experimento apenas para as colunas disponíveis
    agg_dict = {}
    for col in available_columns:
        agg_dict[col] = ['median', 'mean', 'std', 'min', 'max']
    
    summary = combined_df.groupby('experiment').agg(agg_dict).round(6)
    
    # Flatten das colunas
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    # Adiciona informações da estratégia e algoritmo
    summary['strategy_algorithm'] = strategy_algorithm
    
    print(f"  Processados {len(combined_df)} registros de CPU Power")
    return summary

def summarize_resources_by_phase(base_path, strategy_algorithm):
    """Sumariza dados de CPU, GPU, SSD I/O e CPU Power mapeados por fase"""
    print(f"Processando recursos por fase para {strategy_algorithm}...")
    
    experiment_dirs = get_experiment_dirs(base_path, strategy_algorithm)
    print(f"  Encontrados {len(experiment_dirs)} experimentos")
    
    all_cpu_by_phase = []
    all_gpu_by_phase = []
    all_ssd_io_by_phase = []
    all_cpu_power_by_phase = []
    
    for exp_dir in experiment_dirs:
        # Carrega dados de tempo e recursos
        time_df = load_time_data(exp_dir)
        cpu_df = load_cpu_data(exp_dir)
        gpu_df = load_gpu_data(exp_dir)
        ssd_io_df = load_ssd_io_data(exp_dir)
        cpu_power_df = load_cpu_power_data(exp_dir)
        
        if time_df is not None:
            # Mapeia CPU por fase
            if cpu_df is not None:
                cpu_by_phase = map_resources_to_phases(time_df, cpu_df, 'cpu')
                if cpu_by_phase is not None:
                    all_cpu_by_phase.append(cpu_by_phase)
            
            # Mapeia GPU por fase (apenas para estratégias não-serial)
            if 'serial' not in strategy_algorithm.lower() and gpu_df is not None:
                gpu_by_phase = map_resources_to_phases(time_df, gpu_df, 'gpu')
                if gpu_by_phase is not None:
                    all_gpu_by_phase.append(gpu_by_phase)
            
            # Mapeia SSD I/O por fase
            if ssd_io_df is not None:
                ssd_io_by_phase = map_resources_to_phases(time_df, ssd_io_df, 'ssd_io')
                if ssd_io_by_phase is not None:
                    all_ssd_io_by_phase.append(ssd_io_by_phase)
            
            # Mapeia CPU Power por fase
            if cpu_power_df is not None:
                cpu_power_by_phase = map_resources_to_phases(time_df, cpu_power_df, 'cpu_power')
                if cpu_power_by_phase is not None:
                    all_cpu_power_by_phase.append(cpu_power_by_phase)
    
    # Combina dados de CPU por fase
    cpu_summary = None
    if all_cpu_by_phase:
        combined_cpu = pd.concat(all_cpu_by_phase, ignore_index=True)
        
        # Calcula estatísticas por fase
        cpu_metrics = [col for col in combined_cpu.columns if any(metric in col for metric in ['CPU_USAGE_PERCENTAGE', 'MEM_USAGE_PERCENTAGE', 'MEM_USAGE_MB'])]
        
        if cpu_metrics:
            agg_dict = {}
            for metric in cpu_metrics:
                if metric.endswith(('_median', '_mean', '_max', '_min', '_std', '_count')):
                    agg_dict[metric] = ['median', 'mean', 'std', 'min', 'max']
            
            if agg_dict:
                cpu_summary = combined_cpu.groupby(['STRATEGY', 'PHASE']).agg(agg_dict).round(6)
                cpu_summary.columns = ['_'.join(col).strip() for col in cpu_summary.columns]
                cpu_summary = cpu_summary.reset_index()
                cpu_summary['strategy_algorithm'] = strategy_algorithm
    
    # Combina dados de GPU por fase
    gpu_summary = None
    if all_gpu_by_phase:
        combined_gpu = pd.concat(all_gpu_by_phase, ignore_index=True)
        
        # Calcula estatísticas por fase
        gpu_metrics = [col for col in combined_gpu.columns if any(metric in col for metric in ['GPU_USAGE_PERCENTAGE', 'MEM_USAGE_PERCENTAGE', 'MEM_USAGE_MB', 'POWER_W', 'TEMP_C'])]
        
        if gpu_metrics:
            agg_dict = {}
            for metric in gpu_metrics:
                if metric.endswith(('_median', '_mean', '_max', '_min', '_std', '_count')):
                    agg_dict[metric] = ['median', 'mean', 'std', 'min', 'max']
            
            if agg_dict:
                gpu_summary = combined_gpu.groupby(['STRATEGY', 'PHASE']).agg(agg_dict).round(6)
                gpu_summary.columns = ['_'.join(col).strip() for col in gpu_summary.columns]
                gpu_summary = gpu_summary.reset_index()
                gpu_summary['strategy_algorithm'] = strategy_algorithm
    
    # Combina dados de SSD I/O por fase
    ssd_io_summary = None
    if all_ssd_io_by_phase:
        combined_ssd_io = pd.concat(all_ssd_io_by_phase, ignore_index=True)
        
        # Calcula estatísticas por fase
        ssd_io_metrics = [col for col in combined_ssd_io.columns if any(metric in col for metric in ['READ_TIME_MS', 'WRITE_TIME_MS', 'READ_BYTES', 'WRITE_BYTES', 
                                                                                                        'READ_OPERATIONS', 'WRITE_OPERATIONS', 'IO_WAIT_TIME_MS', 
                                                                                                        'AVG_READ_TIME_MS', 'AVG_WRITE_TIME_MS'])]
        
        if ssd_io_metrics:
            agg_dict = {}
            for metric in ssd_io_metrics:
                if metric.endswith(('_median', '_mean', '_max', '_min', '_std', '_count')):
                    agg_dict[metric] = ['median', 'mean', 'std', 'min', 'max']
            
            if agg_dict:
                ssd_io_summary = combined_ssd_io.groupby(['STRATEGY', 'PHASE']).agg(agg_dict).round(6)
                ssd_io_summary.columns = ['_'.join(col).strip() for col in ssd_io_summary.columns]
                ssd_io_summary = ssd_io_summary.reset_index()
                ssd_io_summary['strategy_algorithm'] = strategy_algorithm
    
    # Combina dados de CPU Power por fase
    cpu_power_summary = None
    if all_cpu_power_by_phase:
        combined_cpu_power = pd.concat(all_cpu_power_by_phase, ignore_index=True)
        
        # Calcula estatísticas por fase
        cpu_power_metrics = [col for col in combined_cpu_power.columns if any(metric in col for metric in ['CPU_POWER_W', 'CPU_FREQ_MHZ', 'CPU_TEMP_C', 'PACKAGE_POWER_W', 
                                                                                                           'CORES_ACTIVE', 'CPU_UTIL_PERCENT'])]
        
        if cpu_power_metrics:
            agg_dict = {}
            for metric in cpu_power_metrics:
                if metric.endswith(('_median', '_mean', '_max', '_min', '_std', '_count')):
                    agg_dict[metric] = ['median', 'mean', 'std', 'min', 'max']
            
            if agg_dict:
                cpu_power_summary = combined_cpu_power.groupby(['STRATEGY', 'PHASE']).agg(agg_dict).round(6)
                cpu_power_summary.columns = ['_'.join(col).strip() for col in cpu_power_summary.columns]
                cpu_power_summary = cpu_power_summary.reset_index()
                cpu_power_summary['strategy_algorithm'] = strategy_algorithm
    
    print(f"  Processados recursos por fase para {len(experiment_dirs)} experimentos")
    return cpu_summary, gpu_summary, ssd_io_summary, cpu_power_summary

def main():
    parser = argparse.ArgumentParser(description='Sumariza dados de tempo e recursos das estratégias Landsat')
    parser.add_argument('--input', '-i', type=str, default='results/output-csv-lsd-0707',
                       help='Caminho para o diretório output-csv-lsd-0707 (padrão: results/output-csv-lsd-0707)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Modo verboso')
    
    args = parser.parse_args()
    
    # Configuração dos caminhos
    base_path = Path(args.input)
    results_dir = Path('summarized_results')
    
    # Verifica se o diretório de entrada existe
    if not base_path.exists():
        print(f"Erro: Diretório de entrada não encontrado: {base_path}")
        sys.exit(1)
    
    # Cria o diretório results se não existir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Estratégias/algoritmos disponíveis (baseado na nova estrutura)
    strategy_algorithms = [
        'parallel2-sebal',
        'parallel4-sebal', 
        'parallel8-sebal',
        'parallel12-sebal',
        'parallel2-steep',
        'parallel4-steep',
        'parallel8-steep', 
        'parallel12-steep',
        'kernels-raw-sebal',
        'kernels-raw-steep',
        'kernels-streams-sebal',
        'kernels-streams-steep',
        'serial-sebal',
        'serial-steep'
    ]
    
    if args.verbose:
        print(f"Estratégias/Algoritmos: {strategy_algorithms}")
        print(f"Caminho base: {base_path.absolute()}")
    
    print("=== INICIANDO PROCESSAMENTO DE TODOS OS DADOS ===\n")
    
    # Contadores para estatísticas
    total_strategies_processed = 0
    total_files_generated = 0
    
    # Processa cada estratégia/algoritmo
    for strategy_algorithm in strategy_algorithms:
        print(f"\n--- Processando {strategy_algorithm} ---")
        
        # Verifica se o diretório existe
        strategy_path = base_path / strategy_algorithm
        if not strategy_path.exists():
            print(f"  Diretório não encontrado: {strategy_path}")
            continue
        
        total_strategies_processed += 1
        
        # Processa dados de tempo
        time_summary = summarize_time_data(base_path, strategy_algorithm)
        if time_summary is not None:
            save_summary_to_csv(time_summary, results_dir, strategy_algorithm, 'time')
            total_files_generated += 1
        
        # Processa dados de CPU
        cpu_summary = summarize_cpu_data(base_path, strategy_algorithm)
        if cpu_summary is not None:
            save_summary_to_csv(cpu_summary, results_dir, strategy_algorithm, 'cpu')
            total_files_generated += 1
        
        # Processa dados de GPU (apenas para estratégias não-serial)
        if 'serial' not in strategy_algorithm.lower():
            gpu_summary = summarize_gpu_data(base_path, strategy_algorithm)
            if gpu_summary is not None:
                save_summary_to_csv(gpu_summary, results_dir, strategy_algorithm, 'gpu')
                total_files_generated += 1
        
        # Processa dados de SSD I/O
        ssd_io_summary = summarize_ssd_io_data(base_path, strategy_algorithm)
        if ssd_io_summary is not None:
            save_summary_to_csv(ssd_io_summary, results_dir, strategy_algorithm, 'ssd_io')
            total_files_generated += 1
        
        # Processa dados de CPU Power
        cpu_power_summary = summarize_cpu_power_data(base_path, strategy_algorithm)
        if cpu_power_summary is not None:
            save_summary_to_csv(cpu_power_summary, results_dir, strategy_algorithm, 'cpu_power')
            total_files_generated += 1
        
        # Processa recursos por fase
        cpu_by_phase, gpu_by_phase, ssd_io_by_phase, cpu_power_by_phase = summarize_resources_by_phase(base_path, strategy_algorithm)
        if cpu_by_phase is not None:
            save_summary_to_csv(cpu_by_phase, results_dir, strategy_algorithm, 'cpu_by_phase')
            total_files_generated += 1
        if gpu_by_phase is not None:
            save_summary_to_csv(gpu_by_phase, results_dir, strategy_algorithm, 'gpu_by_phase')
            total_files_generated += 1
        if ssd_io_by_phase is not None:
            save_summary_to_csv(ssd_io_by_phase, results_dir, strategy_algorithm, 'ssd_io_by_phase')
            total_files_generated += 1
        if cpu_power_by_phase is not None:
            save_summary_to_csv(cpu_power_by_phase, results_dir, strategy_algorithm, 'cpu_power_by_phase')
            total_files_generated += 1
    
    print("\n=== PROCESSAMENTO CONCLUÍDO ===")
    
    # Resumo final
    print("\n=== RESUMO FINAL ===")
    print(f"Estratégias/Algoritmos processados: {total_strategies_processed}")
    print(f"Arquivos CSV gerados: {total_files_generated}")
    
    print(f"\nEstrutura de arquivos gerados:")
    for strategy_algorithm in strategy_algorithms:
        strategy_path = results_dir / strategy_algorithm
        if strategy_path.exists():
            print(f"\n{strategy_algorithm}/:")
            if (strategy_path / 'final-time.csv').exists():
                print(f"  - final-time.csv")
            if (strategy_path / 'cpu-time.csv').exists():
                print(f"  - cpu-time.csv")
            if 'serial' not in strategy_algorithm.lower() and (strategy_path / 'gpu-time.csv').exists():
                print(f"  - gpu-time.csv")
            if (strategy_path / 'ssd-io-time.csv').exists():
                print(f"  - ssd-io-time.csv")
            if (strategy_path / 'cpu-power-time.csv').exists():
                print(f"  - cpu-power-time.csv")
            if (strategy_path / 'cpu-by-phase.csv').exists():
                print(f"  - cpu-by-phase.csv")
            if 'serial' not in strategy_algorithm.lower() and (strategy_path / 'gpu-by-phase.csv').exists():
                print(f"  - gpu-by-phase.csv")
            if (strategy_path / 'ssd-io-by-phase.csv').exists():
                print(f"  - ssd-io-by-phase.csv")
            if (strategy_path / 'cpu-power-by-phase.csv').exists():
                print(f"  - cpu-power-by-phase.csv")
    
    print("\nProcessamento concluído com sucesso!")

if __name__ == "__main__":
    main() 