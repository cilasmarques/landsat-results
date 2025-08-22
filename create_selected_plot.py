#!/usr/bin/env python3
"""
Script para gerar plot de tempo total apenas com estratégias selecionadas:
- CPU Serial Double (serial-double-*)
- GPU CUDA Double (kernels-double-r-*)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from plotnine import *
import warnings
warnings.filterwarnings('ignore')

# Macros para nomenclatura dos gráficos
CPU_SERIAL_DOUBLE = 'CPU Serial E/S assincrona'
GPU_CUDA_DOUBLE = 'GPU E/S assincrona'

# Macros para comparação GPU
GPU_NORMAL = 'GPU'
GPU_ASYNC = 'GPU E/S\nassincrona'

# Macros para dados que serão usados  
STRATEGY_ALGORITHMS = [
    'serial-double-st-sebal', 'serial-double-st-steep',
    'kernels-double-fm-st-sebal', 'kernels-double-fm-st-steep'
]

# Estratégias GPU para comparação
STRATEGY_ALGORITHMS_GPU_COMPARISON = [
    'kernels-double-fm-r-sebal', 'kernels-double-fm-r-steep',
    'kernels-double-fm-st-sebal', 'kernels-double-fm-st-steep'
]

# Estratégias para comparação de leitura
STRATEGY_ALGORITHMS_READING_COMPARISON = [
    'serial-double-r-sebal', 'serial-double-r-steep',
    'serial-double-st-sebal', 'serial-double-st-steep',
    'kernels-double-fm-r-sebal', 'kernels-double-fm-r-steep',
    'kernels-double-fm-st-sebal', 'kernels-double-fm-st-steep'
]

# Estratégias selecionadas (padronizado)
SELECTED_STRATEGIES = [CPU_SERIAL_DOUBLE, GPU_CUDA_DOUBLE]
GPU_COMPARISON_STRATEGIES = [GPU_NORMAL, GPU_ASYNC]

def extract_strategy_and_algorithm(df):
    """Extrai estratégia e algoritmo da coluna strategy_algorithm"""
    df = df.copy()
    
    # Mapear estratégias e algoritmos
    strategy_mapping = {
        'serial-double-st': CPU_SERIAL_DOUBLE,
        'kernels-double-fm-st': GPU_CUDA_DOUBLE
    }
    
    algorithm_mapping = {
        'sebal': 'SEBAL',
        'steep': 'STEEP',
    }
    
    # Extrair estratégia e algoritmo
    df['strategy_precision'] = df['strategy_algorithm'].str.rsplit('-', n=1, expand=True)[0]
    df['algorithm'] = df['strategy_algorithm'].str.rsplit('-', n=1, expand=True)[1]
    
    # Aplicar mapeamentos
    df['strategy'] = df['strategy_precision'].map(strategy_mapping)
    df['algorithm'] = df['algorithm'].map(algorithm_mapping)
    
    # Remover coluna temporária
    df = df.drop('strategy_precision', axis=1)
    
    return df

def extract_strategy_and_algorithm_gpu_comparison(df):
    """Extrai estratégia e algoritmo para comparação GPU (r vs st)"""
    df = df.copy()
    
    # Mapear estratégias GPU
    strategy_mapping = {
        'kernels-double-fm-r': GPU_NORMAL,
        'kernels-double-fm-st': GPU_ASYNC
    }
    
    algorithm_mapping = {
        'sebal': 'SEBAL',
        'steep': 'STEEP',
    }
    
    # Extrair estratégia e algoritmo
    df['strategy_precision'] = df['strategy_algorithm'].str.rsplit('-', n=1, expand=True)[0]
    df['algorithm'] = df['strategy_algorithm'].str.rsplit('-', n=1, expand=True)[1]
    
    # Aplicar mapeamentos
    df['strategy'] = df['strategy_precision'].map(strategy_mapping)
    df['algorithm'] = df['algorithm'].map(algorithm_mapping)
    
    # Remover coluna temporária
    df = df.drop('strategy_precision', axis=1)
    
    return df

def extract_strategy_and_algorithm_reading_comparison(df):
    """Extrai estratégia e algoritmo para comparação de leitura"""
    df = df.copy()
    
    # Mapear estratégias
    strategy_mapping = {
        'serial-double-r': 'CPU Serial',
        'serial-double-st': 'CPU Serial\nE/S assincrona',
        'kernels-double-fm-r': 'GPU',
        'kernels-double-fm-st': 'GPU E/S\nassincrona'
    }
    
    algorithm_mapping = {
        'sebal': 'SEBAL',
        'steep': 'STEEP',
    }
    
    # Extrair estratégia e algoritmo
    df['strategy_precision'] = df['strategy_algorithm'].str.rsplit('-', n=1, expand=True)[0]
    df['algorithm'] = df['strategy_algorithm'].str.rsplit('-', n=1, expand=True)[1]
    
    # Aplicar mapeamentos
    df['strategy'] = df['strategy_precision'].map(strategy_mapping)
    df['algorithm'] = df['algorithm'].map(algorithm_mapping)
    
    # Remover coluna temporária
    df = df.drop('strategy_precision', axis=1)
    
    return df

def load_time_data(results_dir):
    """Carrega dados de tempo das estratégias selecionadas"""
    all_data = []
    
    for strategy_algorithm in STRATEGY_ALGORITHMS:
        file_path = Path(results_dir) / strategy_algorithm / 'final-time.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            # Filtrar apenas a fase P_TOTAL
            df = df[df['PHASE'] == 'P_TOTAL']
            all_data.append(df)
            print(f"Carregado: {strategy_algorithm} - {len(df)} registros")
        else:
            print(f"Arquivo não encontrado: {file_path}")
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    return None

def load_time_data_gpu_comparison(results_dir):
    """Carrega dados de tempo para comparação GPU (r vs st)"""
    all_data = []
    
    for strategy_algorithm in STRATEGY_ALGORITHMS_GPU_COMPARISON:
        file_path = Path(results_dir) / strategy_algorithm / 'final-time.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            # Filtrar apenas a fase P_TOTAL
            df = df[df['PHASE'] == 'P_TOTAL']
            all_data.append(df)
            print(f"Carregado GPU comparison: {strategy_algorithm} - {len(df)} registros")
        else:
            print(f"Arquivo não encontrado: {file_path}")
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    return None

def load_time_data_reading_comparison(results_dir):
    """Carrega dados de tempo da fase de leitura para todas as estratégias"""
    all_data = []
    
    for strategy_algorithm in STRATEGY_ALGORITHMS_READING_COMPARISON:
        file_path = Path(results_dir) / strategy_algorithm / 'final-time.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            # Filtrar apenas a fase P0_READ_INPUT
            df = df[df['PHASE'] == 'P0_READ_INPUT']
            all_data.append(df)
            print(f"Carregado reading comparison: {strategy_algorithm} - {len(df)} registros")
        else:
            print(f"Arquivo não encontrado: {file_path}")
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    return None

def create_selected_strategies_plot(time_data, output_dir):
    """Cria plot de tempo total apenas com as estratégias selecionadas"""
    print("Gerando plot de tempo total com estratégias selecionadas...")

    # Extrair strategy e algorithm
    time_data = extract_strategy_and_algorithm(time_data)
    
    # Converter de milissegundos para segundos
    time_data = time_data.copy()
    time_data['TIMESTAMP_median'] = time_data['TIMESTAMP_median'] / 1000.0

    # Definir ordem das estratégias
    time_data['strategy'] = pd.Categorical(time_data['strategy'], categories=SELECTED_STRATEGIES, ordered=True)

    # Calcular estatísticas com intervalo de confiança
    def calculate_stats(group):
        mean_val = group['TIMESTAMP_median'].mean()
        std_val = group['TIMESTAMP_median'].std()
        n = len(group)
        se = std_val / np.sqrt(n) if n > 1 else 0
        ci_95 = 1.96 * se
        return pd.Series({
            'mean': mean_val,
            'std': std_val,
            'se': se,
            'ci_95': ci_95,
            'n': n
        })

    # Agrupar por estratégia e algoritmo, calculando estatísticas
    total_time_data = time_data.groupby(['strategy', 'algorithm']).agg({
        'TIMESTAMP_median': ['mean', 'std', 'count']
    }).reset_index()
    
    # Renomear colunas
    total_time_data.columns = ['strategy', 'algorithm', 'mean', 'std', 'n']
    
    # Calcular intervalo de confiança
    total_time_data['se'] = total_time_data['std'] / np.sqrt(total_time_data['n'])
    total_time_data['ci_95'] = 1.96 * total_time_data['se']

    # Plot separado para STEEP
    steep_data = total_time_data[total_time_data['algorithm'] == 'STEEP']
    if not steep_data.empty:
        p_steep = (ggplot(steep_data, aes(x='strategy', y='mean')) +
                  geom_bar(stat='identity', fill='#4ECDC4', width=0.7) +
                  geom_errorbar(aes(ymin='mean-ci_95', ymax='mean+ci_95'), 
                               width=0.2, color='black', size=1) +
                  labs(title='Tempo Total de Execução - CPU vs GPU Double - Algoritmo STEEP (IC 95%)',
                       x='Estratégia', y='Tempo Total (segundos)') +
                  scale_y_continuous(limits=(0, 2), breaks=[x/10 for x in range(0, 21, 2)]) +
                  theme_bw() +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(10, 6)))
        p_steep.save(output_dir / 'y-tempo_total_cpu_vs_gpu_double_steep.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'y-tempo_total_cpu_vs_gpu_double_steep.png'}")

    # Plot separado para SEBAL
    sebal_data = total_time_data[total_time_data['algorithm'] == 'SEBAL']
    if not sebal_data.empty:
        p_sebal = (ggplot(sebal_data, aes(x='strategy', y='mean')) +
                  geom_bar(stat='identity', fill='#FF6B6B', width=0.7) +
                  geom_errorbar(aes(ymin='mean-ci_95', ymax='mean+ci_95'), 
                               width=0.2, color='black', size=1) +
                  labs(title='Tempo Total de Execução - CPU vs GPU Double - Algoritmo SEBAL (IC 95%)',
                       x='Estratégia', y='Tempo Total (segundos)') +
                  scale_y_continuous(limits=(0, 2), breaks=[x/10 for x in range(0, 21, 2)]) +
                  theme_bw() +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(10, 6)))
        p_sebal.save(output_dir / 'y-tempo_total_cpu_vs_gpu_double_sebal.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'y-tempo_total_cpu_vs_gpu_double_sebal.png'}")

def create_gpu_comparison_plot(time_data, output_dir):
    """Cria plot de tempo total comparando GPU vs GPU E/S assíncrona"""
    print("Gerando plot de comparação GPU vs GPU E/S assíncrona...")

    # Extrair strategy e algorithm para comparação GPU
    time_data = extract_strategy_and_algorithm_gpu_comparison(time_data)
    
    # Converter de milissegundos para segundos
    time_data = time_data.copy()
    time_data['TIMESTAMP_median'] = time_data['TIMESTAMP_median'] / 1000.0

    # Definir ordem das estratégias
    time_data['strategy'] = pd.Categorical(time_data['strategy'], categories=GPU_COMPARISON_STRATEGIES, ordered=True)

    # Agrupar por estratégia e algoritmo, calculando estatísticas
    total_time_data = time_data.groupby(['strategy', 'algorithm']).agg({
        'TIMESTAMP_median': ['mean', 'std', 'count']
    }).reset_index()
    
    # Renomear colunas
    total_time_data.columns = ['strategy', 'algorithm', 'mean', 'std', 'n']
    
    # Calcular intervalo de confiança
    total_time_data['se'] = total_time_data['std'] / np.sqrt(total_time_data['n'])
    total_time_data['ci_95'] = 1.96 * total_time_data['se']

    # Plot separado para STEEP
    steep_data = total_time_data[total_time_data['algorithm'] == 'STEEP']
    if not steep_data.empty:
        p_steep = (ggplot(steep_data, aes(x='strategy', y='mean')) +
                  geom_bar(stat='identity', fill='#4ECDC4', width=0.7) +
                  geom_errorbar(aes(ymin='mean-ci_95', ymax='mean+ci_95'), 
                               width=0.2, color='black', size=1) +
                  labs(title='Tempo Total de Execução - GPU vs GPU E/S Assíncrona - Algoritmo STEEP (IC 95%)',
                       x='Estratégia', y='Tempo Total (segundos)') +
                  scale_y_continuous(limits=(0, 2), breaks=[x/10 for x in range(0, 21, 2)]) +
                  theme_bw() +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(10, 6)))
        p_steep.save(output_dir / 'z-tempo_total_gpu_comparison_steep.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'z-tempo_total_gpu_comparison_steep.png'}")

    # Plot separado para SEBAL
    sebal_data = total_time_data[total_time_data['algorithm'] == 'SEBAL']
    if not sebal_data.empty:
        p_sebal = (ggplot(sebal_data, aes(x='strategy', y='mean')) +
                  geom_bar(stat='identity', fill='#FF6B6B', width=0.7) +
                  geom_errorbar(aes(ymin='mean-ci_95', ymax='mean+ci_95'), 
                               width=0.2, color='black', size=1) +
                  labs(title='Tempo Total de Execução - GPU vs GPU E/S Assíncrona - Algoritmo SEBAL (IC 95%)',
                       x='Estratégia', y='Tempo Total (segundos)') +
                  scale_y_continuous(limits=(0, 2), breaks=[x/10 for x in range(0, 21, 2)]) +
                  theme_bw() +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(10, 6)))
        p_sebal.save(output_dir / 'z-tempo_total_gpu_comparison_sebal.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'z-tempo_total_gpu_comparison_sebal.png'}")

def create_reading_comparison_plot(time_data, output_dir):
    """Cria plot de tempo de leitura comparando todas as estratégias"""
    print("Gerando plot de comparação de tempo de leitura...")

    # Extrair strategy e algorithm para comparação de leitura
    time_data = extract_strategy_and_algorithm_reading_comparison(time_data)
    
    # Converter de milissegundos para segundos
    time_data = time_data.copy()
    time_data['TIMESTAMP_median'] = time_data['TIMESTAMP_median'] / 1000.0

    # Definir ordem das estratégias
    strategy_order = ['CPU Serial', 'CPU Serial\nE/S assincrona', 'GPU', 'GPU E/S\nassincrona']
    time_data['strategy'] = pd.Categorical(time_data['strategy'], categories=strategy_order, ordered=True)

    # Agrupar por estratégia e algoritmo, calculando estatísticas
    reading_time_data = time_data.groupby(['strategy', 'algorithm']).agg({
        'TIMESTAMP_median': ['mean', 'std', 'count']
    }).reset_index()
    
    # Renomear colunas
    reading_time_data.columns = ['strategy', 'algorithm', 'mean', 'std', 'n']
    
    # Calcular intervalo de confiança
    reading_time_data['se'] = reading_time_data['std'] / np.sqrt(reading_time_data['n'])
    reading_time_data['ci_95'] = 1.96 * reading_time_data['se']

    # Plot separado para STEEP
    steep_data = reading_time_data[reading_time_data['algorithm'] == 'STEEP']
    if not steep_data.empty:
        p_steep = (ggplot(steep_data, aes(x='strategy', y='mean')) +
                  geom_bar(stat='identity', fill='#4ECDC4', width=0.7) +
                  geom_errorbar(aes(ymin='mean-ci_95', ymax='mean+ci_95'), 
                               width=0.2, color='black', size=1) +
                  labs(title='Tempo de Leitura - Todas as Estratégias - Algoritmo STEEP (IC 95%)',
                       x='Estratégia', y='Tempo de Leitura (segundos)') +
                  scale_y_continuous(limits=(0, 1.4), breaks=[x/10 for x in range(0, 15, 2)]) +
                  theme_bw() +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(12, 6)))
        p_steep.save(output_dir / 'reading_comparison_steep.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'reading_comparison_steep.png'}")

    # Plot separado para SEBAL
    sebal_data = reading_time_data[reading_time_data['algorithm'] == 'SEBAL']
    if not sebal_data.empty:
        p_sebal = (ggplot(sebal_data, aes(x='strategy', y='mean')) +
                  geom_bar(stat='identity', fill='#FF6B6B', width=0.7) +
                  geom_errorbar(aes(ymin='mean-ci_95', ymax='mean+ci_95'), 
                               width=0.2, color='black', size=1) +
                  labs(title='Tempo de Leitura - Todas as Estratégias - Algoritmo SEBAL (IC 95%)',
                       x='Estratégia', y='Tempo de Leitura (segundos)') +
                  scale_y_continuous(limits=(0, 1.4), breaks=[x/10 for x in range(0, 15, 2)]) +
                  theme_bw() +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(12, 6)))
        p_sebal.save(output_dir / 'reading_comparison_sebal.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'reading_comparison_sebal.png'}")

def create_reading_speedup_table(time_data, output_dir):
    """Cria tabela com tempos e speedups das estratégias de leitura"""
    print("Gerando tabela de speedup de leitura...")

    # Extrair strategy e algorithm para comparação de leitura
    time_data = extract_strategy_and_algorithm_reading_comparison(time_data)
    
    # Converter de milissegundos para segundos
    time_data = time_data.copy()
    time_data['TIMESTAMP_median'] = time_data['TIMESTAMP_median'] / 1000.0

    # Definir ordem das estratégias
    strategy_order = ['CPU Serial', 'CPU Serial\nE/S assincrona', 'GPU', 'GPU E/S\nassincrona']
    time_data['strategy'] = pd.Categorical(time_data['strategy'], categories=strategy_order, ordered=True)

    # Agrupar por estratégia e algoritmo, calculando estatísticas
    reading_time_data = time_data.groupby(['strategy', 'algorithm']).agg({
        'TIMESTAMP_median': ['mean', 'std', 'count']
    }).reset_index()
    
    # Renomear colunas
    reading_time_data.columns = ['strategy', 'algorithm', 'mean', 'std', 'n']
    
    # Calcular speedup para cada algoritmo
    def calculate_speedup(df_alg):
        # CPU Serial é a referência (speedup = 1)
        cpu_serial_time = df_alg[df_alg['strategy'] == 'CPU Serial']['mean'].iloc[0]
        df_alg['speedup'] = cpu_serial_time / df_alg['mean']
        return df_alg

    # Aplicar speedup para cada algoritmo
    sebal_data = reading_time_data[reading_time_data['algorithm'] == 'SEBAL'].copy()
    steep_data = reading_time_data[reading_time_data['algorithm'] == 'STEEP'].copy()
    
    if not sebal_data.empty:
        sebal_data = calculate_speedup(sebal_data)
        # Formatar para exibição SEBAL
        sebal_data['tempo_formatado'] = sebal_data['mean'].apply(lambda x: f"{x:.2f}")
        sebal_data['speedup_formatado'] = sebal_data['speedup'].apply(lambda x: f"{x:.2f}")
        
        print("\n=== TABELA DE SPEEDUP - SEBAL ===")
        print("Estratégia                    | Tempo (s) | Speedup")
        print("-" * 50)
        for _, row in sebal_data.iterrows():
            strategy = row['strategy'].replace('\n', ' ')
            tempo = row['tempo_formatado']
            speedup = row['speedup_formatado']
            print(f"{strategy:<30} | {tempo:>9} | {speedup:>7}")
    
    if not steep_data.empty:
        steep_data = calculate_speedup(steep_data)
        # Formatar para exibição STEEP
        steep_data['tempo_formatado'] = steep_data['mean'].apply(lambda x: f"{x:.2f}")
        steep_data['speedup_formatado'] = steep_data['speedup'].apply(lambda x: f"{x:.2f}")
        
        print("\n=== TABELA DE SPEEDUP - STEEP ===")
        print("Estratégia                    | Tempo (s) | Speedup")
        print("-" * 50)
        for _, row in steep_data.iterrows():
            strategy = row['strategy'].replace('\n', ' ')
            tempo = row['tempo_formatado']
            speedup = row['speedup_formatado']
            print(f"{strategy:<30} | {tempo:>9} | {speedup:>7}")
    
    # Combinar dados para salvar em CSV
    speedup_data = pd.concat([sebal_data, steep_data], ignore_index=True)
    
    # Salvar dados em CSV
    speedup_data.to_csv(output_dir / 'reading_speedup_table.csv', index=False)
    print(f"\nTabela salva em: {output_dir / 'reading_speedup_table.csv'}")

def load_time_data_phases_selected(results_dir):
    """Carrega dados de tempo (todas as fases) para CPU Serial Double e GPU CUDA Double"""
    all_data = []
    for strategy_algorithm in STRATEGY_ALGORITHMS:
        file_path = Path(results_dir) / strategy_algorithm / 'final-time.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"Carregado fases: {strategy_algorithm} - {len(df)} linhas")
        else:
            print(f"Arquivo não encontrado: {file_path}")
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

def load_time_data_phases_gpu_comparison(results_dir):
    """Carrega dados de tempo (todas as fases) para comparação GPU"""
    all_data = []
    for strategy_algorithm in STRATEGY_ALGORITHMS_GPU_COMPARISON:
        file_path = Path(results_dir) / strategy_algorithm / 'final-time.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"Carregado fases GPU comparison: {strategy_algorithm} - {len(df)} linhas")
        else:
            print(f"Arquivo não encontrado: {file_path}")
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

def _prepare_phases_for_double_only(time_data):
    """Prepara dados (combina P5/P6, segundos, mapeia nomes) apenas para CPU/GPU Double."""
    # Extrair strategy/algorithm
    df = extract_strategy_and_algorithm(time_data)
    # Filtrar apenas CPU Serial Double e GPU CUDA Double
    df = df[df['strategy'].isin(SELECTED_STRATEGIES)].copy()

    # Combinar P5 e P6
    p5_p6 = df[df['PHASE'].isin(['P5_COPY_HOST', 'P6_SAVE_PRODS'])]
    if not p5_p6.empty:
        combo = p5_p6.groupby(['strategy_algorithm', 'strategy', 'algorithm'])['TIMESTAMP_median'].sum().reset_index()
        combo['PHASE'] = 'P5_P6_COPY_SAVE'
        df = df[~df['PHASE'].isin(['P5_COPY_HOST', 'P6_SAVE_PRODS'])]
        df = pd.concat([df, combo], ignore_index=True)

    # Converter ms -> s
    df = df.copy()
    df['TIMESTAMP_median'] = df['TIMESTAMP_median'] / 1000.0

    # Renomear estratégias para rótulo (usar macros diretamente)
    strategy_mapping = {
        'CPU Serial E/S assincrona': CPU_SERIAL_DOUBLE,
        'GPU E/S assincrona': GPU_CUDA_DOUBLE,
    }
    df['strategy_plot'] = df['strategy'].map(strategy_mapping)

    # Mapeamento PT
    phase_mapping = {
        'P0_READ_INPUT': 'Leitura dos\ndados de entrada',
        'P1_INITIAL_PROD': 'Produtos\nIniciais',
        'P2_PIXEL_SEL': 'Seleção de\nPixels',
        'P3_RAH': 'Produtos\nIntermediários',
        'P4_FINAL_PROD': 'Produtos\nFinais',
        'P5_P6_COPY_SAVE': 'Escrita dos\ndados de saída',
        'P_TOTAL': 'Tempo\nTotal'
    }
    phase_order_pt = [
        'Leitura dos\ndados de entrada', 'Produtos\nIniciais', 'Seleção de\nPixels',
        'Produtos\nIntermediários', 'Produtos\nFinais', 'Escrita dos\ndados de saída', 'Tempo\nTotal'
    ]
    return df, phase_mapping, phase_order_pt

def _prepare_phases_for_gpu_comparison(time_data):
    """Prepara dados (combina P5/P6, segundos, mapeia nomes) para comparação GPU."""
    # Extrair strategy/algorithm para comparação GPU
    df = extract_strategy_and_algorithm_gpu_comparison(time_data)
    # Filtrar apenas estratégias GPU
    df = df[df['strategy'].isin(GPU_COMPARISON_STRATEGIES)].copy()

    # Combinar P5 e P6
    p5_p6 = df[df['PHASE'].isin(['P5_COPY_HOST', 'P6_SAVE_PRODS'])]
    if not p5_p6.empty:
        combo = p5_p6.groupby(['strategy_algorithm', 'strategy', 'algorithm'])['TIMESTAMP_median'].sum().reset_index()
        combo['PHASE'] = 'P5_P6_COPY_SAVE'
        df = df[~df['PHASE'].isin(['P5_COPY_HOST', 'P6_SAVE_PRODS'])]
        df = pd.concat([df, combo], ignore_index=True)

    # Converter ms -> s
    df = df.copy()
    df['TIMESTAMP_median'] = df['TIMESTAMP_median'] / 1000.0

    # Renomear estratégias para rótulo
    strategy_mapping = {
        'GPU': GPU_NORMAL,
        'GPU E/S assincrona': GPU_ASYNC,
    }
    df['strategy_plot'] = df['strategy'].map(strategy_mapping)

    # Mapeamento PT
    phase_mapping = {
        'P0_READ_INPUT': 'Leitura dos\ndados de entrada',
        'P1_INITIAL_PROD': 'Produtos\nIniciais',
        'P2_PIXEL_SEL': 'Seleção de\nPixels',
        'P3_RAH': 'Produtos\nIntermediários',
        'P4_FINAL_PROD': 'Produtos\nFinais',
        'P5_P6_COPY_SAVE': 'Escrita dos\ndados de saída',
        'P_TOTAL': 'Tempo\nTotal'
    }
    phase_order_pt = [
        'Leitura dos\ndados de entrada', 'Produtos\nIniciais', 'Seleção de\nPixels',
        'Produtos\nIntermediários', 'Produtos\nFinais', 'Escrita dos\ndados de saída', 'Tempo\nTotal'
    ]
    return df, phase_mapping, phase_order_pt

def _build_heatmap_df(df, algorithm, phase_mapping, phase_order_pt, y_key='strategy_plot'):
    """Monta o dataframe do heatmap com LABEL tempo (perc) e ordenação de fases."""
    df_alg = df[df['algorithm'] == algorithm].copy()
    if df_alg.empty:
        return None

    # Totais por linha (para percentual)
    totals = df_alg[df_alg['PHASE'] == 'P_TOTAL'].groupby([y_key])['TIMESTAMP_median'].mean().reset_index()
    totals = totals.rename(columns={'TIMESTAMP_median': 'TOTAL_SEC'})

    pivot = df_alg.pivot_table(values='TIMESTAMP_median', index=y_key, columns='PHASE', aggfunc='mean')
    heatmap = pivot.reset_index().melt(id_vars=[y_key], var_name='PHASE', value_name='TIMESTAMP_median')
    heatmap = heatmap.merge(totals, on=y_key, how='left')

    heatmap['PHASE_PT'] = heatmap['PHASE'].map(phase_mapping)
    heatmap['PHASE_PT'] = pd.Categorical(heatmap['PHASE_PT'], categories=phase_order_pt, ordered=True)
    heatmap = heatmap.dropna(subset=['TIMESTAMP_median', 'PHASE_PT', 'TOTAL_SEC'])

    heatmap['PERC'] = (heatmap['TIMESTAMP_median'] / heatmap['TOTAL_SEC']) * 100.0
    heatmap.loc[heatmap['PHASE'] == 'P_TOTAL', 'PERC'] = 100.0

    # Formatação: tempo 3 casas decimais, percentual 2 casas decimais, vírgula
    heatmap['TEMPO_STR'] = heatmap['TIMESTAMP_median'].apply(lambda x: f"{x:.3f}".replace('.', ','))
    heatmap['PERC_STR'] = heatmap['PERC'].apply(lambda x: f"{x:.2f}".replace('.', ','))
    heatmap['LABEL'] = heatmap['TEMPO_STR'] + ' (' + heatmap['PERC_STR'] + '%)'
    return heatmap

def create_heatmap_single_line_double(time_data, output_dir, algorithm, strategy_label, filename):
    """Heatmap de uma única linha (CPU Serial ou GPU) para um algoritmo (STEEP/SEBAL)."""
    df, phase_mapping, phase_order_pt = _prepare_phases_for_double_only(time_data)
    # Filtrar estratégia desejada
    df = df[df['strategy_plot'] == strategy_label].copy()
    if df.empty:
        print(f"  Aviso: Sem dados para {strategy_label} {algorithm}")
        return

    heatmap = _build_heatmap_df(df, algorithm, phase_mapping, phase_order_pt, y_key='strategy_plot')
    if heatmap is None or heatmap.empty:
        print(f"  Aviso: Sem dados para {strategy_label} {algorithm}")
        return

    p = (ggplot(heatmap, aes(x='PHASE_PT', y='strategy_plot', fill='TIMESTAMP_median')) +
         geom_tile() +
         geom_text(aes(label='LABEL'), size=11, color='black') +
         scale_fill_gradient(low='#F5F5F5', high=('#4ECDC4' if algorithm == 'STEEP' else '#FF6B6B'), name='Tempo (s)') +
         labs(title=f'Heatmap de Tempos - {algorithm} ({strategy_label})', x='Fase', y='Estratégia') +
         theme_bw() +
         theme(axis_text_x=element_text(rotation=45, hjust=1, size=11),
               axis_text_y=element_text(size=11),
               axis_title_x=element_text(size=11),
               axis_title_y=element_text(size=11),
               plot_title=element_text(size=12),
               legend_text=element_text(size=11),
               legend_title=element_text(size=11),
               figure_size=(12, 2.5)))
    # Adicionar prefixo y- ao nome do arquivo
    filename_with_prefix = f"y-{filename}"
    p.save(output_dir / filename_with_prefix, dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / filename_with_prefix}")

def create_combined_sebal_heatmap(time_data, output_dir):
    """Cria plot combinado dos heatmaps SEBAL CPU e GPU em grid vertical"""
    print("Gerando plot combinado dos heatmaps SEBAL CPU e GPU...")
    
    df, phase_mapping, phase_order_pt = _prepare_phases_for_double_only(time_data)
    
    # Filtrar apenas dados SEBAL
    df_sebal = df[df['algorithm'] == 'SEBAL'].copy()
    if df_sebal.empty:
        print("  Aviso: Sem dados SEBAL encontrados")
        return
    

    # Criar heatmaps separados para CPU e GPU
    cpu_data = df_sebal[df_sebal['strategy'] == 'CPU Serial E/S assincrona'].copy()
    gpu_data = df_sebal[df_sebal['strategy'] == 'GPU E/S assincrona'].copy()
    
    if cpu_data.empty or gpu_data.empty:
        print("  Aviso: Dados insuficientes para CPU ou GPU SEBAL")
        return
    
    # Construir dataframes dos heatmaps
    cpu_heatmap = _build_heatmap_df(cpu_data, 'SEBAL', phase_mapping, phase_order_pt, y_key='strategy_plot')
    gpu_heatmap = _build_heatmap_df(gpu_data, 'SEBAL', phase_mapping, phase_order_pt, y_key='strategy_plot')
    
    if cpu_heatmap is None or gpu_heatmap is None:
        print("  Aviso: Erro ao construir heatmaps")
        return
    
    # Combinar os dataframes e adicionar identificador
    cpu_heatmap['plot_type'] = 'CPU Serial E/S assincrona'
    gpu_heatmap['plot_type'] = 'GPU E/S assincrona'
    combined_heatmap = pd.concat([cpu_heatmap, gpu_heatmap], ignore_index=True)
    
    # Definir ordem dos plots
    combined_heatmap['plot_type'] = pd.Categorical(combined_heatmap['plot_type'], 
                                                  categories=['CPU Serial E/S assincrona', 'GPU E/S assincrona'], ordered=True)
    
    # Criar o plot combinado
    p = (ggplot(combined_heatmap, aes(x='PHASE_PT', y='strategy_plot', fill='TIMESTAMP_median')) +
         geom_tile() +
         geom_text(aes(label='LABEL'), size=11, color='black') +
         scale_fill_gradient(low='#F5F5F5', high='#FF6B6B', name='Tempo (s)') +
         labs(title='Heatmap de Tempos - SEBAL (CPU vs GPU)', 
              x='Fase', y='Estratégia') +
         theme_bw() +
         theme(axis_text_x=element_text(rotation=45, hjust=1, size=11), 
               axis_text_y=element_text(size=11),
               axis_title_x=element_text(size=11),
               axis_title_y=element_text(size=11),
               figure_size=(12, 5),
               strip_text=element_text(size=12, face='bold'),
               plot_title=element_text(size=14, face='bold'),
               legend_text=element_text(size=11),
               legend_title=element_text(size=11)) +
         facet_wrap('~plot_type', ncol=1, scales='free_y'))
    
    p.save(output_dir / 'y-heatmap_sebal_cpu_gpu_combined.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'y-heatmap_sebal_cpu_gpu_combined.png'}")

def create_combined_steep_heatmap(time_data, output_dir):
    """Cria plot combinado dos heatmaps STEEP CPU e GPU em grid vertical"""
    print("Gerando plot combinado dos heatmaps STEEP CPU e GPU...")
    
    df, phase_mapping, phase_order_pt = _prepare_phases_for_double_only(time_data)
    
    # Filtrar apenas dados STEEP
    df_steep = df[df['algorithm'] == 'STEEP'].copy()
    if df_steep.empty:
        print("  Aviso: Sem dados STEEP encontrados")
        return
    
    # Criar heatmaps separados para CPU e GPU
    cpu_data = df_steep[df_steep['strategy'] == 'CPU Serial E/S assincrona'].copy()
    gpu_data = df_steep[df_steep['strategy'] == 'GPU E/S assincrona'].copy()
    
    if cpu_data.empty or gpu_data.empty:
        print("  Aviso: Dados insuficientes para CPU ou GPU STEEP")
        return
    
    # Construir dataframes dos heatmaps
    cpu_heatmap = _build_heatmap_df(cpu_data, 'STEEP', phase_mapping, phase_order_pt, y_key='strategy_plot')
    gpu_heatmap = _build_heatmap_df(gpu_data, 'STEEP', phase_mapping, phase_order_pt, y_key='strategy_plot')
    
    if cpu_heatmap is None or gpu_heatmap is None:
        print("  Aviso: Erro ao construir heatmaps")
        return
    
    # Combinar os dataframes e adicionar identificador
    cpu_heatmap['plot_type'] = 'CPU Serial E/S assincrona'
    gpu_heatmap['plot_type'] = 'GPU E/S assincrona'
    combined_heatmap = pd.concat([cpu_heatmap, gpu_heatmap], ignore_index=True)
    
    # Definir ordem dos plots
    combined_heatmap['plot_type'] = pd.Categorical(combined_heatmap['plot_type'], 
                                                  categories=['CPU Serial E/S assincrona', 'GPU E/S assincrona'], ordered=True)
    
    # Criar o plot combinado
    p = (ggplot(combined_heatmap, aes(x='PHASE_PT', y='strategy_plot', fill='TIMESTAMP_median')) +
         geom_tile() +
         geom_text(aes(label='LABEL'), size=11, color='black') +
         scale_fill_gradient(low='#F5F5F5', high='#4ECDC4', name='Tempo (s)') +
         labs(title='Heatmap de Tempos - STEEP (CPU vs GPU)', 
              x='Fase', y='Estratégia') +
         theme_bw() +
         theme(axis_text_x=element_text(rotation=45, hjust=1, size=11), 
               axis_text_y=element_text(size=11),
               axis_title_x=element_text(size=11),
               axis_title_y=element_text(size=11),
               figure_size=(12, 5),
               strip_text=element_text(size=12, face='bold'),
               plot_title=element_text(size=14, face='bold'),
               legend_text=element_text(size=11),
               legend_title=element_text(size=11)) +
         facet_wrap('~plot_type', ncol=1, scales='free_y'))
    
    p.save(output_dir / 'y-heatmap_steep_cpu_gpu_combined.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'y-heatmap_steep_cpu_gpu_combined.png'}")

def create_combined_sebal_heatmap_gpu_comparison(time_data, output_dir):
    """Cria plot combinado dos heatmaps SEBAL GPU vs GPU E/S assíncrona"""
    print("Gerando plot combinado dos heatmaps SEBAL GPU vs GPU E/S assíncrona...")
    
    df, phase_mapping, phase_order_pt = _prepare_phases_for_gpu_comparison(time_data)
    
    # Filtrar apenas dados SEBAL
    df_sebal = df[df['algorithm'] == 'SEBAL'].copy()
    if df_sebal.empty:
        print("  Aviso: Sem dados SEBAL encontrados")
        return
    
    # Criar heatmaps separados para GPU normal e GPU assíncrona
    gpu_normal_data = df_sebal[df_sebal['strategy'] == 'GPU'].copy()
    gpu_async_data = df_sebal[df_sebal['strategy'] == 'GPU E/S assincrona'].copy()
    
    if gpu_normal_data.empty or gpu_async_data.empty:
        print("  Aviso: Dados insuficientes para GPU comparison SEBAL")
        return
    
    # Construir dataframes dos heatmaps
    gpu_normal_heatmap = _build_heatmap_df(gpu_normal_data, 'SEBAL', phase_mapping, phase_order_pt, y_key='strategy_plot')
    gpu_async_heatmap = _build_heatmap_df(gpu_async_data, 'SEBAL', phase_mapping, phase_order_pt, y_key='strategy_plot')
    
    if gpu_normal_heatmap is None or gpu_async_heatmap is None:
        print("  Aviso: Erro ao construir heatmaps")
        return
    
    # Combinar os dataframes e adicionar identificador
    gpu_normal_heatmap['plot_type'] = 'GPU'
    gpu_async_heatmap['plot_type'] = 'GPU E/S assincrona'
    combined_heatmap = pd.concat([gpu_normal_heatmap, gpu_async_heatmap], ignore_index=True)
    
    # Definir ordem dos plots
    combined_heatmap['plot_type'] = pd.Categorical(combined_heatmap['plot_type'], 
                                                  categories=['GPU', 'GPU E/S assincrona'], ordered=True)
    
    # Criar o plot combinado
    p = (ggplot(combined_heatmap, aes(x='PHASE_PT', y='strategy_plot', fill='TIMESTAMP_median')) +
         geom_tile() +
         geom_text(aes(label='LABEL'), size=11, color='black') +
         scale_fill_gradient(low='#F5F5F5', high='#FF6B6B', name='Tempo (s)') +
         labs(title='Heatmap de Tempos - SEBAL (GPU vs GPU E/S Assíncrona)', 
              x='Fase', y='Estratégia') +
         theme_bw() +
         theme(axis_text_x=element_text(rotation=45, hjust=1, size=11), 
               axis_text_y=element_text(size=11),
               axis_title_x=element_text(size=11),
               axis_title_y=element_text(size=11),
               figure_size=(12, 5),
               strip_text=element_text(size=12, face='bold'),
               plot_title=element_text(size=14, face='bold'),
               legend_text=element_text(size=11),
               legend_title=element_text(size=11)) +
         facet_wrap('~plot_type', ncol=1, scales='free_y'))
    
    p.save(output_dir / 'z-heatmap_sebal_gpu_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'z-heatmap_sebal_gpu_comparison.png'}")

def create_combined_steep_heatmap_gpu_comparison(time_data, output_dir):
    """Cria plot combinado dos heatmaps STEEP GPU vs GPU E/S assíncrona"""
    print("Gerando plot combinado dos heatmaps STEEP GPU vs GPU E/S assíncrona...")
    
    df, phase_mapping, phase_order_pt = _prepare_phases_for_gpu_comparison(time_data)
    
    # Filtrar apenas dados STEEP
    df_steep = df[df['algorithm'] == 'STEEP'].copy()
    if df_steep.empty:
        print("  Aviso: Sem dados STEEP encontrados")
        return
    
    # Criar heatmaps separados para GPU normal e GPU assíncrona
    gpu_normal_data = df_steep[df_steep['strategy'] == 'GPU'].copy()
    gpu_async_data = df_steep[df_steep['strategy'] == 'GPU E/S assincrona'].copy()
    
    if gpu_normal_data.empty or gpu_async_data.empty:
        print("  Aviso: Dados insuficientes para GPU comparison STEEP")
        return
    
    # Construir dataframes dos heatmaps
    gpu_normal_heatmap = _build_heatmap_df(gpu_normal_data, 'STEEP', phase_mapping, phase_order_pt, y_key='strategy_plot')
    gpu_async_heatmap = _build_heatmap_df(gpu_async_data, 'STEEP', phase_mapping, phase_order_pt, y_key='strategy_plot')
    
    if gpu_normal_heatmap is None or gpu_async_heatmap is None:
        print("  Aviso: Erro ao construir heatmaps")
        return
    
    # Combinar os dataframes e adicionar identificador
    gpu_normal_heatmap['plot_type'] = 'GPU'
    gpu_async_heatmap['plot_type'] = 'GPU E/S assincrona'
    combined_heatmap = pd.concat([gpu_normal_heatmap, gpu_async_heatmap], ignore_index=True)
    
    # Definir ordem dos plots
    combined_heatmap['plot_type'] = pd.Categorical(combined_heatmap['plot_type'], 
                                                  categories=['GPU', 'GPU E/S assincrona'], ordered=True)
    
    # Criar o plot combinado
    p = (ggplot(combined_heatmap, aes(x='PHASE_PT', y='strategy_plot', fill='TIMESTAMP_median')) +
         geom_tile() +
         geom_text(aes(label='LABEL'), size=11, color='black') +
         scale_fill_gradient(low='#F5F5F5', high='#4ECDC4', name='Tempo (s)') +
         labs(title='Heatmap de Tempos - STEEP (GPU vs GPU E/S Assíncrona)', 
              x='Fase', y='Estratégia') +
         theme_bw() +
         theme(axis_text_x=element_text(rotation=45, hjust=1, size=11), 
               axis_text_y=element_text(size=11),
               axis_title_x=element_text(size=11),
               axis_title_y=element_text(size=11),
               figure_size=(12, 5),
               strip_text=element_text(size=12, face='bold'),
               plot_title=element_text(size=14, face='bold'),
               legend_text=element_text(size=11),
               legend_title=element_text(size=11)) +
         facet_wrap('~plot_type', ncol=1, scales='free_y'))
    
    p.save(output_dir / 'z-heatmap_steep_gpu_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'z-heatmap_steep_gpu_comparison.png'}")

def main():
    """Função principal"""
    results_dir = Path('summarized_results')
    output_dir = Path('plots')
    
    if not results_dir.exists():
        print(f"Erro: Diretório {results_dir} não encontrado!")
        return
    
    # Criar diretório de saída se não existir
    output_dir.mkdir(exist_ok=True)
    
    print(f"Carregando dados de {results_dir}")
    print(f"Salvando plots em {output_dir}")
    
    # Carregar dados para plot CPU vs GPU Double
    time_data = load_time_data(results_dir)
    
    if time_data is None:
        print("Erro: Nenhum dado de tempo encontrado!")
        return
    
    print(f"Dados carregados para CPU vs GPU Double: {len(time_data)} registros")
    
    # Gerar plot CPU vs GPU Double
    create_selected_strategies_plot(time_data, output_dir)
    
    # Heatmaps de linha única para CPU Serial e GPU
    time_phases_selected = load_time_data_phases_selected(results_dir)
    if time_phases_selected is not None:
        # Linhas únicas
        create_heatmap_single_line_double(time_phases_selected, output_dir, 'STEEP', 'CPU Serial', 'heatmap_steep_cpu_serial.png')
        create_heatmap_single_line_double(time_phases_selected, output_dir, 'STEEP', 'GPU', 'heatmap_steep_gpu.png')
        create_heatmap_single_line_double(time_phases_selected, output_dir, 'SEBAL', 'CPU Serial', 'heatmap_sebal_cpu_serial.png')
        create_heatmap_single_line_double(time_phases_selected, output_dir, 'SEBAL', 'GPU', 'heatmap_sebal_gpu.png')
        
        # Plot combinado SEBAL CPU e GPU
        create_combined_sebal_heatmap(time_phases_selected, output_dir)
        
        # Plot combinado STEEP CPU e GPU
        create_combined_steep_heatmap(time_phases_selected, output_dir)

    # Carregar dados para comparação GPU
    time_data_gpu_comparison = load_time_data_gpu_comparison(results_dir)
    
    if time_data_gpu_comparison is not None:
        print(f"Dados carregados para comparação GPU: {len(time_data_gpu_comparison)} registros")
        
        # Gerar plots de comparação GPU
        create_gpu_comparison_plot(time_data_gpu_comparison, output_dir)
        
        # Carregar dados das fases para comparação GPU
        time_phases_gpu_comparison = load_time_data_phases_gpu_comparison(results_dir)
        if time_phases_gpu_comparison is not None:
            # Plots combinados de comparação GPU
            create_combined_sebal_heatmap_gpu_comparison(time_phases_gpu_comparison, output_dir)
            create_combined_steep_heatmap_gpu_comparison(time_phases_gpu_comparison, output_dir)

    # Carregar dados para comparação de leitura
    time_data_reading_comparison = load_time_data_reading_comparison(results_dir)
    
    if time_data_reading_comparison is not None:
        print(f"Dados carregados para comparação de leitura: {len(time_data_reading_comparison)} registros")
        
        # Gerar plots de comparação de leitura
        create_reading_comparison_plot(time_data_reading_comparison, output_dir)
        
        # Gerar tabela de speedup de leitura
        create_reading_speedup_table(time_data_reading_comparison, output_dir)

    print(f"\nPlots gerados em: {output_dir}")

if __name__ == '__main__':
    main()
