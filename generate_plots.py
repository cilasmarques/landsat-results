#!/usr/bin/env python3
"""
Script para gerar plots dos dados de tempo e utilização de recursos das estratégias Landsat.

Este script cria visualizações usando plotnine (ggplot para Python) para:
1. Tempos de processamento das fases principais
2. Utilização de CPU e GPU por fase
3. Comparações entre estratégias e algoritmos
4. Análise de eficiência por fase

Adaptado para a estrutura output-csv-lsd-0707.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from plotnine import *
import warnings
warnings.filterwarnings('ignore')

def set_strategy_order(df):
    """Define a ordem das estratégias nos plots"""
    # Nova ordem baseada na estrutura desejada
    strategy_order = [
        'CPU Serial Double',
        'CPU Serial Float',
        'CPU 2 Threads Float',
        'CPU 4 Threads Float',
        'CPU 8 Threads Float',
        'CPU 12 Threads Float',
        'GPU CUDA Double',
        'GPU CUDA Float',
        'GPU CUDA Stream Float'
    ]
    df['strategy'] = pd.Categorical(df['strategy'], categories=strategy_order, ordered=True)
    return df

def extract_strategy_and_algorithm(df):
    """Extrai estratégia e algoritmo da coluna strategy_algorithm"""
    df = df.copy()
    
    # Mapear estratégias e algoritmos
    strategy_mapping = {
        'serial-double': 'CPU Serial Double',
        'serial-float': 'CPU Serial Float',
        'parallel2-float': 'CPU 2 Threads Float',
        'parallel4-float': 'CPU 4 Threads Float', 
        'parallel8-float': 'CPU 8 Threads Float',
        'parallel12-float': 'CPU 12 Threads Float',
        'kernels-double-fm': 'GPU CUDA Double',
        'kernels-float-r': 'GPU CUDA Float',
        'kernels-float-st': 'GPU CUDA Stream Float'
    }
    
    algorithm_mapping = {
        'sebal': 'SEBAL',
        'steep': 'STEEP',
    }
    
    # Extrair estratégia e algoritmo
    # Para estratégias serial e parallel: strategy-precision-algorithm
    # Para estratégias GPU: kernels-precision-variant-algorithm
    
    # Primeiro, vamos criar uma coluna temporária para estratégia+precisão
    df['strategy_precision'] = df['strategy_algorithm'].str.rsplit('-', n=1, expand=True)[0]
    df['algorithm'] = df['strategy_algorithm'].str.rsplit('-', n=1, expand=True)[1]
    
    # Aplicar mapeamentos
    df['strategy'] = df['strategy_precision'].map(strategy_mapping)
    df['algorithm'] = df['algorithm'].map(algorithm_mapping)
    
    # Remover coluna temporária
    df = df.drop('strategy_precision', axis=1)
    
    return df

def load_time_data(results_dir):
    """Carrega todos os dados de tempo dos arquivos final-time.csv"""
    all_data = []
    
    # Lista de estratégias/algoritmos baseada na estrutura atual
    strategy_algorithms = [
        'serial-double-sebal', 'serial-double-steep',
        'serial-float-sebal', 'serial-float-steep',
        'parallel2-float-sebal', 'parallel2-float-steep',
        'parallel4-float-sebal',  'parallel4-float-steep',
        'parallel8-float-sebal', 'parallel8-float-steep', 
        'parallel12-float-sebal', 'parallel12-float-steep',
        'kernels-double-fm-sebal', 'kernels-double-fm-steep',
        'kernels-float-r-sebal', 'kernels-float-r-steep',
        'kernels-float-st-sebal', 'kernels-float-st-steep',
    ]
    
    # Fases principais que queremos analisar
    main_phases = ['P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 
                   'P3_RAH', 'P4_FINAL_PROD', 'P5_COPY_HOST', 'P6_SAVE_PRODS', 'P_TOTAL']
    
    for strategy_algorithm in strategy_algorithms:
        file_path = Path(results_dir) / strategy_algorithm / 'final-time.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            # Filtra apenas as fases principais
            df = df[df['PHASE'].isin(main_phases)]
            all_data.append(df)
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    return None

def load_resource_data(results_dir, resource_type):
    """Carrega dados de CPU ou GPU"""
    all_data = []
    
    # Lista de estratégias/algoritmos
    strategy_algorithms = [
        'serial-double-sebal', 'serial-double-steep',
        'serial-float-sebal', 'serial-float-steep',
        'parallel2-float-sebal', 'parallel2-float-steep',
        'parallel4-float-sebal',  'parallel4-float-steep',
        'parallel8-float-sebal', 'parallel8-float-steep', 
        'parallel12-float-sebal', 'parallel12-float-steep',
        'kernels-double-fm-sebal', 'kernels-double-fm-steep',
        'kernels-float-r-sebal', 'kernels-float-r-steep',
        'kernels-float-st-sebal', 'kernels-float-st-steep',
    ]
    
    for strategy_algorithm in strategy_algorithms:
        file_path = Path(results_dir) / strategy_algorithm / f'{resource_type}-time.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['resource_type'] = resource_type
            all_data.append(df)
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    return None

def load_resource_by_phase_data(results_dir, resource_type):
    """Carrega dados de CPU ou GPU mapeados por fase"""
    all_data = []
    
    # Lista de estratégias/algoritmos
    strategy_algorithms = [
        'serial-double-sebal', 'serial-double-steep',
        'serial-float-sebal', 'serial-float-steep',
        'parallel2-float-sebal', 'parallel2-float-steep',
        'parallel4-float-sebal',  'parallel4-float-steep',
        'parallel8-float-sebal', 'parallel8-float-steep', 
        'parallel12-float-sebal', 'parallel12-float-steep',
        'kernels-double-fm-sebal', 'kernels-double-fm-steep',
        'kernels-float-r-sebal', 'kernels-float-r-steep',
        'kernels-float-st-sebal', 'kernels-float-st-steep',
    ]
    
    for strategy_algorithm in strategy_algorithms:
        file_path = Path(results_dir) / strategy_algorithm / f'{resource_type}-by-phase.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['resource_type'] = resource_type
            all_data.append(df)
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    return None

def load_ssd_io_data(results_dir):
    """Carrega dados de SSD I/O"""
    all_data = []
    
    strategy_algorithms = [
        'serial-double-sebal', 'serial-double-steep',
        'serial-float-sebal', 'serial-float-steep',
        'parallel2-float-sebal', 'parallel2-float-steep',
        'parallel4-float-sebal',  'parallel4-float-steep',
        'parallel8-float-sebal', 'parallel8-float-steep', 
        'parallel12-float-sebal', 'parallel12-float-steep',
        'kernels-double-fm-sebal', 'kernels-double-fm-steep',
        'kernels-float-r-sebal', 'kernels-float-r-steep',
        'kernels-float-st-sebal', 'kernels-float-st-steep',
    ]
    
    for strategy_algorithm in strategy_algorithms:
        file_path = Path(results_dir) / strategy_algorithm / 'ssd-io-time.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            all_data.append(df)
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    return None

def load_cpu_power_data(results_dir):
    """Carrega dados de CPU Power"""
    all_data = []
    
    strategy_algorithms = [
        'serial-double-sebal', 'serial-double-steep',
        'serial-float-sebal', 'serial-float-steep',
        'parallel2-float-sebal', 'parallel2-float-steep',
        'parallel4-float-sebal',  'parallel4-float-steep',
        'parallel8-float-sebal', 'parallel8-float-steep', 
        'parallel12-float-sebal', 'parallel12-float-steep',
        'kernels-double-fm-sebal', 'kernels-double-fm-steep',
        'kernels-float-r-sebal', 'kernels-float-r-steep',
        'kernels-float-st-sebal', 'kernels-float-st-steep',
    ]
    
    for strategy_algorithm in strategy_algorithms:
        file_path = Path(results_dir) / strategy_algorithm / 'cpu-power-time.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            all_data.append(df)
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    return None

def create_time_plots(time_data, output_dir):
    """Cria plots de tempo de processamento"""
    print("Gerando plots de tempo de processamento...")
    
    # Extrair estratégia e algoritmo
    time_data = extract_strategy_and_algorithm(time_data)
    
    # Aplicar ordem das estratégias
    time_data = set_strategy_order(time_data)
    
    # Combinar P5 e P6 em uma única fase
    time_data_combined = time_data.copy()
    p5_p6_data = time_data_combined[time_data_combined['PHASE'].isin(['P5_COPY_HOST', 'P6_SAVE_PRODS'])]
    if not p5_p6_data.empty:
        # Agrupar por strategy_algorithm e somar os tempos de P5 e P6
        p5_p6_combined = p5_p6_data.groupby(['strategy_algorithm', 'strategy', 'algorithm'])['TIMESTAMP_median'].sum().reset_index()
        p5_p6_combined['PHASE'] = 'P5_P6_COPY_SAVE'
        
        # Remover P5 e P6 originais e adicionar a combinação
        time_data_combined = time_data_combined[~time_data_combined['PHASE'].isin(['P5_COPY_HOST', 'P6_SAVE_PRODS'])]
        time_data_combined = pd.concat([time_data_combined, p5_p6_combined], ignore_index=True)
    
    # Plot 1: Comparação de tempos por fase e estratégia (STEEP)
    df_steep = time_data_combined[time_data_combined['algorithm'] == 'STEEP'].copy()
    
    p1 = (ggplot(df_steep, aes(x='PHASE', y='TIMESTAMP_median', fill='strategy')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Tempo de Processamento por Fase - Algoritmo STEEP',
               x='Fase', y='Tempo (milissegundos)', fill='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(12, 8),
                legend_position='none'))
    
    p1.save(output_dir / 'tempo_por_fase_steep.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'tempo_por_fase_steep.png'}")
    
    # Plot 2: Comparação de tempos por fase e estratégia (SEBAL)
    df_sebal = time_data_combined[time_data_combined['algorithm'] == 'SEBAL'].copy()
    
    p2 = (ggplot(df_sebal, aes(x='PHASE', y='TIMESTAMP_median', fill='strategy')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Tempo de Processamento por Fase - Algoritmo SEBAL',
               x='Fase', y='Tempo (milissegundos)', fill='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(12, 8),
                legend_position='none'))
    
    p2.save(output_dir / 'tempo_por_fase_sebal.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'tempo_por_fase_sebal.png'}")
    
    # Plot 3: Comparação entre algoritmos para cada estratégia
    p3 = (ggplot(time_data_combined, aes(x='PHASE', y='TIMESTAMP_median', fill='algorithm')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Comparação de Tempos entre Algoritmos por Estratégia',
               x='Fase', y='Tempo (milissegundos)', fill='Algoritmo') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(14, 10)))
    
    p3.save(output_dir / 'comparacao_algoritmos.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'comparacao_algoritmos.png'}")
    
    # Plot 4: Heatmap de tempos combinado (STEEP em cima, SEBAL embaixo)
    heatmap_data_combined = []
    for algorithm in ['STEEP', 'SEBAL']:
        df_alg = time_data_combined[time_data_combined['algorithm'] == algorithm].copy()
        pivot_data = df_alg.pivot_table(
            values='TIMESTAMP_median', 
            index='strategy', 
            columns='PHASE', 
            aggfunc='mean'
        )
        heatmap_data = pivot_data.reset_index().melt(
            id_vars=['strategy'], 
            var_name='PHASE', 
            value_name='TIMESTAMP_median'
        )
        heatmap_data['algorithm_name'] = algorithm
        heatmap_data_combined.append(heatmap_data)
    combined_heatmap_data = pd.concat(heatmap_data_combined, ignore_index=True)
    
    # Definir cores para as fases
    phase_colors = {
        'P0_READ_INPUT': '#FF8080',
        'P1_INITIAL_PROD': '#FFB347', 
        'P2_PIXEL_SEL': '#FFFF66',
        'P3_RAH': '#90EE90',
        'P4_FINAL_PROD': '#8080FF',
        'P5_P6_COPY_SAVE': '#9B7CB9',
        'P_TOTAL': '#B97C7C'
    }
    phase_order = [
        'P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 
        'P3_RAH', 'P4_FINAL_PROD', 'P5_P6_COPY_SAVE', 'P_TOTAL'
    ]
    combined_heatmap_data['PHASE'] = pd.Categorical(combined_heatmap_data['PHASE'], categories=phase_order, ordered=True)
    
    p4 = (ggplot(combined_heatmap_data, aes(x='PHASE', y='strategy', fill='TIMESTAMP_median')) +
          geom_tile() +
          geom_text(aes(label=combined_heatmap_data['TIMESTAMP_median'].round(1)), 
                   size=8, color='black') +
          scale_fill_gradient(low='#F5F5F5', high='#CCCCCC', name='Tempo (ms)') +
          facet_wrap('~algorithm_name', ncol=1, scales='free_y') +
          labs(title='Heatmap de Tempos de Processamento por Algoritmo',
               x='Fase', y='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(14, 10)))
    p4.save(output_dir / 'heatmap_tempos_combinado.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'heatmap_tempos_combinado.png'}")
    
    # Plot 5: Heatmaps individuais por algoritmo
    for algorithm in ['STEEP', 'SEBAL']:
        df_alg = time_data_combined[time_data_combined['algorithm'] == algorithm].copy()
        pivot_data = df_alg.pivot_table(
            values='TIMESTAMP_median', 
            index='strategy', 
            columns='PHASE', 
            aggfunc='mean'
        )
        heatmap_data = pivot_data.reset_index().melt(
            id_vars=['strategy'], 
            var_name='PHASE', 
            value_name='TIMESTAMP_median'
        )
        phase_order = [
            'P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 
            'P3_RAH', 'P4_FINAL_PROD', 'P5_P6_COPY_SAVE', 'P_TOTAL'
        ]
        heatmap_data['PHASE'] = pd.Categorical(heatmap_data['PHASE'], categories=phase_order, ordered=True)
        
        # Escolher cor baseada no algoritmo
        if algorithm == 'STEEP':
            high_color = '#4ECDC4'  # Verde azulado para STEEP
        else:  # SEBAL
            high_color = '#FF6B6B'  # Vermelho para SEBAL
            
        p4_individual = (ggplot(heatmap_data, aes(x='PHASE', y='strategy', fill='TIMESTAMP_median')) +
                        geom_tile() +
                        geom_text(aes(label=heatmap_data['TIMESTAMP_median'].round(1)), 
                                 size=8, color='black') +
                        scale_fill_gradient(low='#F5F5F5', high=high_color, name='Tempo (ms)') +
                        labs(title=f'Heatmap de Tempos de Processamento - {algorithm}',
                             x='Fase', y='Estratégia') +
                        theme(axis_text_x=element_text(rotation=45, hjust=1),
                              figure_size=(12, 6)))
        p4_individual.save(output_dir / f'heatmap_tempos_{algorithm.lower()}.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / f'heatmap_tempos_{algorithm.lower()}.png'}")

def create_time_plots_filtered(time_data, output_dir):
    """Cria plots de tempo de processamento excluindo P0_READ_INPUT e P_TOTAL"""
    print("Gerando plots de tempo de processamento (fases filtradas)...")
    
    # Extrair estratégia e algoritmo
    time_data = extract_strategy_and_algorithm(time_data)
    
    # Aplicar ordem das estratégias
    time_data = set_strategy_order(time_data)
    
    # Combinar P5 e P6 em uma única fase
    time_data_combined = time_data.copy()
    p5_p6_data = time_data_combined[time_data_combined['PHASE'].isin(['P5_COPY_HOST', 'P6_SAVE_PRODS'])]
    if not p5_p6_data.empty:
        # Agrupar por strategy_algorithm e somar os tempos de P5 e P6
        p5_p6_combined = p5_p6_data.groupby(['strategy_algorithm', 'strategy', 'algorithm'])['TIMESTAMP_median'].sum().reset_index()
        p5_p6_combined['PHASE'] = 'P5_P6_COPY_SAVE'
        
        # Remover P5 e P6 originais e adicionar a combinação
        time_data_combined = time_data_combined[~time_data_combined['PHASE'].isin(['P5_COPY_HOST', 'P6_SAVE_PRODS'])]
        time_data_combined = pd.concat([time_data_combined, p5_p6_combined], ignore_index=True)
    
    # Filtrar fases (excluir P0_READ_INPUT e P_TOTAL)
    filtered_phases = ['P1_INITIAL_PROD', 'P2_PIXEL_SEL', 'P3_RAH', 'P4_FINAL_PROD', 'P5_P6_COPY_SAVE']
    time_data_filtered = time_data_combined[time_data_combined['PHASE'].isin(filtered_phases)].copy()
    
    # Plot 1: Comparação de tempos por fase e estratégia (STEEP) - filtrado
    df_steep = time_data_filtered[time_data_filtered['algorithm'] == 'STEEP'].copy()
    
    p1 = (ggplot(df_steep, aes(x='PHASE', y='TIMESTAMP_median', fill='strategy')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Tempo de Processamento por Fase - Algoritmo STEEP (Fases Principais)',
               x='Fase', y='Tempo (milissegundos)', fill='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(12, 8),
                legend_position='none'))
    
    p1.save(output_dir / 'tempo_por_fase_steep_filtrado.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'tempo_por_fase_steep_filtrado.png'}")
    
    # Plot 2: Comparação de tempos por fase e estratégia (SEBAL) - filtrado
    df_sebal = time_data_filtered[time_data_filtered['algorithm'] == 'SEBAL'].copy()
    
    p2 = (ggplot(df_sebal, aes(x='PHASE', y='TIMESTAMP_median', fill='strategy')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Tempo de Processamento por Fase - Algoritmo SEBAL (Fases Principais)',
               x='Fase', y='Tempo (milissegundos)', fill='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(12, 8),
                legend_position='none'))
    
    p2.save(output_dir / 'tempo_por_fase_sebal_filtrado.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'tempo_por_fase_sebal_filtrado.png'}")
    
    # Plot 3: Comparação entre algoritmos para cada estratégia - filtrado
    p3 = (ggplot(time_data_filtered, aes(x='PHASE', y='TIMESTAMP_median', fill='algorithm')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Comparação de Tempos entre Algoritmos por Estratégia (Fases Principais)',
               x='Fase', y='Tempo (milissegundos)', fill='Algoritmo') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(14, 10)))
    
    p3.save(output_dir / 'comparacao_algoritmos_filtrado.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'comparacao_algoritmos_filtrado.png'}")
    
    # Plot 4: Heatmap de tempos combinado (STEEP em cima, SEBAL embaixo) - filtrado
    heatmap_data_combined = []
    for algorithm in ['STEEP', 'SEBAL']:
        df_alg = time_data_filtered[time_data_filtered['algorithm'] == algorithm].copy()
        pivot_data = df_alg.pivot_table(
            values='TIMESTAMP_median', 
            index='strategy', 
            columns='PHASE', 
            aggfunc='mean'
        )
        heatmap_data = pivot_data.reset_index().melt(
            id_vars=['strategy'], 
            var_name='PHASE', 
            value_name='TIMESTAMP_median'
        )
        heatmap_data['algorithm_name'] = algorithm
        heatmap_data_combined.append(heatmap_data)
    combined_heatmap_data = pd.concat(heatmap_data_combined, ignore_index=True)
    
    # Definir ordem das fases filtradas
    phase_order = filtered_phases
    combined_heatmap_data['PHASE'] = pd.Categorical(combined_heatmap_data['PHASE'], categories=phase_order, ordered=True)
    
    p4 = (ggplot(combined_heatmap_data, aes(x='PHASE', y='strategy', fill='TIMESTAMP_median')) +
          geom_tile() +
          geom_text(aes(label=combined_heatmap_data['TIMESTAMP_median'].round(1)), 
                   size=8, color='black') +
          scale_fill_gradient(low='#F5F5F5', high='#CCCCCC', name='Tempo (ms)') +
          facet_wrap('~algorithm_name', ncol=1, scales='free_y') +
          labs(title='Heatmap de Tempos de Processamento por Algoritmo (Fases Principais)',
               x='Fase', y='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(14, 10)))
    p4.save(output_dir / 'heatmap_tempos_combinado_filtrado.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'heatmap_tempos_combinado_filtrado.png'}")
    
    # Plot 5: Heatmaps individuais por algoritmo - filtrado
    for algorithm in ['STEEP', 'SEBAL']:
        df_alg = time_data_filtered[time_data_filtered['algorithm'] == algorithm].copy()
        pivot_data = df_alg.pivot_table(
            values='TIMESTAMP_median', 
            index='strategy', 
            columns='PHASE', 
            aggfunc='mean'
        )
        heatmap_data = pivot_data.reset_index().melt(
            id_vars=['strategy'], 
            var_name='PHASE', 
            value_name='TIMESTAMP_median'
        )
        heatmap_data['PHASE'] = pd.Categorical(heatmap_data['PHASE'], categories=phase_order, ordered=True)
        
        # Escolher cor baseada no algoritmo
        if algorithm == 'STEEP':
            high_color = '#4ECDC4'  # Verde azulado para STEEP
        else:  # SEBAL
            high_color = '#FF6B6B'  # Vermelho para SEBAL
            
        p4_individual = (ggplot(heatmap_data, aes(x='PHASE', y='strategy', fill='TIMESTAMP_median')) +
                        geom_tile() +
                        geom_text(aes(label=heatmap_data['TIMESTAMP_median'].round(1)), 
                                 size=8, color='black') +
                        scale_fill_gradient(low='#F5F5F5', high=high_color, name='Tempo (ms)') +
                        labs(title=f'Heatmap de Tempos de Processamento - {algorithm} (Fases Principais)',
                             x='Fase', y='Estratégia') +
                        theme(axis_text_x=element_text(rotation=45, hjust=1),
                              figure_size=(12, 6)))
        p4_individual.save(output_dir / f'heatmap_tempos_{algorithm.lower()}_filtrado.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / f'heatmap_tempos_{algorithm.lower()}_filtrado.png'}")

def create_time_plots_filtered_with_read(time_data, output_dir):
    """Cria plots de tempo de processamento incluindo P0_READ_INPUT mas excluindo P_TOTAL"""
    print("Gerando plots de tempo de processamento (incluindo leitura, excluindo total)...")
    
    # Extrair estratégia e algoritmo
    time_data = extract_strategy_and_algorithm(time_data)
    
    # Aplicar ordem das estratégias
    time_data = set_strategy_order(time_data)
    
    # Combinar P5 e P6 em uma única fase
    time_data_combined = time_data.copy()
    p5_p6_data = time_data_combined[time_data_combined['PHASE'].isin(['P5_COPY_HOST', 'P6_SAVE_PRODS'])]
    if not p5_p6_data.empty:
        # Agrupar por strategy_algorithm e somar os tempos de P5 e P6
        p5_p6_combined = p5_p6_data.groupby(['strategy_algorithm', 'strategy', 'algorithm'])['TIMESTAMP_median'].sum().reset_index()
        p5_p6_combined['PHASE'] = 'P5_P6_COPY_SAVE'
        
        # Remover P5 e P6 originais e adicionar a combinação
        time_data_combined = time_data_combined[~time_data_combined['PHASE'].isin(['P5_COPY_HOST', 'P6_SAVE_PRODS'])]
        time_data_combined = pd.concat([time_data_combined, p5_p6_combined], ignore_index=True)
    
    # Filtrar fases (incluir P0_READ_INPUT, excluir P_TOTAL)
    filtered_phases = ['P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 'P3_RAH', 'P4_FINAL_PROD', 'P5_P6_COPY_SAVE']
    time_data_filtered = time_data_combined[time_data_combined['PHASE'].isin(filtered_phases)].copy()
    
    # Plot 1: Comparação de tempos por fase e estratégia (STEEP) - filtrado com leitura
    df_steep = time_data_filtered[time_data_filtered['algorithm'] == 'STEEP'].copy()
    
    p1 = (ggplot(df_steep, aes(x='PHASE', y='TIMESTAMP_median', fill='strategy')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Tempo de Processamento por Fase - Algoritmo STEEP (Incluindo Leitura)',
               x='Fase', y='Tempo (milissegundos)', fill='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(12, 8),
                legend_position='none'))
    
    p1.save(output_dir / 'tempo_por_fase_steep_com_leitura.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'tempo_por_fase_steep_com_leitura.png'}")
    
    # Plot 2: Comparação de tempos por fase e estratégia (SEBAL) - filtrado com leitura
    df_sebal = time_data_filtered[time_data_filtered['algorithm'] == 'SEBAL'].copy()
    
    p2 = (ggplot(df_sebal, aes(x='PHASE', y='TIMESTAMP_median', fill='strategy')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Tempo de Processamento por Fase - Algoritmo SEBAL (Incluindo Leitura)',
               x='Fase', y='Tempo (milissegundos)', fill='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(12, 8),
                legend_position='none'))
    
    p2.save(output_dir / 'tempo_por_fase_sebal_com_leitura.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'tempo_por_fase_sebal_com_leitura.png'}")
    
    # Plot 3: Comparação entre algoritmos para cada estratégia - filtrado com leitura
    p3 = (ggplot(time_data_filtered, aes(x='PHASE', y='TIMESTAMP_median', fill='algorithm')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Comparação de Tempos entre Algoritmos por Estratégia (Incluindo Leitura)',
               x='Fase', y='Tempo (milissegundos)', fill='Algoritmo') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(14, 10)))
    
    p3.save(output_dir / 'comparacao_algoritmos_com_leitura.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'comparacao_algoritmos_com_leitura.png'}")
    
    # Plot 4: Heatmap de tempos combinado (STEEP em cima, SEBAL embaixo) - filtrado com leitura
    heatmap_data_combined = []
    for algorithm in ['STEEP', 'SEBAL']:
        df_alg = time_data_filtered[time_data_filtered['algorithm'] == algorithm].copy()
        pivot_data = df_alg.pivot_table(
            values='TIMESTAMP_median', 
            index='strategy', 
            columns='PHASE', 
            aggfunc='mean'
        )
        heatmap_data = pivot_data.reset_index().melt(
            id_vars=['strategy'], 
            var_name='PHASE', 
            value_name='TIMESTAMP_median'
        )
        heatmap_data['algorithm_name'] = algorithm
        heatmap_data_combined.append(heatmap_data)
    combined_heatmap_data = pd.concat(heatmap_data_combined, ignore_index=True)
    
    # Definir ordem das fases filtradas
    phase_order = filtered_phases
    combined_heatmap_data['PHASE'] = pd.Categorical(combined_heatmap_data['PHASE'], categories=phase_order, ordered=True)
    
    p4 = (ggplot(combined_heatmap_data, aes(x='PHASE', y='strategy', fill='TIMESTAMP_median')) +
          geom_tile() +
          geom_text(aes(label=combined_heatmap_data['TIMESTAMP_median'].round(1)), 
                   size=8, color='black') +
          scale_fill_gradient(low='#F5F5F5', high='#CCCCCC', name='Tempo (ms)') +
          facet_wrap('~algorithm_name', ncol=1, scales='free_y') +
          labs(title='Heatmap de Tempos de Processamento por Algoritmo (Incluindo Leitura)',
               x='Fase', y='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(14, 10)))
    p4.save(output_dir / 'heatmap_tempos_combinado_com_leitura.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'heatmap_tempos_combinado_com_leitura.png'}")
    
    # Plot 5: Heatmaps individuais por algoritmo - filtrado com leitura
    for algorithm in ['STEEP', 'SEBAL']:
        df_alg = time_data_filtered[time_data_filtered['algorithm'] == algorithm].copy()
        pivot_data = df_alg.pivot_table(
            values='TIMESTAMP_median', 
            index='strategy', 
            columns='PHASE', 
            aggfunc='mean'
        )
        heatmap_data = pivot_data.reset_index().melt(
            id_vars=['strategy'], 
            var_name='PHASE', 
            value_name='TIMESTAMP_median'
        )
        heatmap_data['PHASE'] = pd.Categorical(heatmap_data['PHASE'], categories=phase_order, ordered=True)
        
        # Escolher cor baseada no algoritmo
        if algorithm == 'STEEP':
            high_color = '#4ECDC4'  # Verde azulado para STEEP
        else:  # SEBAL
            high_color = '#FF6B6B'  # Vermelho para SEBAL
            
        p4_individual = (ggplot(heatmap_data, aes(x='PHASE', y='strategy', fill='TIMESTAMP_median')) +
                        geom_tile() +
                        geom_text(aes(label=heatmap_data['TIMESTAMP_median'].round(1)), 
                                 size=8, color='black') +
                        scale_fill_gradient(low='#F5F5F5', high=high_color, name='Tempo (ms)') +
                        labs(title=f'Heatmap de Tempos de Processamento - {algorithm} (Incluindo Leitura)',
                             x='Fase', y='Estratégia') +
                        theme(axis_text_x=element_text(rotation=45, hjust=1),
                              figure_size=(12, 6)))
        p4_individual.save(output_dir / f'heatmap_tempos_{algorithm.lower()}_com_leitura.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / f'heatmap_tempos_{algorithm.lower()}_com_leitura.png'}")

def create_time_heatmaps_seconds(time_data, output_dir):
    """Cria heatmaps de tempo de processamento com valores em segundos"""
    print("Gerando heatmaps de tempo de processamento (em segundos)...")
    
    # Extrair estratégia e algoritmo
    time_data = extract_strategy_and_algorithm(time_data)
    
    # Aplicar ordem das estratégias
    time_data = set_strategy_order(time_data)
    
    # Combinar P5 e P6 em uma única fase
    time_data_combined = time_data.copy()
    p5_p6_data = time_data_combined[time_data_combined['PHASE'].isin(['P5_COPY_HOST', 'P6_SAVE_PRODS'])]
    if not p5_p6_data.empty:
        # Agrupar por strategy_algorithm e somar os tempos de P5 e P6
        p5_p6_combined = p5_p6_data.groupby(['strategy_algorithm', 'strategy', 'algorithm'])['TIMESTAMP_median'].sum().reset_index()
        p5_p6_combined['PHASE'] = 'P5_P6_COPY_SAVE'
        
        # Remover P5 e P6 originais e adicionar a combinação
        time_data_combined = time_data_combined[~time_data_combined['PHASE'].isin(['P5_COPY_HOST', 'P6_SAVE_PRODS'])]
        time_data_combined = pd.concat([time_data_combined, p5_p6_combined], ignore_index=True)
    
    # Converter de milissegundos para segundos
    time_data_combined['TIMESTAMP_median'] = time_data_combined['TIMESTAMP_median'] / 1000.0
    
    # Plot 1: Heatmap de tempos combinado (STEEP em cima, SEBAL embaixo) - em segundos
    heatmap_data_combined = []
    for algorithm in ['STEEP', 'SEBAL']:
        df_alg = time_data_combined[time_data_combined['algorithm'] == algorithm].copy()
        pivot_data = df_alg.pivot_table(
            values='TIMESTAMP_median', 
            index='strategy', 
            columns='PHASE', 
            aggfunc='mean'
        )
        heatmap_data = pivot_data.reset_index().melt(
            id_vars=['strategy'], 
            var_name='PHASE', 
            value_name='TIMESTAMP_median'
        )
        heatmap_data['algorithm_name'] = algorithm
        heatmap_data_combined.append(heatmap_data)
    combined_heatmap_data = pd.concat(heatmap_data_combined, ignore_index=True)
    
    # Definir ordem das fases
    phase_order = [
        'P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 
        'P3_RAH', 'P4_FINAL_PROD', 'P5_P6_COPY_SAVE', 'P_TOTAL'
    ]
    combined_heatmap_data['PHASE'] = pd.Categorical(combined_heatmap_data['PHASE'], categories=phase_order, ordered=True)
    
    # Remover valores NaN antes de criar o heatmap
    combined_heatmap_data = combined_heatmap_data.dropna(subset=['TIMESTAMP_median'])
    
    p1 = (ggplot(combined_heatmap_data, aes(x='PHASE', y='strategy', fill='TIMESTAMP_median')) +
          geom_tile() +
          geom_text(aes(label=combined_heatmap_data['TIMESTAMP_median'].round(4)), 
                   size=8, color='black') +
          scale_fill_gradient(low='#F5F5F5', high='#CCCCCC', name='Tempo (s)') +
          facet_wrap('~algorithm_name', ncol=1, scales='free_y') +
          labs(title='Heatmap de Tempos de Processamento por Algoritmo (em Segundos)',
               x='Fase', y='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(14, 10)))
    p1.save(output_dir / 'heatmap_tempos_combinado_segundos.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'heatmap_tempos_combinado_segundos.png'}")
    
    # Plot 2: Heatmaps individuais por algoritmo - em segundos
    for algorithm in ['STEEP', 'SEBAL']:
        df_alg = time_data_combined[time_data_combined['algorithm'] == algorithm].copy()
        pivot_data = df_alg.pivot_table(
            values='TIMESTAMP_median', 
            index='strategy', 
            columns='PHASE', 
            aggfunc='mean'
        )
        heatmap_data = pivot_data.reset_index().melt(
            id_vars=['strategy'], 
            var_name='PHASE', 
            value_name='TIMESTAMP_median'
        )
        heatmap_data['PHASE'] = pd.Categorical(heatmap_data['PHASE'], categories=phase_order, ordered=True)
        
        # Escolher cor baseada no algoritmo
        if algorithm == 'STEEP':
            high_color = '#4ECDC4'  # Verde azulado para STEEP
        else:  # SEBAL
            high_color = '#FF6B6B'  # Vermelho para SEBAL
            
        p2_individual = (ggplot(heatmap_data, aes(x='PHASE', y='strategy', fill='TIMESTAMP_median')) +
                        geom_tile() +
                        geom_text(aes(label=heatmap_data['TIMESTAMP_median'].round(4)), 
                                 size=8, color='black') +
                        scale_fill_gradient(low='#F5F5F5', high=high_color, name='Tempo (s)') +
                        labs(title=f'Heatmap de Tempos de Processamento - {algorithm} (em Segundos)',
                             x='Fase', y='Estratégia') +
                        theme(axis_text_x=element_text(rotation=45, hjust=1),
                              figure_size=(12, 6)))
        p2_individual.save(output_dir / f'heatmap_tempos_{algorithm.lower()}_segundos.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / f'heatmap_tempos_{algorithm.lower()}_segundos.png'}")
    
    # Plot 3: Heatmap com nomes das fases em português
    # Mapear nomes das fases para português
    phase_mapping = {
        'P0_READ_INPUT': 'Leitura dos dados de entrada',
        'P1_INITIAL_PROD': 'Produtos Iniciais',
        'P2_PIXEL_SEL': 'Seleção de Pixels',
        'P3_RAH': 'Produtos Intermediários',
        'P4_FINAL_PROD': 'Produtos Finais',
        'P5_P6_COPY_SAVE': 'Escrita dos dados de saída',
        'P_TOTAL': 'Tempo Total'
    }
    
    # Aplicar mapeamento aos dados
    combined_heatmap_data_pt = combined_heatmap_data.copy()
    combined_heatmap_data_pt['PHASE_PT'] = combined_heatmap_data_pt['PHASE'].map(phase_mapping)
    
    # Definir ordem das fases em português
    phase_order_pt = [
        'Leitura dos dados de entrada', 'Produtos Iniciais', 'Seleção de Pixels',
        'Produtos Intermediários', 'Produtos Finais', 'Escrita dos dados de saída', 'Tempo Total'
    ]
    combined_heatmap_data_pt['PHASE_PT'] = pd.Categorical(combined_heatmap_data_pt['PHASE_PT'], categories=phase_order_pt, ordered=True)
    
    p3_pt = (ggplot(combined_heatmap_data_pt, aes(x='PHASE_PT', y='strategy', fill='TIMESTAMP_median')) +
             geom_tile() +
             geom_text(aes(label=combined_heatmap_data_pt['TIMESTAMP_median'].round(4)), 
                      size=8, color='black') +
             scale_fill_gradient(low='#F5F5F5', high='#CCCCCC', name='Tempo (s)') +
             facet_wrap('~algorithm_name', ncol=1, scales='free_y') +
             labs(title='Heatmap de Tempos de Processamento por Algoritmo (em Segundos)',
                  x='Fase', y='Estratégia') +
             theme(axis_text_x=element_text(rotation=45, hjust=1),
                   figure_size=(14, 10)))
    p3_pt.save(output_dir / 'heatmap_tempos_combinado_segundos_pt.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'heatmap_tempos_combinado_segundos_pt.png'}")
    
    # Plot 4: Heatmaps individuais por algoritmo com nomes em português
    for algorithm in ['STEEP', 'SEBAL']:
        df_alg = time_data_combined[time_data_combined['algorithm'] == algorithm].copy()
        pivot_data = df_alg.pivot_table(
            values='TIMESTAMP_median', 
            index='strategy', 
            columns='PHASE', 
            aggfunc='mean'
        )
        heatmap_data = pivot_data.reset_index().melt(
            id_vars=['strategy'], 
            var_name='PHASE', 
            value_name='TIMESTAMP_median'
        )
        
        # Aplicar mapeamento para português
        heatmap_data['PHASE_PT'] = heatmap_data['PHASE'].map(phase_mapping)
        heatmap_data['PHASE_PT'] = pd.Categorical(heatmap_data['PHASE_PT'], categories=phase_order_pt, ordered=True)
        
        # Escolher cor baseada no algoritmo
        if algorithm == 'STEEP':
            high_color = '#4ECDC4'  # Verde azulado para STEEP
        else:  # SEBAL
            high_color = '#FF6B6B'  # Vermelho para SEBAL
            
        p4_individual_pt = (ggplot(heatmap_data, aes(x='PHASE_PT', y='strategy', fill='TIMESTAMP_median')) +
                           geom_tile() +
                           geom_text(aes(label=heatmap_data['TIMESTAMP_median'].round(4)), 
                                    size=8, color='black') +
                           scale_fill_gradient(low='#F5F5F5', high=high_color, name='Tempo (s)') +
                           labs(title=f'Heatmap de Tempos de Processamento - {algorithm} (em Segundos)',
                                x='Fase', y='Estratégia') +
                           theme(axis_text_x=element_text(rotation=45, hjust=1),
                                 figure_size=(12, 6)))
        p4_individual_pt.save(output_dir / f'heatmap_tempos_{algorithm.lower()}_segundos_pt.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / f'heatmap_tempos_{algorithm.lower()}_segundos_pt.png'}")

def remove_outliers_iqr(df, value_col, group_cols=None):
    """Remove outliers usando o método do IQR para cada grupo."""
    if group_cols is None:
        group_cols = []
    def iqr_filter(group):
        q1 = group[value_col].quantile(0.25)
        q3 = group[value_col].quantile(0.70)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return group[(group[value_col] >= lower) & (group[value_col] <= upper)]
    if group_cols:
        return df.groupby(group_cols, group_keys=False).apply(iqr_filter).reset_index(drop=True)
    else:
        return iqr_filter(df)

def create_resource_plots(cpu_data, gpu_data, output_dir):
    """Cria plots de utilização de recursos"""
    print("Gerando plots de utilização de recursos...")
    
    if cpu_data is not None:
        # Extrair estratégia e algoritmo
        cpu_data = extract_strategy_and_algorithm(cpu_data)
        cpu_data = set_strategy_order(cpu_data)
        # Remover outliers CPU
        cpu_data_cpu = remove_outliers_iqr(cpu_data, 'CPU_USAGE_PERCENTAGE_max', ['strategy', 'algorithm'])
        cpu_data_mem = remove_outliers_iqr(cpu_data, 'MEM_USAGE_MB_max', ['strategy', 'algorithm'])
        # Normalizar CPU_USAGE_PERCENTAGE_max para porcentagem
        cpu_data_cpu = cpu_data_cpu.copy()
        cpu_data_cpu['CPU_USAGE_PERCENTAGE_max'] = (cpu_data_cpu['CPU_USAGE_PERCENTAGE_max'] / 1200) * 100
        
        # Plot CPU para STEEP
        cpu_data_steep = cpu_data_cpu[cpu_data_cpu['algorithm'] == 'STEEP']
        if not cpu_data_steep.empty:
            p5_steep = (ggplot(cpu_data_steep, aes(x='strategy', y='CPU_USAGE_PERCENTAGE_max', fill='strategy')) +
                       geom_boxplot(outlier_shape=None, outlier_size=2, width=0.7, alpha=0.7) +
                       labs(title='Distribuição da Utilização de CPU por Estratégia - Algoritmo STEEP (Normalizado 0-100%)',
                            x='Estratégia', y='CPU (%)', fill='Estratégia') +
                       theme(axis_text_x=element_text(rotation=45, hjust=1),
                             figure_size=(10, 6),
                             legend_position='none') +
                       coord_cartesian(ylim=(0, 100)))
            p5_steep.save(output_dir / 'utilizacao_cpu_steep.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / 'utilizacao_cpu_steep.png'}")
        
        # Plot CPU para SEBAL
        cpu_data_sebal = cpu_data_cpu[cpu_data_cpu['algorithm'] == 'SEBAL']
        if not cpu_data_sebal.empty:
            p5_sebal = (ggplot(cpu_data_sebal, aes(x='strategy', y='CPU_USAGE_PERCENTAGE_max', fill='strategy')) +
                       geom_boxplot(outlier_shape=None, outlier_size=2, width=0.7, alpha=0.7) +
                       labs(title='Distribuição da Utilização de CPU por Estratégia - Algoritmo SEBAL (Normalizado 0-100%)',
                            x='Estratégia', y='CPU (%)', fill='Estratégia') +
                       theme(axis_text_x=element_text(rotation=45, hjust=1),
                             figure_size=(10, 6),
                             legend_position='none') +
                       coord_cartesian(ylim=(0, 100)))
            p5_sebal.save(output_dir / 'utilizacao_cpu_sebal.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / 'utilizacao_cpu_sebal.png'}")
        
        # Boxplot Memória RAM
        p6 = (ggplot(cpu_data_mem, aes(x='strategy', y='MEM_USAGE_MB_max', fill='algorithm')) +
              geom_boxplot(outlier_shape=None, outlier_size=2, width=0.7, alpha=0.7) +
              labs(title='Distribuição da Utilização de Memória RAM por Estratégia e Algoritmo',
                   x='Estratégia', y='Memória RAM (MB)', fill='Algoritmo') +
              theme(axis_text_x=element_text(rotation=45, hjust=1),
                    figure_size=(10, 6)))
        p6.save(output_dir / 'utilizacao_memoria.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'utilizacao_memoria.png'}")

    if gpu_data is not None:
        # Extrair estratégia e algoritmo
        gpu_data = extract_strategy_and_algorithm(gpu_data)
        gpu_data = set_strategy_order(gpu_data)
        # Filtrar apenas estratégias que realmente usam GPU (utilização > 1%)
        gpu_data_filtered = gpu_data[gpu_data['GPU_USAGE_PERCENTAGE_max'] > 1.0].copy()
        # Remover outliers GPU
        gpu_data_gpu = remove_outliers_iqr(gpu_data_filtered, 'GPU_USAGE_PERCENTAGE_max', ['strategy', 'algorithm'])
        # Normalizar GPU_USAGE_PERCENTAGE_max para porcentagem
        gpu_data_gpu = gpu_data_gpu.copy()
        gpu_data_gpu['GPU_USAGE_PERCENTAGE_max'] = gpu_data_gpu['GPU_USAGE_PERCENTAGE_max']
        gpu_data_mem = remove_outliers_iqr(gpu_data_filtered, 'MEM_USAGE_MB_max', ['strategy', 'algorithm'])
        gpu_data_power = remove_outliers_iqr(gpu_data_filtered, 'POWER_W_max', ['strategy', 'algorithm'])
        gpu_data_temp = remove_outliers_iqr(gpu_data_filtered, 'TEMP_C_max', ['strategy', 'algorithm'])
        if not gpu_data_filtered.empty:
            # Plot GPU para STEEP
            gpu_data_steep = gpu_data_gpu[gpu_data_gpu['algorithm'] == 'STEEP']
            if not gpu_data_steep.empty:
                p7_steep = (ggplot(gpu_data_steep, aes(x='strategy', y='GPU_USAGE_PERCENTAGE_max', fill='strategy')) +
                           geom_boxplot(outlier_shape=None, outlier_size=2, width=0.7, alpha=0.7) +
                           labs(title='Distribuição da Utilização de GPU por Estratégia - Algoritmo STEEP',
                                x='Estratégia', y='GPU (%)', fill='Estratégia') +
                           theme(axis_text_x=element_text(rotation=45, hjust=1),
                                 figure_size=(10, 6),
                                 legend_position='none') +
                           coord_cartesian(ylim=(0, 100)))
                p7_steep.save(output_dir / 'utilizacao_gpu_steep.png', dpi=300, bbox_inches='tight')
                print(f"  Salvo: {output_dir / 'utilizacao_gpu_steep.png'}")
            
            # Plot GPU para SEBAL
            gpu_data_sebal = gpu_data_gpu[gpu_data_gpu['algorithm'] == 'SEBAL']
            if not gpu_data_sebal.empty:
                p7_sebal = (ggplot(gpu_data_sebal, aes(x='strategy', y='GPU_USAGE_PERCENTAGE_max', fill='strategy')) +
                           geom_boxplot(outlier_shape=None, outlier_size=2, width=0.7, alpha=0.7) +
                           labs(title='Distribuição da Utilização de GPU por Estratégia - Algoritmo SEBAL',
                                x='Estratégia', y='GPU (%)', fill='Estratégia') +
                           theme(axis_text_x=element_text(rotation=45, hjust=1),
                                 figure_size=(10, 6),
                                 legend_position='none') +
                           coord_cartesian(ylim=(0, 100)))
                p7_sebal.save(output_dir / 'utilizacao_gpu_sebal.png', dpi=300, bbox_inches='tight')
                print(f"  Salvo: {output_dir / 'utilizacao_gpu_sebal.png'}")
            
            # Boxplot Memória GPU
            p8 = (ggplot(gpu_data_mem, aes(x='strategy', y='MEM_USAGE_MB_max', fill='algorithm')) +
                  geom_boxplot(outlier_shape=None, outlier_size=2, width=0.7, alpha=0.7) +
                  labs(title='Distribuição da Utilização de Memória GPU por Estratégia e Algoritmo',
                       x='Estratégia', y='Memória GPU (MB)', fill='Algoritmo') +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(10, 6)))
            p8.save(output_dir / 'utilizacao_memoria_gpu.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / 'utilizacao_memoria_gpu.png'}")
            # Boxplot GPU Power
            p9 = (ggplot(gpu_data_power, aes(x='strategy', y='POWER_W_max', fill='algorithm')) +
                  geom_boxplot(outlier_shape=None, outlier_size=2, width=0.7, alpha=0.7) +
                  labs(title='Distribuição do Consumo de Energia GPU por Estratégia e Algoritmo',
                       x='Estratégia', y='Potência (W)', fill='Algoritmo') +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(10, 6)))
            p9.save(output_dir / 'energia_gpu.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / 'energia_gpu.png'}")
            # Boxplot GPU Temperature
            p10 = (ggplot(gpu_data_temp, aes(x='strategy', y='TEMP_C_max', fill='algorithm')) +
                   geom_boxplot(outlier_shape=None, outlier_size=2, width=0.7, alpha=0.7) +
                   labs(title='Distribuição da Temperatura GPU por Estratégia e Algoritmo',
                        x='Estratégia', y='Temperatura (°C)', fill='Algoritmo') +
                   theme(axis_text_x=element_text(rotation=45, hjust=1),
                         figure_size=(10, 6)))
            p10.save(output_dir / 'temperatura_gpu.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / 'temperatura_gpu.png'}")
        else:
            print("  Aviso: Nenhum dado GPU com utilização > 1% encontrado")

def create_resource_by_phase_plots(cpu_by_phase_data, gpu_by_phase_data, time_data, output_dir):
    """Cria plots de utilização de recursos por fase"""
    print("Gerando plots de utilização de recursos por fase...")
    
    if gpu_by_phase_data is not None:
        # Extrair estratégia e algoritmo
        gpu_by_phase_data = extract_strategy_and_algorithm(gpu_by_phase_data)
        gpu_by_phase_data = set_strategy_order(gpu_by_phase_data)
        
        # Filtrar apenas estratégias que realmente usam GPU (utilização > 1%)
        gpu_by_phase_data_filtered = gpu_by_phase_data[gpu_by_phase_data['GPU_USAGE_PERCENTAGE_max_median'] > 1.0].copy()
        
        if not gpu_by_phase_data_filtered.empty:
            # (Removido) Plot GPU por fase - Mediana dos máximos
            pass
        else:
            print("  Aviso: Nenhum dado GPU por fase com utilização > 1% encontrado")

def create_total_time_stacked_plot(time_data, output_dir):
    """Cria gráfico de barras empilhadas do tempo total por estratégia, empilhando por fase"""
    print("Gerando gráfico de tempo total empilhado por fase...")

    # Extrair strategy e algorithm antes de qualquer filtro
    time_data = extract_strategy_and_algorithm(time_data)
    time_data = set_strategy_order(time_data)

    # Filtrar apenas as fases principais (P*)
    main_phases = ['P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 'P3_RAH', 'P4_FINAL_PROD', 'P6_SAVE_PRODS']
    time_data = time_data[time_data['PHASE'].isin(main_phases)]

    # Agrupar por estratégia, algoritmo e fase, somando o tempo
    stacked = time_data.groupby(['strategy', 'algorithm', 'PHASE'])['TIMESTAMP_median'].sum().reset_index()
    
    # Converter de milissegundos para segundos
    stacked['TIMESTAMP_median'] = stacked['TIMESTAMP_median'] / 1000.0

    # Garantir ordem das fases
    phase_order = main_phases
    stacked['PHASE'] = pd.Categorical(stacked['PHASE'], categories=phase_order, ordered=True)

    p = (ggplot(stacked, aes(x='strategy', y='TIMESTAMP_median', fill='PHASE')) +
         geom_bar(stat='identity', position='stack', width=0.7) +
         facet_wrap('~algorithm', ncol=1) +
         labs(title='Tempo Total de Execução por Estratégia (barras empilhadas por fase)',
              x='Estratégia', y='Tempo (segundos)', fill='Fase') +
         theme(axis_text_x=element_text(rotation=45, hjust=1),
               figure_size=(12, 8)))
    p.save(output_dir / 'tempo_total_empilhado.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'tempo_total_empilhado.png'}")

def create_total_time_stacked_plot_gpu(time_data, output_dir):
    """Cria gráfico de barras empilhadas do tempo total por estratégia, empilhando por fase, apenas para abordagens GPU."""
    print("Gerando gráfico de tempo total empilhado por fase (GPU)...")

    # Extrair strategy e algorithm antes de qualquer filtro
    time_data = extract_strategy_and_algorithm(time_data)

    # Filtrar apenas as fases principais (P*) (sem P_TOTAL)
    main_phases = ['P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 'P3_RAH', 'P4_FINAL_PROD', 'P5_COPY_HOST', 'P6_SAVE_PRODS']
    time_data = time_data[time_data['PHASE'].isin(main_phases)]

    # Filtrar apenas estratégias que usam GPU (GPU CUDA Double, GPU CUDA Float, GPU CUDA Stream Float)
    gpu_strategies = ['GPU CUDA Double', 'GPU CUDA Float', 'GPU CUDA Stream Float']
    time_data = time_data[time_data['strategy'].isin(gpu_strategies)]
    
    # Aplicar ordem apenas para as estratégias GPU
    time_data['strategy'] = pd.Categorical(time_data['strategy'], categories=gpu_strategies, ordered=True)
    
    # Garantir que só existam essas estratégias no eixo X
    stacked = time_data.groupby(['strategy', 'algorithm', 'PHASE'])['TIMESTAMP_median'].sum().reset_index()
    stacked = stacked[stacked['strategy'].isin(gpu_strategies)]

    # Garantir ordem das fases
    phase_order = main_phases
    stacked['PHASE'] = pd.Categorical(stacked['PHASE'], categories=phase_order, ordered=True)

    p = (ggplot(stacked, aes(x='strategy', y='TIMESTAMP_median', fill='PHASE')) +
         geom_bar(stat='identity', position='stack', width=0.7) +
         facet_wrap('~algorithm', ncol=1) +
         labs(title='Tempo Total de Execução por Estratégia GPU (barras empilhadas por fase)',
              x='Estratégia', y='Tempo (segundos)', fill='Fase') +
         theme(axis_text_x=element_text(rotation=45, hjust=1),
               figure_size=(12, 8)))
    p.save(output_dir / 'tempo_total_empilhado_gpu.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'tempo_total_empilhado_gpu.png'}")

def create_total_time_normal_plot(time_data, output_dir):
    """Cria gráfico de barras normais do tempo total por estratégia e algoritmo"""
    print("Gerando gráfico de tempo total com barras normais...")

    # Extrair strategy e algorithm antes de qualquer filtro
    time_data = extract_strategy_and_algorithm(time_data)
    time_data = set_strategy_order(time_data)

    # Filtrar apenas a fase P_TOTAL (tempo total)
    time_data = time_data[time_data['PHASE'] == 'P_TOTAL']

    # Calcular estatísticas com intervalo de confiança
    def calculate_stats(group):
        mean_val = group['TIMESTAMP_median'].mean() / 1000.0  # Converter para segundos
        std_val = group['TIMESTAMP_median'].std() / 1000.0    # Converter para segundos
        n = len(group)
        # Erro padrão da média
        se = std_val / np.sqrt(n) if n > 1 else 0
        # Intervalo de confiança 95% (1.96 * erro padrão)
        ci_95 = 1.96 * se
        return pd.Series({
            'mean': mean_val,
            'std': std_val,
            'se': se,
            'ci_95': ci_95,
            'n': n
        })

    # Agrupar por estratégia e algoritmo, calculando estatísticas
    total_time_data = time_data.groupby(['strategy', 'algorithm']).apply(calculate_stats).reset_index()

    # Plot para STEEP
    steep_data = total_time_data[total_time_data['algorithm'] == 'STEEP']
    if not steep_data.empty:
        p_steep = (ggplot(steep_data, aes(x='strategy', y='mean')) +
                  geom_bar(stat='identity', fill='#FF6B6B', width=0.7) +
                  geom_errorbar(aes(ymin='mean-ci_95', ymax='mean+ci_95'), 
                               width=0.2, color='black', size=1) +
                  labs(title='Tempo Total de Execução por Estratégia - Algoritmo STEEP (IC 95%)',
                       x='Estratégia', y='Tempo Total (segundos)') +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(12, 8)))
        p_steep.save(output_dir / 'tempo_total_normal_steep.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'tempo_total_normal_steep.png'}")

    # Plot para SEBAL
    sebal_data = total_time_data[total_time_data['algorithm'] == 'SEBAL']
    if not sebal_data.empty:
        p_sebal = (ggplot(sebal_data, aes(x='strategy', y='mean')) +
                  geom_bar(stat='identity', fill='#4ECDC4', width=0.7) +
                  geom_errorbar(aes(ymin='mean-ci_95', ymax='mean+ci_95'), 
                               width=0.2, color='black', size=1) +
                  labs(title='Tempo Total de Execução por Estratégia - Algoritmo SEBAL (IC 95%)',
                       x='Estratégia', y='Tempo Total (segundos)') +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(12, 8)))
        p_sebal.save(output_dir / 'tempo_total_normal_sebal.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'tempo_total_normal_sebal.png'}")

def create_total_time_normal_plot_gpu(time_data, output_dir):
    """Cria gráfico de barras normais do tempo total apenas para estratégias GPU"""
    print("Gerando gráfico de tempo total com barras normais (GPU)...")

    # Extrair strategy e algorithm antes de qualquer filtro
    time_data = extract_strategy_and_algorithm(time_data)

    # Filtrar apenas a fase P_TOTAL (tempo total)
    time_data = time_data[time_data['PHASE'] == 'P_TOTAL']
    
    # Converter de milissegundos para segundos
    time_data = time_data.copy()
    time_data['TIMESTAMP_median'] = time_data['TIMESTAMP_median'] / 1000.0

    # Filtrar apenas estratégias que usam GPU
    gpu_strategies = ['GPU CUDA Double', 'GPU CUDA Float', 'GPU CUDA Stream Float']
    time_data = time_data[time_data['strategy'].isin(gpu_strategies)]
    
    # Aplicar ordem apenas para as estratégias GPU
    time_data['strategy'] = pd.Categorical(time_data['strategy'], categories=gpu_strategies, ordered=True)

    # Calcular estatísticas com intervalo de confiança
    def calculate_stats(group):
        mean_val = group['TIMESTAMP_median'].mean() / 1000.0  # Converter para segundos
        std_val = group['TIMESTAMP_median'].std() / 1000.0    # Converter para segundos
        n = len(group)
        # Erro padrão da média
        se = std_val / np.sqrt(n) if n > 1 else 0
        # Intervalo de confiança 95% (1.96 * erro padrão)
        ci_95 = 1.96 * se
        return pd.Series({
            'mean': mean_val,
            'std': std_val,
            'se': se,
            'ci_95': ci_95,
            'n': n
        })

    # Agrupar por estratégia e algoritmo, calculando estatísticas
    total_time_data = time_data.groupby(['strategy', 'algorithm']).apply(calculate_stats).reset_index()

    # Plot para STEEP (GPU)
    steep_data = total_time_data[total_time_data['algorithm'] == 'STEEP']
    if not steep_data.empty:
        p_steep = (ggplot(steep_data, aes(x='strategy', y='mean')) +
                  geom_bar(stat='identity', fill='#FF6B6B', width=0.7) +
                  geom_errorbar(aes(ymin='mean-ci_95', ymax='mean+ci_95'), 
                               width=0.2, color='black', size=1) +
                  labs(title='Tempo Total de Execução por Estratégia GPU - Algoritmo STEEP (IC 95%)',
                       x='Estratégia', y='Tempo Total (segundos)') +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(10, 6)))
        p_steep.save(output_dir / 'tempo_total_normal_gpu_steep.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'tempo_total_normal_gpu_steep.png'}")

    # Plot para SEBAL (GPU)
    sebal_data = total_time_data[total_time_data['algorithm'] == 'SEBAL']
    if not sebal_data.empty:
        p_sebal = (ggplot(sebal_data, aes(x='strategy', y='mean')) +
                  geom_bar(stat='identity', fill='#4ECDC4', width=0.7) +
                  geom_errorbar(aes(ymin='mean-ci_95', ymax='mean+ci_95'), 
                               width=0.2, color='black', size=1) +
                  labs(title='Tempo Total de Execução por Estratégia GPU - Algoritmo SEBAL (IC 95%)',
                       x='Estratégia', y='Tempo Total (segundos)') +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(10, 6)))
        p_sebal.save(output_dir / 'tempo_total_normal_gpu_sebal.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'tempo_total_normal_gpu_sebal.png'}")

def create_total_time_boxplot(time_data, output_dir):
    """Cria boxplots do tempo total por estratégia e algoritmo"""
    print("Gerando boxplots do tempo total...")

    # Extrair strategy e algorithm antes de qualquer filtro
    time_data = extract_strategy_and_algorithm(time_data)
    time_data = set_strategy_order(time_data)

    # Filtrar apenas a fase P_TOTAL (tempo total)
    time_data = time_data[time_data['PHASE'] == 'P_TOTAL']
    
    # Converter de milissegundos para segundos
    time_data = time_data.copy()
    time_data['TIMESTAMP_median'] = time_data['TIMESTAMP_median'] / 1000.0

    # Plot para STEEP
    steep_data = time_data[time_data['algorithm'] == 'STEEP']
    if not steep_data.empty:
        p_steep = (ggplot(steep_data, aes(x='strategy', y='TIMESTAMP_median', fill='strategy')) +
                  geom_boxplot(outlier_shape=None, outlier_size=2, width=0.7, alpha=0.7) +
                  labs(title='Distribuição do Tempo Total por Estratégia - Algoritmo STEEP',
                       x='Estratégia', y='Tempo Total (segundos)', fill='Estratégia') +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(12, 8),
                        legend_position='none'))
        p_steep.save(output_dir / 'tempo_total_boxplot_steep.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'tempo_total_boxplot_steep.png'}")

    # Plot para SEBAL
    sebal_data = time_data[time_data['algorithm'] == 'SEBAL']
    if not sebal_data.empty:
        p_sebal = (ggplot(sebal_data, aes(x='strategy', y='TIMESTAMP_median', fill='strategy')) +
                  geom_boxplot(outlier_shape=None, outlier_size=2, width=0.7, alpha=0.7) +
                  labs(title='Distribuição do Tempo Total por Estratégia - Algoritmo SEBAL',
                       x='Estratégia', y='Tempo Total (segundos)', fill='Estratégia') +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(12, 8),
                        legend_position='none'))
        p_sebal.save(output_dir / 'tempo_total_boxplot_sebal.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'tempo_total_boxplot_sebal.png'}")

def create_total_time_boxplot_gpu(time_data, output_dir):
    """Cria boxplots do tempo total apenas para estratégias GPU"""
    print("Gerando boxplots do tempo total (GPU)...")

    # Extrair strategy e algorithm antes de qualquer filtro
    time_data = extract_strategy_and_algorithm(time_data)

    # Filtrar apenas a fase P_TOTAL (tempo total)
    time_data = time_data[time_data['PHASE'] == 'P_TOTAL']
    
    # Converter de milissegundos para segundos
    time_data = time_data.copy()
    time_data['TIMESTAMP_median'] = time_data['TIMESTAMP_median'] / 1000.0

    # Filtrar apenas estratégias que usam GPU
    gpu_strategies = ['GPU CUDA Double', 'GPU CUDA Float', 'GPU CUDA Stream Float']
    time_data = time_data[time_data['strategy'].isin(gpu_strategies)]
    
    # Aplicar ordem apenas para as estratégias GPU
    time_data['strategy'] = pd.Categorical(time_data['strategy'], categories=gpu_strategies, ordered=True)

    # Plot para STEEP (GPU)
    steep_data = time_data[time_data['algorithm'] == 'STEEP']
    if not steep_data.empty:
        p_steep = (ggplot(steep_data, aes(x='strategy', y='TIMESTAMP_median', fill='strategy')) +
                  geom_boxplot(outlier_shape=None, outlier_size=2, width=0.7, alpha=0.7) +
                  labs(title='Distribuição do Tempo Total por Estratégia GPU - Algoritmo STEEP',
                       x='Estratégia', y='Tempo Total (segundos)', fill='Estratégia') +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(10, 6),
                        legend_position='none'))
        p_steep.save(output_dir / 'tempo_total_boxplot_gpu_steep.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'tempo_total_boxplot_gpu_steep.png'}")

    # Plot para SEBAL (GPU)
    sebal_data = time_data[time_data['algorithm'] == 'SEBAL']
    if not sebal_data.empty:
        p_sebal = (ggplot(sebal_data, aes(x='strategy', y='TIMESTAMP_median', fill='strategy')) +
                  geom_boxplot(outlier_shape=None, outlier_size=2, width=0.7, alpha=0.7) +
                  labs(title='Distribuição do Tempo Total por Estratégia GPU - Algoritmo SEBAL',
                       x='Estratégia', y='Tempo Total (segundos)', fill='Estratégia') +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(10, 6),
                        legend_position='none'))
        p_sebal.save(output_dir / 'tempo_total_boxplot_gpu_sebal.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'tempo_total_boxplot_gpu_sebal.png'}")

def create_summary_plots(time_data, cpu_data, gpu_data, output_dir):
    """Cria plots de resumo e eficiência"""
    print("Gerando plots de resumo e eficiência...")
    
    if cpu_data is not None and time_data is not None:
        # Extrair estratégia e algoritmo
        cpu_data = extract_strategy_and_algorithm(cpu_data)
        time_data = extract_strategy_and_algorithm(time_data)
        
        # Combinar dados de CPU e tempo total
        total_time = time_data.groupby(['strategy_algorithm', 'strategy', 'algorithm'])['TIMESTAMP_median'].sum().reset_index()
        total_time.columns = ['strategy_algorithm', 'strategy', 'algorithm', 'total_time']
        
        efficiency_data = cpu_data.merge(
            total_time[['strategy_algorithm', 'total_time']], 
            on='strategy_algorithm', 
            how='inner'
        )
        
        if not efficiency_data.empty:
            # Calcular eficiência (tempo / utilização)
            efficiency_data['cpu_efficiency'] = efficiency_data['total_time'] / efficiency_data['CPU_USAGE_PERCENTAGE_max']
            
            p12 = (ggplot(efficiency_data, aes(x='strategy', y='cpu_efficiency', fill='algorithm')) +
                   geom_bar(stat='identity', position='dodge', width=0.7) +
                   labs(title='Eficiência CPU (Tempo/Utilização) por Estratégia e Algoritmo',
                        x='Estratégia', y='Eficiência CPU', fill='Algoritmo') +
                   theme(axis_text_x=element_text(rotation=45, hjust=1),
                         figure_size=(10, 6)))
            
            p12.save(output_dir / 'eficiencia_cpu.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / 'eficiencia_cpu.png'}")

    if gpu_data is not None and time_data is not None:
        # Extrair estratégia e algoritmo
        gpu_data = extract_strategy_and_algorithm(gpu_data)
        time_data = extract_strategy_and_algorithm(time_data)
        
        # Filtrar apenas estratégias que realmente usam GPU (utilização > 1%)
        gpu_data_filtered = gpu_data[gpu_data['GPU_USAGE_PERCENTAGE_max'] > 1.0].copy()
        
        if not gpu_data_filtered.empty:
            # Combinar dados de GPU e tempo total
            total_time = time_data.groupby(['strategy_algorithm', 'strategy', 'algorithm'])['TIMESTAMP_median'].sum().reset_index()
            total_time.columns = ['strategy_algorithm', 'strategy', 'algorithm', 'total_time']
            
            efficiency_data = gpu_data_filtered.merge(
                total_time[['strategy_algorithm', 'total_time']], 
                on='strategy_algorithm', 
                how='inner'
            )
            
            if not efficiency_data.empty:
                # Calcular eficiência (tempo / utilização)
                efficiency_data['gpu_efficiency'] = efficiency_data['total_time'] / efficiency_data['GPU_USAGE_PERCENTAGE_max']
                
                p13 = (ggplot(efficiency_data, aes(x='strategy', y='gpu_efficiency', fill='algorithm')) +
                       geom_bar(stat='identity', position='dodge', width=0.7) +
                       labs(title='Eficiência GPU (Tempo/Utilização) por Estratégia e Algoritmo',
                            x='Estratégia', y='Eficiência GPU', fill='Algoritmo') +
                       theme(axis_text_x=element_text(rotation=45, hjust=1),
                             figure_size=(10, 6)))
                
                p13.save(output_dir / 'eficiencia_gpu.png', dpi=300, bbox_inches='tight')
                print(f"  Salvo: {output_dir / 'eficiencia_gpu.png'}")
            else:
                print("  Aviso: Nenhum dado GPU válido encontrado para eficiência")
        else:
            print("  Aviso: Nenhum dado GPU com utilização > 1% encontrado")

def create_cpu_power_plot(cpu_power_data, output_dir):
    """Cria plot de energia da CPU por estratégia e algoritmo"""
    print("Gerando plot de energia da CPU...")
    if cpu_power_data is not None:
        cpu_power_data = extract_strategy_and_algorithm(cpu_power_data)
        cpu_power_data = set_strategy_order(cpu_power_data)
        if 'CPU_POWER_W_median' in cpu_power_data.columns:
            cpu_power_data_filt = remove_outliers_iqr(cpu_power_data, 'CPU_POWER_W_median', ['strategy', 'algorithm'])
            p_cpu = (
                ggplot(cpu_power_data_filt, aes(x='strategy', y='CPU_POWER_W_median', fill='algorithm')) +
                geom_boxplot(outlier_shape=None, outlier_size=2, width=0.7, alpha=0.7) +
                labs(title='Distribuição do Consumo de Energia da CPU por Estratégia e Algoritmo',
                     x='Estratégia', y='Potência CPU (W)', fill='Algoritmo') +
                theme(axis_text_x=element_text(rotation=45, hjust=1),
                      figure_size=(10, 6))
            )
            p_cpu.save(output_dir / 'energia_cpu.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / 'energia_cpu.png'}")
        else:
            print("  Aviso: Coluna CPU_POWER_W_median não encontrada em cpu_power_data")

def create_io_ssd_plot(ssd_io_data, output_dir):
    """Cria plot de IO do SSD por estratégia e algoritmo"""
    print("Gerando plot de IO do SSD...")
    if ssd_io_data is not None:
        ssd_io_data = extract_strategy_and_algorithm(ssd_io_data)
        ssd_io_data = set_strategy_order(ssd_io_data)
        if 'READ_BYTES_median' in ssd_io_data.columns and 'WRITE_BYTES_median' in ssd_io_data.columns:
            ssd_io_data['TOTAL_BYTES'] = ssd_io_data['READ_BYTES_median'] + ssd_io_data['WRITE_BYTES_median']
            ssd_io_data['TOTAL_MB'] = ssd_io_data['TOTAL_BYTES'] / (1024 * 1024)
            ssd_io_data_filt = remove_outliers_iqr(ssd_io_data, 'TOTAL_MB', ['strategy', 'algorithm'])
            p_io = (
                ggplot(ssd_io_data_filt, aes(x='strategy', y='TOTAL_MB', fill='algorithm')) +
                geom_boxplot(outlier_shape=None, outlier_size=2, width=0.7, alpha=0.7) +
                labs(title='Distribuição do IO do SSD por Estratégia e Algoritmo',
                     x='Estratégia', y='IO SSD (MB)', fill='Algoritmo') +
                theme(axis_text_x=element_text(rotation=45, hjust=1),
                      figure_size=(10, 6))
            )
            p_io.save(output_dir / 'io_ssd.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / 'io_ssd.png'}")
        else:
            print("  Aviso: Colunas READ_BYTES_median ou WRITE_BYTES_median não encontradas em ssd_io_data")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Gerar plots dos dados de tempo e recursos Landsat')
    parser.add_argument('--results-dir', type=str, default='summarized_results',
                       help='Diretório com os dados sumarizados (padrão: summarized_results)')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Diretório de saída para os plots (padrão: plots)')
    parser.add_argument('--verbose', action='store_true',
                       help='Modo verboso')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"Erro: Diretório {results_dir} não encontrado!")
        sys.exit(1)
    
    # Criar diretório de saída se não existir
    output_dir.mkdir(exist_ok=True)
    
    if args.verbose:
        print(f"Carregando dados de {results_dir}")
        print(f"Salvando plots em {output_dir}")
    
    # Carregar dados
    time_data = load_time_data(results_dir)
    cpu_data = load_resource_data(results_dir, 'cpu')
    gpu_data = load_resource_data(results_dir, 'gpu')
    cpu_by_phase_data = load_resource_by_phase_data(results_dir, 'cpu')
    gpu_by_phase_data = load_resource_by_phase_data(results_dir, 'gpu')
    ssd_io_data = load_ssd_io_data(results_dir)
    cpu_power_data = load_cpu_power_data(results_dir)
    
    if time_data is None:
        print("Erro: Nenhum dado de tempo encontrado!")
        sys.exit(1)
    
    # Gerar plots
    create_time_plots(time_data, output_dir)
    create_time_plots_filtered(time_data, output_dir) # Adicionado o novo plot
    create_time_plots_filtered_with_read(time_data, output_dir) # Novo plot com leitura
    create_time_heatmaps_seconds(time_data, output_dir) # Novo plot em segundos
    create_resource_plots(cpu_data, gpu_data, output_dir)
    create_resource_by_phase_plots(cpu_by_phase_data, gpu_by_phase_data, time_data, output_dir)
    create_total_time_stacked_plot(time_data, output_dir)
    create_total_time_stacked_plot_gpu(time_data, output_dir)
    create_total_time_normal_plot(time_data, output_dir)
    create_total_time_normal_plot_gpu(time_data, output_dir)
    create_total_time_boxplot(time_data, output_dir)
    create_total_time_boxplot_gpu(time_data, output_dir)
    create_summary_plots(time_data, cpu_data, gpu_data, output_dir)
    create_cpu_power_plot(cpu_power_data, output_dir)
    create_io_ssd_plot(ssd_io_data, output_dir)
    
    print(f"\nTodos os plots foram gerados em: {output_dir}")

if __name__ == '__main__':
    main() 