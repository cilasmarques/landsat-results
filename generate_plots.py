#!/usr/bin/env python3
"""
Script para gerar plots dos dados de tempo e utilização de recursos das estratégias Landsat.

Este script cria visualizações usando plotnine (ggplot para Python) para:
1. Tempos de processamento das fases principais
2. Utilização de CPU e GPU por fase
3. Comparações entre estratégias e algoritmos
4. Análise de eficiência por fase
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
    strategy_order = ['serial', 'kernels-raw', 'kernels-streams', 'kernels-pc']
    df['strategy'] = df['strategy'].replace({'kernels-streams-async': 'kernels-streams'})
    df['strategy'] = pd.Categorical(df['strategy'], categories=strategy_order, ordered=True)
    return df

def set_algorithm_labels(df):
    """Substitui kernels-0 por STEEP e kernels-1 por SEBAL na coluna algorithm e ajusta legendas"""
    df = df.copy()
    if 'algorithm' in df.columns:
        df['algorithm'] = df['algorithm'].replace({'kernels-0': 'STEEP', 'kernels-1': 'SEBAL'})
    return df

def set_strategy_labels(df):
    """Substitui kernels-streams-async por kernels-streams na coluna strategy"""
    df = df.copy()
    if 'strategy' in df.columns:
        df['strategy'] = df['strategy'].replace({'kernels-streams-async': 'kernels-streams'})
    return df

def load_time_data(results_dir):
    """Carrega todos os dados de tempo dos arquivos final-time.csv"""
    all_data = []
    
    # Estratégias e algoritmos (ordem específica para os plots)
    # Usar nomes reais das pastas para carregar dados
    strategies = ['serial', 'kernels-raw', 'kernels-streams-async', 'kernels-pc']
    algorithms = ['kernels-0', 'kernels-1']
    
    # Fases principais que queremos analisar
    main_phases = ['P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 
                   'P3_RAH', 'P4_FINAL_PROD', 'P5_COPY_HOST', 'P6_SAVE_PRODS']
    
    for strategy in strategies:
        for algorithm in algorithms:
            file_path = Path(results_dir) / strategy / algorithm / 'final-time.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Filtra apenas as fases principais
                df = df[df['PHASE'].isin(main_phases)]
                # Não sobrescrever a coluna strategy que já existe
                df['algorithm'] = algorithm
                all_data.append(df)
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        # Remover kernels-streams-seq se existir
        combined_data = combined_data[combined_data['strategy'] != 'kernels-streams-seq']
        return combined_data
    return None

def load_resource_data(results_dir, resource_type):
    """Carrega dados de CPU ou GPU"""
    all_data = []
    
    # Usar nomes reais das pastas para carregar dados
    strategies = ['serial', 'kernels-raw', 'kernels-streams-async', 'kernels-pc']
    algorithms = ['kernels-0', 'kernels-1']
    
    for strategy in strategies:
        for algorithm in algorithms:
            if resource_type == 'gpu' and strategy == 'serial':
                continue  # Serial não tem dados de GPU
                
            file_path = Path(results_dir) / strategy / algorithm / f'{resource_type}-time.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['algorithm'] = algorithm
                df['resource_type'] = resource_type
                all_data.append(df)
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        # Remover kernels-streams-seq se existir
        combined_data = combined_data[combined_data['strategy'] != 'kernels-streams-seq']
        return combined_data
    return None

def load_resource_by_phase_data(results_dir, resource_type):
    """Carrega dados de CPU ou GPU mapeados por fase"""
    all_data = []
    
    # Usar nomes reais das pastas para carregar dados
    strategies = ['serial', 'kernels-raw', 'kernels-streams-async', 'kernels-pc']
    algorithms = ['kernels-0', 'kernels-1']
    
    for strategy in strategies:
        for algorithm in algorithms:
            if resource_type == 'gpu' and strategy == 'serial':
                continue  # Serial não tem dados de GPU
                
            file_path = Path(results_dir) / strategy / algorithm / f'{resource_type}-by-phase.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['algorithm'] = algorithm
                df['resource_type'] = resource_type
                all_data.append(df)
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        # Remover kernels-streams-seq se existir
        combined_data = combined_data[combined_data['strategy'] != 'kernels-streams-seq']
        return combined_data
    return None

def create_time_plots(time_data, output_dir):
    """Cria plots de tempo de processamento"""
    print("Gerando plots de tempo de processamento...")
    
    # Garantir que não há valores nulos e que é string
    valid_strategies = ['serial', 'kernels-raw', 'kernels-streams-async', 'kernels-pc']
    time_data = time_data[time_data['strategy'].isin(valid_strategies)]
    time_data = time_data[time_data['strategy'].apply(lambda x: isinstance(x, str))]
    time_data['strategy'] = time_data['strategy'].astype(str)
    # Aplicar ordem das estratégias e labels
    time_data = set_strategy_order(time_data)
    time_data = set_algorithm_labels(time_data)
    time_data = set_strategy_labels(time_data)
    
    # Plot 1: Comparação de tempos por fase e estratégia (STEEP)
    df_k0 = time_data[time_data['algorithm'] == 'STEEP'].copy()
    
    p1 = (ggplot(df_k0, aes(x='PHASE', y='TIMESTAMP_median', fill='strategy')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Tempo de Processamento por Fase - Algoritmo STEEP',
               x='Fase', y='Tempo (segundos)', fill='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(12, 8),
                legend_position='none'))
    
    p1.save(output_dir / 'tempo_por_fase_steep.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'tempo_por_fase_steep.png'}")
    
    # Plot 2: Comparação de tempos por fase e estratégia (SEBAL)
    df_k1 = time_data[time_data['algorithm'] == 'SEBAL'].copy()
    
    p2 = (ggplot(df_k1, aes(x='PHASE', y='TIMESTAMP_median', fill='strategy')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Tempo de Processamento por Fase - Algoritmo SEBAL',
               x='Fase', y='Tempo (segundos)', fill='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(12, 8),
                legend_position='none'))
    
    p2.save(output_dir / 'tempo_por_fase_sebal.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'tempo_por_fase_sebal.png'}")
    
    # Plot 3: Comparação entre algoritmos para cada estratégia
    p3 = (ggplot(time_data, aes(x='PHASE', y='TIMESTAMP_median', fill='algorithm')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Comparação de Tempos entre Algoritmos por Estratégia',
               x='Fase', y='Tempo (segundos)', fill='Algoritmo') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(14, 10)))
    
    p3.save(output_dir / 'comparacao_algoritmos.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'comparacao_algoritmos.png'}")
    
    # Plot 4: Heatmap de tempos combinado (STEEP em cima, SEBAL embaixo)
    heatmap_data_combined = []
    for algorithm in ['STEEP', 'SEBAL']:
        df_alg = time_data[time_data['algorithm'] == algorithm].copy()
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
    combined_heatmap_data['algorithm_name'] = pd.Categorical(
        combined_heatmap_data['algorithm_name'], 
        categories=['STEEP', 'SEBAL'], 
        ordered=True
    )
    p4 = (ggplot(combined_heatmap_data, aes(x='PHASE', y='strategy', fill='TIMESTAMP_median')) +
          geom_tile() +
          geom_text(aes(label=combined_heatmap_data['TIMESTAMP_median'].round(1)), 
                   size=8, color='white') +
          scale_fill_gradient(low='blue', high='red', name='Tempo (s)') +
          facet_wrap('~algorithm_name', ncol=1, scales='free_y') +
          labs(title='Heatmap de Tempos de Processamento por Algoritmo',
               x='Fase', y='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(14, 10)))
    p4.save(output_dir / 'heatmap_tempos_combinado.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'heatmap_tempos_combinado.png'}")
    
    # Plot 4.1: Heatmap de tempos separados (STEEP e SEBAL individuais)
    for algorithm in ['STEEP', 'SEBAL']:
        df_alg = time_data[time_data['algorithm'] == algorithm].copy()
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
        p4_individual = (ggplot(heatmap_data, aes(x='PHASE', y='strategy', fill='TIMESTAMP_median')) +
                        geom_tile() +
                        geom_text(aes(label=heatmap_data['TIMESTAMP_median'].round(1)), 
                                 size=8, color='white') +
                        scale_fill_gradient(low='blue', high='red', name='Tempo (s)') +
                        labs(title=f'Heatmap de Tempos de Processamento - {algorithm}',
                             x='Fase', y='Estratégia') +
                        theme(axis_text_x=element_text(rotation=45, hjust=1),
                              figure_size=(12, 6)))
        p4_individual.save(output_dir / f'heatmap_tempos_{algorithm.lower()}.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / f'heatmap_tempos_{algorithm.lower()}.png'}")

def create_resource_plots(cpu_data, gpu_data, output_dir):
    """Cria plots de utilização de recursos"""
    print("Gerando plots de utilização de recursos...")
    
    valid_strategies = ['serial', 'kernels-raw', 'kernels-streams-async', 'kernels-pc']
    if cpu_data is not None:
        cpu_data = cpu_data[cpu_data['strategy'].isin(valid_strategies)]
        cpu_data = cpu_data[cpu_data['strategy'].apply(lambda x: isinstance(x, str))]
        cpu_data['strategy'] = cpu_data['strategy'].astype(str)
        # Aplicar ordem das estratégias e labels
        cpu_data = set_strategy_order(cpu_data)
        cpu_data = set_algorithm_labels(cpu_data)
        cpu_data = set_strategy_labels(cpu_data)
        
        # Boxplot CPU: Utilização máxima por execução
        p5 = (ggplot(cpu_data, aes(x='strategy', y='CPU_USAGE_PERCENTAGE_max', fill='algorithm')) +
              geom_boxplot(outlier_shape='o', outlier_size=2, width=0.7, alpha=0.7) +
              labs(title='Distribuição da Utilização de CPU por Estratégia e Algoritmo',
                   x='Estratégia', y='CPU (%)', fill='Algoritmo') +
              theme(axis_text_x=element_text(rotation=45, hjust=1),
                    figure_size=(10, 6)))
        p5.save(output_dir / 'utilizacao_cpu.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'utilizacao_cpu.png'}")

        # Boxplot Memória RAM
        p6 = (ggplot(cpu_data, aes(x='strategy', y='MEM_USAGE_PERCENTAGE_max', fill='algorithm')) +
              geom_boxplot(outlier_shape='o', outlier_size=2, width=0.7, alpha=0.7) +
              labs(title='Distribuição da Utilização de Memória RAM por Estratégia e Algoritmo',
                   x='Estratégia', y='Memória (%)', fill='Algoritmo') +
              theme(axis_text_x=element_text(rotation=45, hjust=1),
                    figure_size=(10, 6)))
        p6.save(output_dir / 'utilizacao_memoria.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'utilizacao_memoria.png'}")

    if gpu_data is not None:
        gpu_data = gpu_data[gpu_data['strategy'].isin(valid_strategies)]
        gpu_data = gpu_data[gpu_data['strategy'].apply(lambda x: isinstance(x, str))]
        gpu_data['strategy'] = gpu_data['strategy'].astype(str)
        # Aplicar ordem das estratégias e labels
        gpu_data = set_strategy_order(gpu_data)
        gpu_data = set_algorithm_labels(gpu_data)
        gpu_data = set_strategy_labels(gpu_data)
        
        # Boxplot GPU: Utilização máxima por execução
        p7 = (ggplot(gpu_data, aes(x='strategy', y='GPU_USAGE_PERCENTAGE_max', fill='algorithm')) +
              geom_boxplot(outlier_shape='o', outlier_size=2, width=0.7, alpha=0.7) +
              labs(title='Distribuição da Utilização de GPU por Estratégia e Algoritmo',
                   x='Estratégia', y='GPU (%)', fill='Algoritmo') +
              theme(axis_text_x=element_text(rotation=45, hjust=1),
                    figure_size=(10, 6)))
        p7.save(output_dir / 'utilizacao_gpu.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'utilizacao_gpu.png'}")

        # Boxplot Memória GPU
        p8 = (ggplot(gpu_data, aes(x='strategy', y='MEM_USAGE_PERCENTAGE_max', fill='algorithm')) +
              geom_boxplot(outlier_shape='o', outlier_size=2, width=0.7, alpha=0.7) +
              labs(title='Distribuição da Utilização de Memória GPU por Estratégia e Algoritmo',
                   x='Estratégia', y='Memória GPU (%)', fill='Algoritmo') +
              theme(axis_text_x=element_text(rotation=45, hjust=1),
                    figure_size=(10, 6)))
        p8.save(output_dir / 'utilizacao_memoria_gpu.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'utilizacao_memoria_gpu.png'}")

def create_resource_by_phase_plots(cpu_by_phase_data, gpu_by_phase_data, time_data, output_dir):
    """Cria plots de utilização de recursos por fase"""
    print("Gerando plots de utilização de recursos por fase...")
    
    # Fases principais que queremos analisar
    main_phases = ['P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 
                   'P3_RAH', 'P4_FINAL_PROD', 'P5_COPY_HOST', 'P6_SAVE_PRODS']
    
    valid_strategies = ['serial', 'kernels-raw', 'kernels-streams-async', 'kernels-pc']
    if cpu_by_phase_data is not None:
        cpu_by_phase_data = cpu_by_phase_data[cpu_by_phase_data['strategy'].isin(valid_strategies)]
        cpu_by_phase_data = cpu_by_phase_data[cpu_by_phase_data['strategy'].apply(lambda x: isinstance(x, str))]
        cpu_by_phase_data['strategy'] = cpu_by_phase_data['strategy'].astype(str)
        # Aplicar ordem das estratégias e labels
        cpu_by_phase_data = set_strategy_order(cpu_by_phase_data)
        cpu_by_phase_data = set_algorithm_labels(cpu_by_phase_data)
        cpu_by_phase_data = set_strategy_labels(cpu_by_phase_data)
        
        # Filtrar apenas fases principais
        cpu_by_phase_data = cpu_by_phase_data[cpu_by_phase_data['PHASE'].isin(main_phases)]
        
        # Plot CPU por fase - Mediana dos máximos
        p11 = (ggplot(cpu_by_phase_data, aes(x='PHASE', y='CPU_USAGE_PERCENTAGE_max_median', fill='strategy')) +
               geom_bar(stat='identity', position='dodge', width=0.7) +
               facet_wrap('~algorithm', scales='free_y', ncol=2) +
               labs(title='Utilização de CPU por Fase (Mediana dos Máximos)',
                    x='Fase', y='CPU (%)', fill='Estratégia') +
               theme(axis_text_x=element_text(rotation=45, hjust=1),
                     figure_size=(14, 8)))
        
        p11.save(output_dir / 'cpu_por_fase.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'cpu_por_fase.png'}")
        
        # Plot CPU Memory por fase
        p12 = (ggplot(cpu_by_phase_data, aes(x='PHASE', y='MEM_USAGE_PERCENTAGE_max_median', fill='strategy')) +
               geom_bar(stat='identity', position='dodge', width=0.7) +
               facet_wrap('~algorithm', scales='free_y', ncol=2) +
               labs(title='Utilização de Memória RAM por Fase (Mediana dos Máximos)',
                    x='Fase', y='Memória (%)', fill='Estratégia') +
               theme(axis_text_x=element_text(rotation=45, hjust=1),
                     figure_size=(14, 8)))
        
        p12.save(output_dir / 'memoria_por_fase.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'memoria_por_fase.png'}")
    
    if gpu_by_phase_data is not None:
        gpu_by_phase_data = gpu_by_phase_data[gpu_by_phase_data['strategy'].isin(valid_strategies)]
        gpu_by_phase_data = gpu_by_phase_data[gpu_by_phase_data['strategy'].apply(lambda x: isinstance(x, str))]
        gpu_by_phase_data['strategy'] = gpu_by_phase_data['strategy'].astype(str)
        # Aplicar ordem das estratégias e labels
        gpu_by_phase_data = set_strategy_order(gpu_by_phase_data)
        gpu_by_phase_data = set_algorithm_labels(gpu_by_phase_data)
        gpu_by_phase_data = set_strategy_labels(gpu_by_phase_data)
        
        # Filtrar apenas fases principais
        gpu_by_phase_data = gpu_by_phase_data[gpu_by_phase_data['PHASE'].isin(main_phases)]
        
        # Plot GPU por fase - Mediana dos máximos
        p13 = (ggplot(gpu_by_phase_data, aes(x='PHASE', y='GPU_USAGE_PERCENTAGE_max_median', fill='strategy')) +
               geom_bar(stat='identity', position='dodge', width=0.7) +
               facet_wrap('~algorithm', scales='free_y', ncol=2) +
               labs(title='Utilização de GPU por Fase (Mediana dos Máximos)',
                    x='Fase', y='GPU (%)', fill='Estratégia') +
               theme(axis_text_x=element_text(rotation=45, hjust=1),
                     figure_size=(14, 8)))
        
        p13.save(output_dir / 'gpu_por_fase.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'gpu_por_fase.png'}")
        
        # Plot GPU Memory por fase
        p14 = (ggplot(gpu_by_phase_data, aes(x='PHASE', y='MEM_USAGE_PERCENTAGE_max_median', fill='strategy')) +
               geom_bar(stat='identity', position='dodge', width=0.7) +
               facet_wrap('~algorithm', scales='free_y', ncol=2) +
               labs(title='Utilização de Memória GPU por Fase (Mediana dos Máximos)',
                    x='Fase', y='Memória GPU (%)', fill='Estratégia') +
               theme(axis_text_x=element_text(rotation=45, hjust=1),
                     figure_size=(14, 8)))
        
        p14.save(output_dir / 'memoria_gpu_por_fase.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'memoria_gpu_por_fase.png'}")
        
        # Plot GPU Power por fase
        p15 = (ggplot(gpu_by_phase_data, aes(x='PHASE', y='POWER_W_median_median', fill='strategy')) +
               geom_bar(stat='identity', position='dodge', width=0.7) +
               facet_wrap('~algorithm', scales='free_y', ncol=2) +
               labs(title='Consumo de Energia GPU por Fase (Mediana dos Medianos)',
                    x='Fase', y='Potência (W)', fill='Estratégia') +
               theme(axis_text_x=element_text(rotation=45, hjust=1),
                     figure_size=(14, 8)))
        
        p15.save(output_dir / 'energia_gpu_por_fase.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'energia_gpu_por_fase.png'}")
        
        # Plot GPU Temperature por fase
        p16 = (ggplot(gpu_by_phase_data, aes(x='PHASE', y='TEMP_C_median_median', fill='strategy')) +
               geom_bar(stat='identity', position='dodge', width=0.7) +
               facet_wrap('~algorithm', scales='free_y', ncol=2) +
               labs(title='Temperatura GPU por Fase (Mediana dos Medianos)',
                    x='Fase', y='Temperatura (°C)', fill='Estratégia') +
               theme(axis_text_x=element_text(rotation=45, hjust=1),
                     figure_size=(14, 8)))
        
        p16.save(output_dir / 'temperatura_gpu_por_fase.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'temperatura_gpu_por_fase.png'}")

def create_efficiency_plots(cpu_by_phase_data, gpu_by_phase_data, time_data, output_dir):
    """Cria plots de eficiência por fase"""
    print("Gerando plots de eficiência por fase...")
    
    # Fases principais que queremos analisar
    main_phases = ['P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 
                   'P3_RAH', 'P4_FINAL_PROD', 'P5_COPY_HOST', 'P6_SAVE_PRODS']
    
    valid_strategies = ['serial', 'kernels-raw', 'kernels-streams-async', 'kernels-pc']
    if cpu_by_phase_data is not None and time_data is not None:
        cpu_by_phase_data = cpu_by_phase_data[cpu_by_phase_data['strategy'].isin(valid_strategies)]
        cpu_by_phase_data = cpu_by_phase_data[cpu_by_phase_data['strategy'].apply(lambda x: isinstance(x, str))]
        cpu_by_phase_data['strategy'] = cpu_by_phase_data['strategy'].astype(str)
        time_data = time_data[time_data['strategy'].isin(valid_strategies)]
        time_data = time_data[time_data['strategy'].apply(lambda x: isinstance(x, str))]
        time_data['strategy'] = time_data['strategy'].astype(str)
        # Aplicar ordem das estratégias e labels
        time_data = set_strategy_order(time_data)
        time_data = set_algorithm_labels(time_data)
        time_data = set_strategy_labels(time_data)
        
        # Filtrar apenas fases principais
        cpu_by_phase_data = cpu_by_phase_data[cpu_by_phase_data['PHASE'].isin(main_phases)]
        time_data_filtered = time_data[time_data['PHASE'].isin(main_phases)]
        
        # Combinar dados de tempo e CPU
        efficiency_data = time_data_filtered.merge(
            cpu_by_phase_data[['strategy', 'algorithm', 'PHASE', 'CPU_USAGE_PERCENTAGE_max_median']], 
            on=['strategy', 'algorithm', 'PHASE'], 
            how='inner'
        )
        
        # Calcular eficiência (tempo / utilização de CPU)
        efficiency_data['cpu_efficiency'] = efficiency_data['TIMESTAMP_median'] / efficiency_data['CPU_USAGE_PERCENTAGE_max_median']
        
        if not efficiency_data.empty:
            # Plot eficiência CPU por fase
            p17 = (ggplot(efficiency_data, aes(x='PHASE', y='cpu_efficiency', fill='strategy')) +
                   geom_bar(stat='identity', position='dodge', width=0.7) +
                   facet_wrap('~algorithm', scales='free_y', ncol=2) +
                   labs(title='Eficiência CPU por Fase (Tempo/Utilização)',
                        x='Fase', y='Eficiência CPU', fill='Estratégia') +
                   theme(axis_text_x=element_text(rotation=45, hjust=1),
                         figure_size=(14, 8)))
            p17.save(output_dir / 'eficiencia_cpu_por_fase.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / 'eficiencia_cpu_por_fase.png'}")
        else:
            print("[AVISO] Não há dados suficientes para plotar eficiência CPU por fase.")
    
    if gpu_by_phase_data is not None and time_data is not None:
        gpu_by_phase_data = gpu_by_phase_data[gpu_by_phase_data['strategy'].isin(valid_strategies)]
        gpu_by_phase_data = gpu_by_phase_data[gpu_by_phase_data['strategy'].apply(lambda x: isinstance(x, str))]
        gpu_by_phase_data['strategy'] = gpu_by_phase_data['strategy'].astype(str)
        time_data = time_data[time_data['strategy'].isin(valid_strategies)]
        time_data = time_data[time_data['strategy'].apply(lambda x: isinstance(x, str))]
        time_data['strategy'] = time_data['strategy'].astype(str)
        # Aplicar ordem das estratégias e labels
        time_data = set_strategy_order(time_data)
        time_data = set_algorithm_labels(time_data)
        time_data = set_strategy_labels(time_data)
        
        # Filtrar apenas fases principais
        gpu_by_phase_data = gpu_by_phase_data[gpu_by_phase_data['PHASE'].isin(main_phases)]
        time_data_filtered = time_data[time_data['PHASE'].isin(main_phases)]
        
        # Combinar dados de tempo e GPU
        efficiency_data = time_data_filtered.merge(
            gpu_by_phase_data[['strategy', 'algorithm', 'PHASE', 'GPU_USAGE_PERCENTAGE_max_median']], 
            on=['strategy', 'algorithm', 'PHASE'], 
            how='inner'
        )
        
        # Calcular eficiência (tempo / utilização de GPU)
        efficiency_data['gpu_efficiency'] = efficiency_data['TIMESTAMP_median'] / efficiency_data['GPU_USAGE_PERCENTAGE_max_median']
        
        if not efficiency_data.empty:
            # Plot eficiência GPU por fase
            p18 = (ggplot(efficiency_data, aes(x='PHASE', y='gpu_efficiency', fill='strategy')) +
                   geom_bar(stat='identity', position='dodge', width=0.7) +
                   facet_wrap('~algorithm', scales='free_y', ncol=2) +
                   labs(title='Eficiência GPU por Fase (Tempo/Utilização)',
                        x='Fase', y='Eficiência GPU', fill='Estratégia') +
                   theme(axis_text_x=element_text(rotation=45, hjust=1),
                         figure_size=(14, 8)))
            p18.save(output_dir / 'eficiencia_gpu_por_fase.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / 'eficiencia_gpu_por_fase.png'}")
        else:
            print("[AVISO] Não há dados suficientes para plotar eficiência GPU por fase.")

def create_heatmap_plots(cpu_by_phase_data, gpu_by_phase_data, output_dir):
    """Cria heatmaps de utilização de recursos por fase"""
    print("Gerando heatmaps de utilização de recursos por fase...")
    
    # Fases principais que queremos analisar
    main_phases = ['P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 
                   'P3_RAH', 'P4_FINAL_PROD', 'P5_COPY_HOST', 'P6_SAVE_PRODS']
    
    valid_strategies = ['serial', 'kernels-raw', 'kernels-streams-async', 'kernels-pc']
    if cpu_by_phase_data is not None:
        cpu_by_phase_data = cpu_by_phase_data[cpu_by_phase_data['strategy'].isin(valid_strategies)]
        cpu_by_phase_data = cpu_by_phase_data[cpu_by_phase_data['strategy'].apply(lambda x: isinstance(x, str))]
        cpu_by_phase_data['strategy'] = cpu_by_phase_data['strategy'].astype(str)
        # Aplicar ordem das estratégias e labels
        cpu_by_phase_data = set_strategy_order(cpu_by_phase_data)
        cpu_by_phase_data = set_algorithm_labels(cpu_by_phase_data)
        cpu_by_phase_data = set_strategy_labels(cpu_by_phase_data)
        
        # Filtrar apenas fases principais
        cpu_by_phase_data = cpu_by_phase_data[cpu_by_phase_data['PHASE'].isin(main_phases)]
        
        # Criar heatmap de CPU por fase
        for algorithm, algorithm_name in [('STEEP', 'STEEP'), ('SEBAL', 'SEBAL')]:
            df_alg = cpu_by_phase_data[cpu_by_phase_data['algorithm'] == algorithm].copy()
            
            # Pivotar dados para heatmap
            pivot_data = df_alg.pivot_table(
                values='CPU_USAGE_PERCENTAGE_max_median', 
                index='strategy', 
                columns='PHASE', 
                aggfunc='mean'
            )
            
            # Converte para formato longo para o plot
            heatmap_data = pivot_data.reset_index().melt(
                id_vars=['strategy'], 
                var_name='PHASE', 
                value_name='CPU_USAGE_PERCENTAGE_max_median'
            )
            
            p19 = (ggplot(heatmap_data, aes(x='PHASE', y='strategy', fill='CPU_USAGE_PERCENTAGE_max_median')) +
                   geom_tile() +
                   geom_text(aes(label=heatmap_data['CPU_USAGE_PERCENTAGE_max_median'].round(1)), 
                            size=8, color='white') +
                   scale_fill_gradient(low='blue', high='red', name='CPU (%)') +
                   labs(title=f'Heatmap de Utilização de CPU por Fase - {algorithm_name}',
                        x='Fase', y='Estratégia') +
                   theme(axis_text_x=element_text(rotation=45, hjust=1),
                         figure_size=(12, 6)))
            
            p19.save(output_dir / f'heatmap_cpu_por_fase_{algorithm_name.lower()}.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / f'heatmap_cpu_por_fase_{algorithm_name.lower()}.png'}")
    
    if gpu_by_phase_data is not None:
        gpu_by_phase_data = gpu_by_phase_data[gpu_by_phase_data['strategy'].isin(valid_strategies)]
        gpu_by_phase_data = gpu_by_phase_data[gpu_by_phase_data['strategy'].apply(lambda x: isinstance(x, str))]
        gpu_by_phase_data['strategy'] = gpu_by_phase_data['strategy'].astype(str)
        # Aplicar ordem das estratégias e labels
        gpu_by_phase_data = set_strategy_order(gpu_by_phase_data)
        gpu_by_phase_data = set_algorithm_labels(gpu_by_phase_data)
        gpu_by_phase_data = set_strategy_labels(gpu_by_phase_data)
        
        # Filtrar apenas fases principais
        gpu_by_phase_data = gpu_by_phase_data[gpu_by_phase_data['PHASE'].isin(main_phases)]
        
        # Criar heatmap de GPU por fase
        for algorithm, algorithm_name in [('STEEP', 'STEEP'), ('SEBAL', 'SEBAL')]:
            df_alg = gpu_by_phase_data[gpu_by_phase_data['algorithm'] == algorithm].copy()
            
            # Pivotar dados para heatmap
            pivot_data = df_alg.pivot_table(
                values='GPU_USAGE_PERCENTAGE_max_median', 
                index='strategy', 
                columns='PHASE', 
                aggfunc='mean'
            )
            
            # Converte para formato longo para o plot
            heatmap_data = pivot_data.reset_index().melt(
                id_vars=['strategy'], 
                var_name='PHASE', 
                value_name='GPU_USAGE_PERCENTAGE_max_median'
            )
            
            p20 = (ggplot(heatmap_data, aes(x='PHASE', y='strategy', fill='GPU_USAGE_PERCENTAGE_max_median')) +
                   geom_tile() +
                   geom_text(aes(label=heatmap_data['GPU_USAGE_PERCENTAGE_max_median'].round(1)), 
                            size=8, color='white') +
                   scale_fill_gradient(low='blue', high='red', name='GPU (%)') +
                   labs(title=f'Heatmap de Utilização de GPU por Fase - {algorithm_name}',
                        x='Fase', y='Estratégia') +
                   theme(axis_text_x=element_text(rotation=45, hjust=1),
                         figure_size=(12, 6)))
            
            p20.save(output_dir / f'heatmap_gpu_por_fase_{algorithm_name.lower()}.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / f'heatmap_gpu_por_fase_{algorithm_name.lower()}.png'}")

def create_comparison_plots(time_data, output_dir):
    """Cria plots de comparação de tempos por fase e estratégia"""
    print("Gerando plots de comparação...")
    
    valid_strategies = ['serial', 'kernels-raw', 'kernels-streams-async', 'kernels-pc']
    time_data = time_data[time_data['strategy'].isin(valid_strategies)]
    time_data = time_data[time_data['strategy'].apply(lambda x: isinstance(x, str))]
    time_data['strategy'] = time_data['strategy'].astype(str)
    # Aplicar ordem das estratégias e labels
    time_data = set_strategy_order(time_data)
    time_data = set_algorithm_labels(time_data)
    time_data = set_strategy_labels(time_data)
    
    # Calcular a mediana do tempo para cada fase e estratégia
    mean_phases = time_data.groupby(['strategy', 'algorithm', 'PHASE']).agg({
        'TIMESTAMP_median': 'median'
    }).reset_index()
    
    # Converter para segundos (assumindo que está em milissegundos)
    mean_phases['time_seconds'] = mean_phases['TIMESTAMP_median'] / 1000
    
    # Definir ordem das fases
    phase_order = ['P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 'P3_RAH', 'P4_FINAL_PROD', 'P6_SAVE_PRODS']
    
    # Filtrar apenas as fases que queremos mostrar
    mean_phases = mean_phases[mean_phases['PHASE'].isin(phase_order)]
    
    # Criar diferentes visualizações
    # 1. Todas as abordagens
    df1 = mean_phases.copy()
    df1['version'] = 'Abordagem geral'
    
    # 2. Apenas abordagens GPU (excluir serial)
    df2 = mean_phases[mean_phases['strategy'] != 'serial'].copy()
    df2 = df2[df2['strategy'] != 'serial']  # Garante que serial não aparece
    df2['version'] = 'Apenas GPU'
    
    # Removido: 3. GPU sem leitura
    # df3 = mean_phases[(mean_phases['strategy'] != 'serial') & 
    #                  (mean_phases['PHASE'] != 'P0_READ_INPUT')].copy()
    # df3['version'] = '03_GPU_sem_leitura'
    
    # Combinar os dataframes (sem a versão 03)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Criar uma coluna combinada para o facet_wrap
    combined_df['algorithm_version'] = combined_df['algorithm'] + ' - ' + combined_df['version']
    
    # Definir cores para as fases
    phase_colors = {
        'P0_READ_INPUT': '#FF8080',
        'P1_INITIAL_PROD': '#FFB347', 
        'P2_PIXEL_SEL': '#FFFF66',
        'P3_RAH': '#90EE90',
        'P4_FINAL_PROD': '#8080FF',
        'P6_SAVE_PRODS': '#9B7CB9'
    }
    
    # Criar o plot
    p_comparison = (ggplot(combined_df, aes(x='strategy', y='time_seconds', fill='PHASE')) +
                   geom_bar(stat='identity', position='stack', width=0.8) +
                   facet_wrap('algorithm_version', scales='free_y', ncol=2) +
                   scale_fill_manual(values=phase_colors) +
                   labs(title='Comparação de Tempos de Execução por Fase e Estratégia',
                        y='Tempo (segundos)', x='Estratégia', fill='Fase') +
                   theme_bw() +
                   theme(legend_position='bottom',
                         figure_size=(15, 10),
                         axis_text_x=element_text(rotation=45, hjust=1)))
    
    p_comparison.save(output_dir / 'comparacao_tempos_fases.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'comparacao_tempos_fases.png'}")

def create_summary_plots(time_data, cpu_data, gpu_data, output_dir):
    """Cria plots de resumo geral"""
    print("Gerando plots de resumo...")
    
    valid_strategies = ['serial', 'kernels-raw', 'kernels-streams-async', 'kernels-pc']
    time_data = time_data[time_data['strategy'].isin(valid_strategies)]
    time_data = time_data[time_data['strategy'].apply(lambda x: isinstance(x, str))]
    time_data['strategy'] = time_data['strategy'].astype(str)
    # Aplicar ordem das estratégias e labels
    time_data = set_strategy_order(time_data)
    time_data = set_algorithm_labels(time_data)
    time_data = set_strategy_labels(time_data)
    
    # Calcula tempo total por estratégia e algoritmo
    total_time = time_data.groupby(['strategy', 'algorithm'])['TIMESTAMP_median'].sum().reset_index()
    total_time.columns = ['strategy', 'algorithm', 'total_time']
    
    p11 = (ggplot(total_time, aes(x='strategy', y='total_time', fill='algorithm')) +
           geom_bar(stat='identity', position='dodge', width=0.7) +
           labs(title='Tempo Total de Processamento por Estratégia e Algoritmo',
                x='Estratégia', y='Tempo Total (segundos)', fill='Algoritmo') +
           theme(axis_text_x=element_text(rotation=45, hjust=1),
                 figure_size=(10, 6)))
    
    p11.save(output_dir / 'tempo_total.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'tempo_total.png'}")
    
    # Comparação de eficiência (tempo vs recursos)
    if cpu_data is not None and gpu_data is not None:
        # Aplicar labels aos dados de recursos
        cpu_data = set_algorithm_labels(cpu_data)
        cpu_data = set_strategy_labels(cpu_data)
        gpu_data = set_algorithm_labels(gpu_data)
        gpu_data = set_strategy_labels(gpu_data)
        
        # Combina dados de tempo e recursos
        efficiency_data = total_time.copy()
        
        # Adiciona dados de CPU (usando mediana dos máximos)
        cpu_avg = cpu_data.groupby(['strategy', 'algorithm']).agg({
            'CPU_USAGE_PERCENTAGE_max': 'median'
        }).reset_index()
        efficiency_data = efficiency_data.merge(cpu_avg, on=['strategy', 'algorithm'])
        
        # Adiciona dados de GPU (usando mediana dos máximos)
        gpu_avg = gpu_data.groupby(['strategy', 'algorithm']).agg({
            'GPU_USAGE_PERCENTAGE_max': 'median'
        }).reset_index()
        efficiency_data = efficiency_data.merge(gpu_avg, on=['strategy', 'algorithm'])
        
        # Calcula eficiência (tempo / utilização de recursos)
        efficiency_data['cpu_efficiency'] = efficiency_data['total_time'] / efficiency_data['CPU_USAGE_PERCENTAGE_max']
        efficiency_data['gpu_efficiency'] = efficiency_data['total_time'] / efficiency_data['GPU_USAGE_PERCENTAGE_max']
        
        p12 = (ggplot(efficiency_data, aes(x='strategy', y='cpu_efficiency', fill='algorithm')) +
               geom_bar(stat='identity', position='dodge', width=0.7) +
               labs(title='Eficiência CPU (Tempo/Utilização) por Estratégia e Algoritmo',
                    x='Estratégia', y='Eficiência CPU', fill='Algoritmo') +
               theme(axis_text_x=element_text(rotation=45, hjust=1),
                     figure_size=(10, 6)))
        
        p12.save(output_dir / 'eficiencia_cpu.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'eficiencia_cpu.png'}")

def main():
    parser = argparse.ArgumentParser(description='Gera plots dos dados de tempo e recursos Landsat')
    parser.add_argument('--input', '-i', type=str, default='results',
                       help='Caminho para o diretório results (padrão: results)')
    parser.add_argument('--output', '-o', type=str, default='plots',
                       help='Caminho para salvar os plots (padrão: plots)')
    
    args = parser.parse_args()
    
    # Configuração dos caminhos
    results_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Verifica se o diretório de entrada existe
    if not results_dir.exists():
        print(f"Erro: Diretório de entrada não encontrado: {results_dir}")
        sys.exit(1)
    
    # Cria o diretório de saída se não existir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== GERANDO PLOTS DOS DADOS LANDSAT ===")
    print(f"Diretório de entrada: {results_dir}")
    print(f"Diretório de saída: {output_dir}")
    
    # Carrega os dados
    print("\nCarregando dados...")
    time_data = load_time_data(results_dir)
    if time_data is not None:
        print(f"  Dados de tempo carregados: {len(time_data)} registros")
    else:
        print("  Erro: Nenhum dado de tempo encontrado")
        sys.exit(1)
    
    cpu_data = load_resource_data(results_dir, 'cpu')
    if cpu_data is not None:
        print(f"  Dados de CPU carregados: {len(cpu_data)} registros")
    
    gpu_data = load_resource_data(results_dir, 'gpu')
    if gpu_data is not None:
        print(f"  Dados de GPU carregados: {len(gpu_data)} registros")
    
    # Carrega dados mapeados por fase
    cpu_by_phase_data = load_resource_by_phase_data(results_dir, 'cpu')
    if cpu_by_phase_data is not None:
        print(f"  Dados de CPU por fase carregados: {len(cpu_by_phase_data)} registros")
    
    gpu_by_phase_data = load_resource_by_phase_data(results_dir, 'gpu')
    if gpu_by_phase_data is not None:
        print(f"  Dados de GPU por fase carregados: {len(gpu_by_phase_data)} registros")
    
    # Gera os plots
    print("\n=== GERANDO PLOTS ===")
    
    # Plots de tempo
    create_time_plots(time_data, output_dir)
    
    # Plots de recursos
    create_resource_plots(cpu_data, gpu_data, output_dir)

    # Plots de comparação
    create_comparison_plots(time_data, output_dir)
    
    # Plots de recursos por fase
    create_resource_by_phase_plots(cpu_by_phase_data, gpu_by_phase_data, time_data, output_dir)
    
    # Plots de eficiência por fase
    create_efficiency_plots(cpu_by_phase_data, gpu_by_phase_data, time_data, output_dir)
    
    # Plots de heatmap por fase
    create_heatmap_plots(cpu_by_phase_data, gpu_by_phase_data, output_dir)
    
    # Plots de resumo
    create_summary_plots(time_data, cpu_data, gpu_data, output_dir)
    
    print(f"\n=== PLOTS GERADOS COM SUCESSO ===")
    print(f"Todos os plots foram salvos em: {output_dir}")
    
    # Lista os arquivos gerados
    plot_files = list(output_dir.glob('*.png'))
    print(f"\nArquivos gerados ({len(plot_files)}):")
    for file in sorted(plot_files):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main() 