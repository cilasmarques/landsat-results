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
        'Serial',
        'Parallel 2',
        'Parallel 4',
        'Parallel 8',
        'Parallel 12',
        'Kernels Raw',
        'Kernels Streams'
    ]
    df['strategy'] = pd.Categorical(df['strategy'], categories=strategy_order, ordered=True)
    return df

def extract_strategy_and_algorithm(df):
    """Extrai estratégia e algoritmo da coluna strategy_algorithm"""
    df = df.copy()
    
    # Mapear estratégias e algoritmos
    strategy_mapping = {
        'serial': 'Serial',
        'parallel2': 'Parallel 2',
        'parallel4': 'Parallel 4', 
        'parallel8': 'Parallel 8',
        'parallel12': 'Parallel 12',
        'kernels-raw': 'Kernels Raw',
        'kernels-streams': 'Kernels Streams'
    }
    
    algorithm_mapping = {
        'sebal': 'SEBAL',
        'steep': 'STEEP'
    }
    
    # Extrair estratégia e algoritmo
    split_cols = df['strategy_algorithm'].str.rsplit('-', n=1, expand=True)
    df['strategy'] = split_cols[0]
    df['algorithm'] = split_cols[1]
    
    # Aplicar mapeamentos
    df['strategy'] = df['strategy'].map(strategy_mapping)
    df['algorithm'] = df['algorithm'].map(algorithm_mapping)
    
    return df

def load_time_data(results_dir):
    """Carrega todos os dados de tempo dos arquivos final-time.csv"""
    all_data = []
    
    # Lista de estratégias/algoritmos baseada na estrutura atual
    strategy_algorithms = [
        'serial-sebal', 'serial-steep',
        'parallel2-sebal', 'parallel2-steep',
        'parallel4-sebal', 'parallel4-steep',
        'parallel8-sebal', 'parallel8-steep', 
        'parallel12-sebal', 'parallel12-steep',
        'kernels-raw-sebal', 'kernels-raw-steep',
        'kernels-streams-sebal', 'kernels-streams-steep'
    ]
    
    # Fases principais que queremos analisar
    main_phases = ['P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 
                   'P3_RAH', 'P4_FINAL_PROD', 'P5_COPY_HOST']
    
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
        'serial-sebal', 'serial-steep',
        'parallel2-sebal', 'parallel2-steep',
        'parallel4-sebal', 'parallel4-steep',
        'parallel8-sebal', 'parallel8-steep', 
        'parallel12-sebal', 'parallel12-steep',
        'kernels-raw-sebal', 'kernels-raw-steep',
        'kernels-streams-sebal', 'kernels-streams-steep'
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
        'serial-sebal', 'serial-steep',
        'parallel2-sebal', 'parallel2-steep',
        'parallel4-sebal', 'parallel4-steep',
        'parallel8-sebal', 'parallel8-steep', 
        'parallel12-sebal', 'parallel12-steep',
        'kernels-raw-sebal', 'kernels-raw-steep',
        'kernels-streams-sebal', 'kernels-streams-steep'
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
        'serial-sebal', 'serial-steep',
        'parallel2-sebal', 'parallel2-steep',
        'parallel4-sebal', 'parallel4-steep',
        'parallel8-sebal', 'parallel8-steep', 
        'parallel12-sebal', 'parallel12-steep',
        'kernels-raw-sebal', 'kernels-raw-steep',
        'kernels-streams-sebal', 'kernels-streams-steep'
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
        'serial-sebal', 'serial-steep',
        'parallel2-sebal', 'parallel2-steep',
        'parallel4-sebal', 'parallel4-steep',
        'parallel8-sebal', 'parallel8-steep', 
        'parallel12-sebal', 'parallel12-steep',
        'kernels-raw-sebal', 'kernels-raw-steep',
        'kernels-streams-sebal', 'kernels-streams-steep'
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
    
    # Plot 1: Comparação de tempos por fase e estratégia (STEEP)
    df_steep = time_data[time_data['algorithm'] == 'STEEP'].copy()
    
    p1 = (ggplot(df_steep, aes(x='PHASE', y='TIMESTAMP_median', fill='strategy')) +
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
    df_sebal = time_data[time_data['algorithm'] == 'SEBAL'].copy()
    
    p2 = (ggplot(df_sebal, aes(x='PHASE', y='TIMESTAMP_median', fill='strategy')) +
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
    
    # Definir cores para as fases
    phase_colors = {
        'P0_READ_INPUT': '#FF8080',
        'P1_INITIAL_PROD': '#FFB347', 
        'P2_PIXEL_SEL': '#FFFF66',
        'P3_RAH': '#90EE90',
        'P4_FINAL_PROD': '#8080FF',
        'P6_SAVE_PRODS': '#9B7CB9'
    }
    
    p4 = (ggplot(combined_heatmap_data, aes(x='PHASE', y='strategy', fill='TIMESTAMP_median')) +
          geom_tile() +
          geom_text(aes(label=combined_heatmap_data['TIMESTAMP_median'].round(1)), 
                   size=8, color='black') +
          scale_fill_gradient(low='#2E5A88', high='#FF8080', name='Tempo (s)') +
          facet_wrap('~algorithm_name', ncol=1, scales='free_y') +
          labs(title='Heatmap de Tempos de Processamento por Algoritmo',
               x='Fase', y='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(14, 10)))
    p4.save(output_dir / 'heatmap_tempos_combinado.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'heatmap_tempos_combinado.png'}")
    
    # Plot 5: Heatmaps individuais por algoritmo
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
                                 size=8, color='black') +
                        scale_fill_gradient(low='#2E5A88', high='#FF8080', name='Tempo (s)') +
                        labs(title=f'Heatmap de Tempos de Processamento - {algorithm}',
                             x='Fase', y='Estratégia') +
                        theme(axis_text_x=element_text(rotation=45, hjust=1),
                              figure_size=(12, 6)))
        p4_individual.save(output_dir / f'heatmap_tempos_{algorithm.lower()}.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / f'heatmap_tempos_{algorithm.lower()}.png'}")

def create_resource_plots(cpu_data, gpu_data, output_dir):
    """Cria plots de utilização de recursos"""
    print("Gerando plots de utilização de recursos...")
    
    if cpu_data is not None:
        # Extrair estratégia e algoritmo
        cpu_data = extract_strategy_and_algorithm(cpu_data)
        cpu_data = set_strategy_order(cpu_data)
        
        # Boxplot CPU: Utilização máxima por execução
        p5 = (ggplot(cpu_data, aes(x='strategy', y='CPU_USAGE_PERCENTAGE_max', fill='algorithm')) +
              geom_boxplot(outlier_shape='o', outlier_size=2, width=0.7, alpha=0.7) +
              labs(title='Distribuição da Utilização de CPU por Estratégia e Algoritmo (Normalizado 0-100%)',
                   x='Estratégia', y='CPU (%)', fill='Algoritmo') +
              theme(axis_text_x=element_text(rotation=45, hjust=1),
                    figure_size=(10, 6)))
        p5.save(output_dir / 'utilizacao_cpu.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'utilizacao_cpu.png'}")

        # Boxplot Memória RAM
        p6 = (ggplot(cpu_data, aes(x='strategy', y='MEM_USAGE_MB_max', fill='algorithm')) +
              geom_boxplot(outlier_shape='o', outlier_size=2, width=0.7, alpha=0.7) +
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
        
        if not gpu_data_filtered.empty:
            # Boxplot GPU: Utilização máxima por execução
            p7 = (ggplot(gpu_data_filtered, aes(x='strategy', y='GPU_USAGE_PERCENTAGE_max', fill='algorithm')) +
                  geom_boxplot(outlier_shape='o', outlier_size=2, width=0.7, alpha=0.7) +
                  labs(title='Distribuição da Utilização de GPU por Estratégia e Algoritmo',
                       x='Estratégia', y='GPU (%)', fill='Algoritmo') +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(10, 6)))
            p7.save(output_dir / 'utilizacao_gpu.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / 'utilizacao_gpu.png'}")

            # Boxplot Memória GPU
            p8 = (ggplot(gpu_data_filtered, aes(x='strategy', y='MEM_USAGE_MB_max', fill='algorithm')) +
                  geom_boxplot(outlier_shape='o', outlier_size=2, width=0.7, alpha=0.7) +
                  labs(title='Distribuição da Utilização de Memória GPU por Estratégia e Algoritmo',
                       x='Estratégia', y='Memória GPU (MB)', fill='Algoritmo') +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(10, 6)))
            p8.save(output_dir / 'utilizacao_memoria_gpu.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / 'utilizacao_memoria_gpu.png'}")

            # Boxplot GPU Power
            p9 = (ggplot(gpu_data_filtered, aes(x='strategy', y='POWER_W_max', fill='algorithm')) +
                  geom_boxplot(outlier_shape='o', outlier_size=2, width=0.7, alpha=0.7) +
                  labs(title='Distribuição do Consumo de Energia GPU por Estratégia e Algoritmo',
                       x='Estratégia', y='Potência (W)', fill='Algoritmo') +
                  theme(axis_text_x=element_text(rotation=45, hjust=1),
                        figure_size=(10, 6)))
            p9.save(output_dir / 'energia_gpu.png', dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / 'energia_gpu.png'}")

            # Boxplot GPU Temperature
            p10 = (ggplot(gpu_data_filtered, aes(x='strategy', y='TEMP_C_max', fill='algorithm')) +
                   geom_boxplot(outlier_shape='o', outlier_size=2, width=0.7, alpha=0.7) +
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
            p_cpu = (
                ggplot(cpu_power_data, aes(x='strategy', y='CPU_POWER_W_median', fill='algorithm')) +
                geom_boxplot(outlier_shape='o', outlier_size=2, width=0.7, alpha=0.7) +
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
            p_io = (
                ggplot(ssd_io_data, aes(x='strategy', y='TOTAL_MB', fill='algorithm')) +
                geom_boxplot(outlier_shape='o', outlier_size=2, width=0.7, alpha=0.7) +
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
    create_resource_plots(cpu_data, gpu_data, output_dir)
    create_resource_by_phase_plots(cpu_by_phase_data, gpu_by_phase_data, time_data, output_dir)
    create_total_time_stacked_plot(time_data, output_dir)
    create_summary_plots(time_data, cpu_data, gpu_data, output_dir)
    create_cpu_power_plot(cpu_power_data, output_dir)
    create_io_ssd_plot(ssd_io_data, output_dir)
    
    print(f"\nTodos os plots foram gerados em: {output_dir}")

if __name__ == '__main__':
    main() 